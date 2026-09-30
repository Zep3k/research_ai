from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal
from math import fsum
from typing import Any, Callable

from pydantic import BaseModel

from .config import Config
from .db import CALL_TELEMETRY_COLUMNS, connect, monthly_spend, utcnow
from .errors import BudgetExceededError, ModelOutputError, TheoryError
from .models import ModelResult
from .prompts import Prompt, render_prompt
from .providers import conservative_call_cost


@dataclass
class InvocationBudget:
    """Local allowance charged only for this invocation's metered receipts."""

    cap_usd: float
    call_ids: list[int] = field(default_factory=list)

    def _recorded_costs(self) -> tuple[float, ...]:
        if not self.call_ids:
            return ()
        with connect() as con:
            rows = con.execute(
                f"SELECT cost_usd FROM api_calls WHERE id IN ({','.join('?' for _ in self.call_ids)}) "
                "AND status IN ('completed','failed')", self.call_ids,
            ).fetchall()
        return tuple(row["cost_usd"] for row in rows)

    @property
    def actual_spend_usd(self) -> float:
        return fsum(self._recorded_costs())

    def can_fit(self, conservative_cost_usd: float) -> bool:
        # Decimal comparison preserves admission at an exact dollar boundary;
        # no epsilon may permit a call above the configured cap.
        spent = sum((Decimal(str(cost)) for cost in self._recorded_costs()), Decimal(0))
        return spent + Decimal(str(conservative_cost_usd)) <= Decimal(str(self.cap_usd))


def budget_guard(
    cfg: Config, *, model: str, prompt: Prompt, max_output_tokens: int, purpose: str,
    response_model: type[BaseModel] | None = None,
) -> float:
    estimated_max = conservative_call_cost(
        model, prompt, max_output_tokens, response_model=response_model,
    )
    spent = monthly_spend()
    projected = spent + estimated_max
    if projected > cfg.monthly_budget_usd:
        raise BudgetExceededError(
            f"The {purpose} call cannot fit within the monthly API budget: "
            f"${spent:.4f} spent + up to ${estimated_max:.4f} for this call > "
            f"${cfg.monthly_budget_usd:.2f}."
        )
    return estimated_max


def _start_call(
    *,
    run_id: int | None,
    workstream_id: int | None = None,
    provider: str,
    model: str,
    purpose: str,
    estimated_max_cost_usd: float,
    prompt_utf8_bytes: int,
    planning_metadata: dict | None = None,
) -> int:
    with connect() as con:
        cur = con.execute(
            """
            INSERT INTO api_calls(
                run_id,workstream_id,provider,model,purpose,input_tokens,output_tokens,cost_usd,
                estimated_max_cost_usd,status,error_message,response_text,created_at,prompt_utf8_bytes,
                planning_metadata_json
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                run_id,
                workstream_id,
                provider,
                model,
                purpose,
                0,
                0,
                0.0,
                estimated_max_cost_usd,
                "started",
                None,
                None,
                utcnow(),
                prompt_utf8_bytes,
                json.dumps(planning_metadata, sort_keys=True) if planning_metadata is not None else None,
            ),
        )
        return int(cur.lastrowid)


def _finish_call(
    call_id: int, *, result: ModelResult | None, error: Exception | None = None
) -> None:
    with connect() as con:
        detail_keys = tuple(key for key in CALL_TELEMETRY_COLUMNS if key != "prompt_utf8_bytes")
        details = tuple(getattr(result, key) if result else None for key in detail_keys)
        con.execute(
            f"""
            UPDATE api_calls
            SET input_tokens=?,output_tokens=?,cost_usd=?,status=?,error_message=?,response_text=?,
                {','.join(f'{key}=?' for key in detail_keys)}
            WHERE id=?
            """,
            (
                result.input_tokens if result else 0,
                result.output_tokens if result else 0,
                result.cost_usd if result else 0.0,
                "failed" if error else "completed",
                str(error)[:2000] if error else None,
                result.text if result else None,
                *details,
                call_id,
            ),
        )


def call_model(
    *,
    run_id: int | None,
    workstream_id: int | None = None,
    provider: Any,
    provider_name: str,
    model: str,
    purpose: str,
    prompt: Prompt,
    max_output_tokens: int,
    estimated_max_cost_usd: float,
    response_model: type[BaseModel] | None = None,
    effort: str = "high",
    validate_response: Callable[[str], None] | None = None,
    planning_metadata: dict | None = None,
    on_started: Callable[[int], None] | None = None,
    invocation_budget: InvocationBudget | None = None,
) -> ModelResult:
    call_id = _start_call(
        run_id=run_id,
        workstream_id=workstream_id,
        provider=provider_name,
        model=model,
        purpose=purpose,
        estimated_max_cost_usd=estimated_max_cost_usd,
        prompt_utf8_bytes=len(render_prompt(prompt).encode("utf-8")),
        planning_metadata=planning_metadata,
    )
    if invocation_budget is not None:
        invocation_budget.call_ids.append(call_id)
    try:
        if on_started is not None:
            on_started(call_id)
        result = provider.complete(
            model=model,
            prompt=prompt,
            effort=effort,
            max_output_tokens=max_output_tokens,
            response_model=response_model,
        )
    except Exception as exc:
        _finish_call(call_id, result=None, error=exc)
        raise TheoryError(f"{provider_name} {purpose} call failed: {exc}") from exc
    if result.response_status != "completed":
        provider_label = "OpenAI" if provider_name == "openai" else provider_name
        if result.response_status == "incomplete":
            reason = result.incomplete_reason or "unknown reason"
            detail = (
                "max_output_tokens exhausted"
                if reason == "max_output_tokens"
                else reason
            )
            error = ModelOutputError(
                f"{provider_label} response was incomplete: {detail}."
            )
        else:
            error = ModelOutputError(
                f"{provider_label} response did not complete "
                f"(status: {result.response_status})."
            )
        _finish_call(call_id, result=result, error=error)
        raise error
    if validate_response is not None:
        try:
            validate_response(result.text)
        except Exception as exc:
            _finish_call(call_id, result=result, error=exc)
            raise
    _finish_call(call_id, result=result)
    return result
