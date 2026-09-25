from __future__ import annotations

from typing import Any

from .config import Config
from .db import connect, monthly_spend, utcnow
from .errors import BudgetExceededError, TheoryError
from .models import ModelResult
from .providers import conservative_call_cost


def budget_guard(
    cfg: Config, *, model: str, prompt: str, max_output_tokens: int, purpose: str
) -> float:
    estimated_max = conservative_call_cost(model, prompt, max_output_tokens)
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
) -> int:
    with connect() as con:
        cur = con.execute(
            """
            INSERT INTO api_calls(
                run_id,workstream_id,provider,model,purpose,input_tokens,output_tokens,cost_usd,
                estimated_max_cost_usd,status,error_message,response_text,created_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
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
            ),
        )
        return int(cur.lastrowid)


def _finish_call(
    call_id: int, *, result: ModelResult | None, error: Exception | None = None
) -> None:
    with connect() as con:
        con.execute(
            """
            UPDATE api_calls
            SET input_tokens=?,output_tokens=?,cost_usd=?,status=?,error_message=?,response_text=?
            WHERE id=?
            """,
            (
                result.input_tokens if result else 0,
                result.output_tokens if result else 0,
                result.cost_usd if result else 0.0,
                "completed" if result else "failed",
                str(error)[:2000] if error else None,
                result.text if result else None,
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
    prompt: str,
    max_output_tokens: int,
    estimated_max_cost_usd: float,
) -> ModelResult:
    call_id = _start_call(
        run_id=run_id,
        workstream_id=workstream_id,
        provider=provider_name,
        model=model,
        purpose=purpose,
        estimated_max_cost_usd=estimated_max_cost_usd,
    )
    try:
        result = provider.complete(
            model=model,
            prompt=prompt,
            effort="high",
            max_output_tokens=max_output_tokens,
        )
    except Exception as exc:
        _finish_call(call_id, result=None, error=exc)
        raise TheoryError(f"{provider_name} {purpose} call failed: {exc}") from exc
    _finish_call(call_id, result=result)
    return result
