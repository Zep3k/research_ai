import json
import re
from typing import TypeVar, Type

from pydantic import BaseModel, ValidationError

from .errors import ModelOutputError

T = TypeVar("T", bound=BaseModel)


def parse_json_model(text: str, model: Type[T]) -> T:
    cleaned = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1)
    try:
        return model.model_validate_json(cleaned)
    except (ValidationError, ValueError):
        pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        try:
            payload, _ = decoder.raw_decode(cleaned[match.start():])
        except json.JSONDecodeError:
            continue
        try:
            return model.model_validate(payload)
        except ValidationError as exc:
            raise ModelOutputError(
                f"Model JSON did not match {model.__name__}: {exc}"
            ) from exc

    preview = cleaned[:500].replace("\n", " ")
    raise ModelOutputError(f"Model did not return a usable JSON object: {preview}")
