import pytest

from theory.errors import ModelOutputError
from theory.jsonutil import parse_json_model
from theory.models import Formalization


def test_parse_fenced_json():
    text = '''```json
    {"precise_question":"Q?","assumptions_to_pin_down":["A"],"search_queries":["q"],"possible_variants":[],"immediate_failure_modes":[]}
    ```'''
    x = parse_json_model(text, Formalization)
    assert x.precise_question == "Q?"
    assert x.search_queries == ["q"]


def test_parse_json_surrounded_by_prose():
    text = 'Result follows: {"precise_question":"Q?","search_queries":[]} done.'
    assert parse_json_model(text, Formalization).precise_question == "Q?"


def test_parse_rejects_wrong_schema_and_extra_fields():
    with pytest.raises(ModelOutputError, match="did not match"):
        parse_json_model('{"precise_question":"Q?","unexpected":true}', Formalization)


def test_parse_rejects_non_json():
    with pytest.raises(ModelOutputError, match="usable JSON"):
        parse_json_model("I declined to produce JSON", Formalization)
