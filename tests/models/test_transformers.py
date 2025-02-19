from enum import Enum

import pytest
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM

from outlines.models.transformers import Mamba, Transformers
from outlines.processors import RegexLogitsProcessor
from outlines.types import Choice, Json, Regex

TEST_MODEL = "hf-internal-testing/tiny-random-GPTJForCausalLM"
TEST_MODEL_SEQ2SEQ = "hf-internal-testing/tiny-random-t5"
TEST_MODEL_MAMBA = "hf-internal-testing/tiny-random-MambaForCausalLM"


def test_transformers_instantiate_simple():
    model = Transformers(TEST_MODEL)
    assert isinstance(model, Transformers)


def test_transformers_instantiate_wrong_kwargs():
    with pytest.raises(TypeError):
        Transformers(TEST_MODEL, model_kwargs={"foo": "bar"})


def test_transformers_instantiate_other_model_class():
    model = Transformers(
        model_name=TEST_MODEL_SEQ2SEQ, model_class=AutoModelForSeq2SeqLM
    )
    assert isinstance(model, Transformers)


def test_transformers_instantiate_mamba():
    model = Mamba(
        model_name=TEST_MODEL_MAMBA,
    )
    assert isinstance(model, Mamba)
    assert isinstance(model, Transformers)


@pytest.fixture
def model():
    return Transformers(TEST_MODEL)


def test_transformers_simple(model):
    result = model.generate("Respond with one word. Not more.", None)
    assert isinstance(result, str)


def test_transformers_call(model):
    result = model("Respond with one word. Not more.")
    assert isinstance(result, str)


def test_transformers_inference_kwargs(model):
    result = model("Respond with one word. Not more.", max_new_tokens=100)
    assert isinstance(result, str)


def test_transformers_invalid_inference_kwargs(model):
    with pytest.raises(ValueError):
        model("Respond with one word. Not more.", foo="bar")


def test_transformers_multiple_input_samples(model):
    result = model("Respond with one word. Not more.")
    assert isinstance(result, str)
    result = model(
        "Respond with one word. Not more.", num_return_sequences=2, num_beams=2
    )
    assert isinstance(result, list)
    assert len(result) == 2
    result = model(
        ["Respond with one word. Not more.", "Respond with one word. Not more."]
    )
    assert isinstance(result, list)
    assert len(result) == 2
    result = model(
        ["Respond with one word. Not more.", "Respond with one word. Not more."],
        num_return_sequences=2,
        num_beams=2,
    )
    assert isinstance(result, list)
    assert len(result) == 2
    for item in result:
        assert isinstance(item, list)
        assert len(item) == 2


def test_transformers_regex(model):
    regex_str = Regex(r"[0-9]").to_regex()
    logits_processor = RegexLogitsProcessor(regex_str, model.tokenizer)
    result = model.generate("Respond with one word. Not more.", logits_processor)
    assert isinstance(result, str)


def test_transformers_json(model):
    class Foo(BaseModel):
        age: int

    regex_str = Json(Foo).to_regex()
    logits_processor = RegexLogitsProcessor(regex_str, model.tokenizer)
    result = model.generate(
        "foo? Respond with one word.", logits_processor, max_new_tokens=100
    )

    assert isinstance(result, str)
    assert "age" in result


def test_transformers_choice(model):
    class Foo(Enum):
        bar = "Bar"
        foor = "Foo"

    regex_str = Choice(Foo).to_regex()
    logits_processor = RegexLogitsProcessor(regex_str, model.tokenizer)
    result = model.generate("foo?", logits_processor)

    assert result == "Foo" or result == "Bar"
