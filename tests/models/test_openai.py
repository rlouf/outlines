import pytest
from pydantic import BaseModel

from outlines.models.openai import JSON, OpenAI

MODEL_NAME = "gpt-4o-mini-2024-07-18"


def test_openai_wrong_init_parameters():
    with pytest.raises(TypeError, match="got an unexpected"):
        OpenAI(MODEL_NAME, foo=10)


def test_openai_wrong_inference_parameters():
    with pytest.raises(TypeError, match="got an unexpected"):
        model = OpenAI(MODEL_NAME)
        model("prompt", foo=10)


@pytest.mark.api_call
def test_openai_simple_call():
    model = OpenAI(MODEL_NAME)
    result = model("Respond with one word. Not more.")
    assert isinstance(result, str)


@pytest.mark.api_call
def test_openai_simple_structured_call():
    model = OpenAI(MODEL_NAME)

    class Foo(BaseModel):
        bar: int

    result = model("foo?", JSON(Foo))
    assert isinstance(result, BaseModel)
