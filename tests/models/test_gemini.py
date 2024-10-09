import io
from enum import Enum

import PIL
import pytest
import requests
from pydantic import BaseModel
from typing_extensions import TypedDict

from outlines.models.gemini import JSON, Gemini, Vision

MODEL_NAME = "gemini-1.5-flash-latest"


def test_gemini_wrong_init_parameters():
    with pytest.raises(TypeError, match="got an unexpected"):
        Gemini(MODEL_NAME, foo=10)


def test_gemini_wrong_inference_parameters():
    with pytest.raises(TypeError, match="got an unexpected"):
        model = Gemini(MODEL_NAME)
        model("prompt", foo=10)


@pytest.mark.api_call
def test_gemini_simple_call():
    model = Gemini(MODEL_NAME)
    result = model("Respond with one word. Not more.")
    assert isinstance(result, str)


@pytest.mark.api_call
def test_gemini_simple_vision():
    model = Gemini(MODEL_NAME)

    url = "https://raw.githubusercontent.com/dottxt-ai/outlines/refs/heads/main/docs/assets/images/logo.png"
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        image = PIL.Image.open(io.BytesIO(r.content))

    result = model(Vision("What does this logo represent?", image))
    assert isinstance(result, str)


@pytest.mark.api_call
def test_gemini_simple_pydantic():
    model = Gemini(MODEL_NAME)

    class Foo(BaseModel):
        bar: int

    result = model("foo?", JSON(Foo))
    assert isinstance(result, BaseModel)


@pytest.mark.xfail(reason="Vision models do not work with structured outputs.")
@pytest.mark.api_call
def test_gemini_simple_vision_pydantic():
    model = Gemini(MODEL_NAME)

    url = "https://raw.githubusercontent.com/dottxt-ai/outlines/refs/heads/main/docs/assets/images/logo.png"
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        image = PIL.Image.open(io.BytesIO(r.content))

    class Logo(BaseModel):
        name: int

    result = model(Vision("What does this logo represent?", image), Logo)
    assert isinstance(result, BaseModel)


@pytest.mark.api_call
def test_gemini_nested_pydantic():
    model = Gemini(MODEL_NAME)

    class Bar(BaseModel):
        fu: str

    class Foo(BaseModel):
        sna: int
        bar: Bar

    result = model("foo?", JSON(Foo))
    assert isinstance(result, BaseModel)
    assert isinstance(result.bar, BaseModel)


@pytest.mark.xfail(
    reason="The Gemini SDK's serialization method does not support JSON Schema dictionaries."
)
@pytest.mark.api_call
def test_gemini_simple_json_schema_dict():
    model = Gemini(MODEL_NAME)

    schema = {
        "properties": {"bar": {"title": "Bar", "type": "integer"}},
        "required": ["bar"],
        "title": "Foo",
        "type": "object",
    }
    result = model("foo?", JSON(schema))
    assert isinstance(result, BaseModel)


@pytest.mark.xfail(
    reason="The Gemini SDK's serialization method does not support JSON Schema strings."
)
@pytest.mark.api_call
def test_gemini_simple_json_schema_string():
    model = Gemini(MODEL_NAME)

    schema = "{'properties': {'bar': {'title': 'Bar', 'type': 'integer'}}, 'required': ['bar'], 'title': 'Foo', 'type': 'object'}"
    result = model("foo?", JSON(schema))
    assert isinstance(result, BaseModel)


@pytest.mark.api_call
def test_gemini_simple_typed_dict():
    model = Gemini(MODEL_NAME)

    class Foo(TypedDict):
        bar: int

    result = model("foo?", JSON(Foo))
    assert isinstance(result, dict)


@pytest.mark.api_call
def test_gemini_simple_enum():
    model = Gemini(MODEL_NAME)

    class Foo(Enum):
        bar = "Bar"
        foor = "Foo"

    result = model("foo?", Foo)
    assert isinstance(result, str)
    assert result == "Foo" or result == "Bar"


@pytest.mark.api_call
def test_gemini_simple_list_pydantic():
    model = Gemini(MODEL_NAME)

    class Foo(BaseModel):
        bar: int

    result = model("foo?", list[JSON(Foo)])
    assert isinstance(result, list)
    assert isinstance(result[0], BaseModel)
