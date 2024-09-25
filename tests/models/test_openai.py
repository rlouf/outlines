import io
import json
from dataclasses import dataclass

import PIL
import pytest
import requests
from pydantic import BaseModel
from typing_extensions import TypedDict

from outlines.models.asyncio.openai import AsyncOpenAI
from outlines.models.openai import OpenAI, Vision

MODEL_NAME = "gpt-4o-mini-2024-07-18"


def test_openai_wrong_init_parameters():
    with pytest.raises(TypeError, match="got an unexpected"):
        OpenAI(MODEL_NAME, foo=10)


def test_async_openai_wrong_init_parameters():
    with pytest.raises(TypeError, match="got an unexpected"):
        AsyncOpenAI(MODEL_NAME, foo=10)


def test_openai_wrong_inference_parameters():
    with pytest.raises(TypeError, match="got an unexpected"):
        model = OpenAI(MODEL_NAME)
        model("prompt", foo=10)


@pytest.mark.asyncio
async def test_async_openai_wrong_inference_parameters():
    with pytest.raises(TypeError, match="got an unexpected"):
        model = AsyncOpenAI(MODEL_NAME)
        await model("prompt", foo=10)


@pytest.mark.api_call
def test_openai_simple_call():
    model = OpenAI(MODEL_NAME)
    result = model("Respond with one word. Not more.")
    assert isinstance(result, str)


@pytest.mark.api_call
@pytest.mark.asyncio
async def test_async_openai_simple_call():
    model = AsyncOpenAI(MODEL_NAME)
    result = await model("Respond with one word. Not more.")
    assert isinstance(result, str)


@pytest.mark.api_call
def test_openai_simple_vision():
    model = OpenAI(MODEL_NAME)

    url = "https://raw.githubusercontent.com/dottxt-ai/outlines/refs/heads/main/docs/assets/images/logo.png"
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        image = PIL.Image.open(io.BytesIO(r.content))

    result = model(Vision("What does this logo represent?", image))
    assert isinstance(result, str)


@pytest.mark.api_call
@pytest.mark.asyncio
async def test_async_openai_simple_vision():
    model = AsyncOpenAI(MODEL_NAME)

    url = "https://raw.githubusercontent.com/dottxt-ai/outlines/refs/heads/main/docs/assets/images/logo.png"
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        image = PIL.Image.open(io.BytesIO(r.content))

    result = await model(Vision("What does this logo represent?", image))
    assert isinstance(result, str)


@pytest.mark.api_call
def test_openai_simple_pydantic():
    model = OpenAI(MODEL_NAME)

    class Foo(BaseModel):
        bar: int

    result = model("foo?", Foo)
    assert isinstance(result, BaseModel)


@pytest.mark.api_call
@pytest.mark.asyncio
async def test_async_openai_simple_pydantic():
    model = AsyncOpenAI(MODEL_NAME)

    class Foo(BaseModel):
        bar: int

    result = await model("foo?", Foo)
    assert isinstance(result, BaseModel)


@pytest.mark.api_call
def test_openai_simple_vision_pydantic():
    model = OpenAI(MODEL_NAME)

    url = "https://raw.githubusercontent.com/dottxt-ai/outlines/refs/heads/main/docs/assets/images/logo.png"
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        image = PIL.Image.open(io.BytesIO(r.content))

    class Logo(BaseModel):
        name: int

    result = model(Vision("What does this logo represent?", image), Logo)
    assert isinstance(result, BaseModel)


@pytest.mark.api_call
@pytest.mark.asyncio
async def test_async_openai_simple_vision_pydantic():
    model = AsyncOpenAI(MODEL_NAME)

    url = "https://raw.githubusercontent.com/dottxt-ai/outlines/refs/heads/main/docs/assets/images/logo.png"
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        image = PIL.Image.open(io.BytesIO(r.content))

    class Logo(BaseModel):
        name: int

    result = await model(Vision("What does this logo represent?", image), Logo)
    assert isinstance(result, BaseModel)


@pytest.mark.api_call
def test_openai_simple_json_schema():
    model = OpenAI(MODEL_NAME)

    class Foo(BaseModel):
        bar: int

    schema = json.dumps(Foo.model_json_schema())

    result = model("foo?", schema)
    assert isinstance(result, str)
    json.loads(result)


@pytest.mark.api_call
@pytest.mark.asyncio
async def test_async_openai_simple_json_schema():
    model = AsyncOpenAI(MODEL_NAME)

    class Foo(BaseModel):
        bar: int

    schema = json.dumps(Foo.model_json_schema())

    result = await model("foo?", schema)
    assert isinstance(result, str)
    json.loads(result)


@pytest.mark.xfail(reason="The OpenAI SDK does not support TypedDict as inputs.")
@pytest.mark.api_call
def test_openai_simple_typed_dict():
    model = OpenAI(MODEL_NAME)

    class Foo(TypedDict):
        bar: int

    result = model("foo?", Foo)
    assert isinstance(result, BaseModel)


@pytest.mark.xfail(reason="The OpenAI SDK does not support TypedDict as inputs.")
@pytest.mark.api_call
@pytest.mark.asyncio
async def test_async_openai_simple_typed_dict():
    model = AsyncOpenAI(MODEL_NAME)

    class Foo(TypedDict):
        bar: int

    result = await model("foo?", Foo)
    assert isinstance(result, BaseModel)


@pytest.mark.xfail(reason="The OpenAI SDK does not support dataclasses as inputs.")
@pytest.mark.api_call
def test_openai_simple_dataclass():
    model = OpenAI(MODEL_NAME)

    @dataclass
    class Foo:
        bar: int

    result = model("foo?", Foo)
    assert isinstance(result, BaseModel)


@pytest.mark.xfail(reason="The OpenAI SDK does not support dataclass as inputs.")
@pytest.mark.api_call
@pytest.mark.asyncio
async def test_async_openai_simple_dataclass():
    model = AsyncOpenAI(MODEL_NAME)

    @dataclass
    class Foo:
        bar: int

    result = await model("foo?", Foo)
    assert isinstance(result, BaseModel)
