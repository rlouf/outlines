import io

import PIL
import pytest
import requests

from outlines.models.anthropic import Anthropic
from outlines.models.openai import Vision

MODEL_NAME = "claude-3-haiku-20240307"


def test_anthropic_wrong_init_parameters():
    with pytest.raises(TypeError, match="got an unexpected"):
        Anthropic(MODEL_NAME, foo=10)


def test_anthropic_wrong_inference_parameters():
    with pytest.raises(TypeError, match="got an unexpected"):
        model = Anthropic(MODEL_NAME)
        model("prompt", foo=10)


@pytest.mark.api_call
def test_anthropic_simple_call():
    model = Anthropic(MODEL_NAME)
    result = model("Respond with one word. Not more.")
    assert isinstance(result, str)


@pytest.mark.api_call
def test_anthropic_simple_vision():
    model = Anthropic(MODEL_NAME)

    url = "https://raw.githubusercontent.com/dottxt-ai/outlines/refs/heads/main/docs/assets/images/logo.png"
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        image = PIL.Image.open(io.BytesIO(r.content))

    result = model(Vision("What does this logo represent?", image))
    assert isinstance(result, str)
