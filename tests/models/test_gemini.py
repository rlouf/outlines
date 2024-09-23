import pytest

from outlines.models.gemini import Gemini

MODEL_NAME = "gemini-1.5-flash-latest"


def test_gemini_wrong_init_parameters():
    with pytest.raises(TypeError, match="got an unexpected"):
        Gemini(MODEL_NAME, foo=10)


@pytest.mark.skip(
    reason="Google does not guard against wrong kwargs passed to the model."
)
def test_gemini_wrong_inference_parameters():
    with pytest.raises(TypeError, match="got an unexpected"):
        model = Gemini(MODEL_NAME)
        model("prompt", foo=10)


@pytest.mark.api_call
def test_gemini_simple_call():
    model = Gemini(MODEL_NAME)
    result = model("Respond with one word. Not more.")
    assert isinstance(result, str)
