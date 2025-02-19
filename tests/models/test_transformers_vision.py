from enum import Enum

import numpy as np
import pytest
from PIL import Image
from transformers import Blip2ForConditionalGeneration, CLIPModel, CLIPProcessor

from outlines.models.transformers_vision import TransformersVision
from outlines.processors import RegexLogitsProcessor
from outlines.types import Choice

TEST_MODEL = "hf-internal-testing/tiny-random-Blip2Model"
TEST_CLIP_MODEL = "openai/clip-vit-base-patch32"


@pytest.fixture
def image():
    return Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))


@pytest.fixture
def model():
    return TransformersVision(
        model_name=TEST_MODEL, model_class=Blip2ForConditionalGeneration
    )


def test_transformers_vision_instantiate_simple():
    model = TransformersVision(
        model_name=TEST_MODEL,
        model_class=Blip2ForConditionalGeneration,
    )
    assert isinstance(model, TransformersVision)


def test_transformers_vision_instantiate_other_processor_class():
    model = TransformersVision(
        model_name=TEST_CLIP_MODEL, model_class=CLIPModel, processor_class=CLIPProcessor
    )
    assert isinstance(model, TransformersVision)


def test_transformers_vision_simple(model, image):
    result = model.generate(("Describe this image in one sentence:", image), None)
    assert isinstance(result, str)


def test_transformers_vision_call(model, image):
    result = model(("Describe this image in one sentence:", image))
    assert isinstance(result, str)


def test_transformers_vision_wrong_input_type(model):
    with pytest.raises(NotImplementedError):
        model.generate("invalid input", None)


def test_transformers_inference_kwargs(model, image):
    result = model(("Describe this image in one sentence:", image), max_new_tokens=100)
    assert isinstance(result, str)


def test_transformers_invalid_inference_kwargs(model):
    with pytest.raises(ValueError):
        model(("Describe this image in one sentence:", image), foo="bar")


def test_transformers_vision_choice(model, image):
    class Foo(Enum):
        white = "white"
        black = "black"

    regex_str = Choice(Foo).to_regex()
    logits_processor = RegexLogitsProcessor(regex_str, model.tokenizer)
    result = model.generate(("Is this image white or black?", image), logits_processor)

    assert isinstance(result, str)


def test_transformers_vision_batch_input_samples(model, image):
    result = model(("Describe this image in one sentence.", image))
    assert isinstance(result, str)
    result = model(
        ("Describe this image in one sentence.", image),
        num_return_sequences=2,
        num_beams=2,
    )
    assert isinstance(result, list)
    assert len(result) == 2
    result = model(
        (
            [
                "Describe this image in one sentence.",
                "Describe this image in one sentence.",
            ],
            [image, image],
        )
    )
    assert isinstance(result, list)
    assert len(result) == 2
    result = model(
        (
            [
                "Describe this image in one sentence.",
                "Describe this image in one sentence.",
            ],
            [image, image],
        ),
        num_return_sequences=2,
        num_beams=2,
    )
    assert isinstance(result, list)
    assert len(result) == 2
    for item in result:
        assert isinstance(item, list)
        assert len(item) == 2
