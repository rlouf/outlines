"""Integration with Anthropic's API."""
import base64
from io import BytesIO
from typing import Union

from .openai import Vision

__all__ = ["Anthropic"]


class Anthropic:
    def __init__(self, model_name: str, *args, **kwargs):
        from anthropic import Anthropic

        self.client = Anthropic(*args, **kwargs)
        self.model_name = model_name

    def __call__(self, input: Union[str, Vision], max_tokens=1024, **inference_kwargs):
        if isinstance(input, Vision):
            image = input.image
            buffer = BytesIO()
            image.save(buffer, format=image.format)
            image_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            image_format = f"image/{image.format.lower()}"
            content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image_format,
                        "data": image_str,
                    },
                },
                {"type": "text", "text": input.prompt},
            ]
        else:
            content = input

        completion = self.client.messages.create(
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
            model=self.model_name,
            max_tokens=max_tokens,
            **inference_kwargs,
        )
        return completion.content[0].text
