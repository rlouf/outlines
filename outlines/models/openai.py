"""Integration with OpenAI's API."""
import base64
import json
from dataclasses import dataclass
from io import BytesIO
from typing import Optional, Union

from PIL import Image
from pydantic import BaseModel

__all__ = ["OpenAI"]


@dataclass
class Vision:
    prompt: str
    image: Image.Image


class OpenAI:
    def __init__(self, model_name: str, *args, **kwargs):
        from openai import OpenAI

        self.client = OpenAI(*args, **kwargs)
        self.model_name = model_name

    def __call__(
        self,
        input: Union[str, Vision],
        output_type: Optional[Union[type[BaseModel], str]] = None,
        **inference_kwargs,
    ):
        if isinstance(output_type, type(BaseModel)):
            result = call_structured_outputs_api(
                self.client,
                self.model_name,
                input,
                output_type,
                **inference_kwargs,
            )
            return result.parsed
        elif isinstance(output_type, str):
            output_type = json.loads(output_type)

            # OpenAI requires `additionalProperties` to be set
            if "additionalProperties" not in output_type:
                output_type["additionalProperties"] = False

            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "default",
                    "strict": True,
                    "schema": output_type,
                },
            }
            result = call_structured_outputs_api(
                self.client,
                self.model_name,
                input,
                response_format,
                **inference_kwargs,
            )
            return result.content
        else:
            return call_api(self.client, self.model_name, input, **inference_kwargs)


def call_structured_outputs_api(
    client,
    model_name,
    input: Union[str, Vision],
    response_format,
    **inference_kwargs,
):
    if isinstance(input, Vision):
        image = input.image
        buffer = BytesIO()
        image.save(buffer, format=image.format)
        image_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        image_format = f"image/{image.format.lower()}"
        content = [
            {"type": "text", "text": input.prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{image_format};base64,{image_str}"},
            },
        ]
    else:
        content = input

    completion = client.beta.chat.completions.parse(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        response_format=response_format,
        **inference_kwargs,
    )
    return completion.choices[0].message


def call_api(client, model_name, input: Union[str, Vision], **inference_kwargs):
    if isinstance(input, Vision):
        image = input.image
        buffer = BytesIO()
        image.save(buffer, format=image.format)
        image_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        image_format = f"image/{image.format.lower()}"
        content = [
            {"type": "text", "text": input.prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{image_format};base64,{image_str}"},
            },
        ]
    else:
        content = input

    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        **inference_kwargs,
    )
    return completion.choices[0].message.content
