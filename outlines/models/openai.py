"""Integration with OpenAI's API."""
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel

__all__ = ["OpenAI"]


@dataclass
class JSON:
    definition: type[BaseModel]


class OpenAI:
    def __init__(self, model_name: str, *args, **kwargs):
        from openai import OpenAI

        self.client = OpenAI(*args, **kwargs)
        self.model_name = model_name

    def __call__(
        self, prompt: str, output_type: Optional[JSON] = None, **inference_kwargs
    ):
        if isinstance(output_type, JSON):
            response_format = output_type.definition
            return call_structured_outputs_api(
                self.client,
                self.model_name,
                prompt,
                response_format,
                **inference_kwargs,
            )
        else:
            return call_api(self.client, self.model_name, prompt, **inference_kwargs)


def call_structured_outputs_api(
    client, model_name, prompt, response_format, **inference_kwargs
):
    completion = client.beta.chat.completions.parse(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        response_format=response_format,
        **inference_kwargs,
    )
    return completion.choices[0].message.parsed


def call_api(client, model_name, prompt, **inference_kwargs):
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        **inference_kwargs,
    )
    return completion.choices[0].message.content
