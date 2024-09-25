"""Integration with OpenAI's API."""
import json
from typing import Optional, Union

from pydantic import BaseModel

__all__ = ["OpenAI"]


class OpenAI:
    def __init__(self, model_name: str, *args, **kwargs):
        from openai import OpenAI

        self.client = OpenAI(*args, **kwargs)
        self.model_name = model_name

    def __call__(
        self,
        prompt: str,
        output_type: Optional[Union[type[BaseModel], str]] = None,
        **inference_kwargs,
    ):
        if isinstance(output_type, type(BaseModel)):
            result = call_structured_outputs_api(
                self.client,
                self.model_name,
                prompt,
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
                prompt,
                response_format,
                **inference_kwargs,
            )
            return result.content
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
    return completion.choices[0].message


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
