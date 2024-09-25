"""Integration with OpenAI's async API."""
import json
from typing import Optional

from pydantic import BaseModel

__all__ = ["AsyncOpenAI"]


class AsyncOpenAI:
    def __init__(self, model_name: str, *args, **kwargs):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(*args, **kwargs)
        self.model_name = model_name

    async def __call__(
        self,
        prompt: str,
        output_type: Optional[type[BaseModel]] = None,
        **inference_kwargs,
    ):
        if isinstance(output_type, type(BaseModel)):
            result = await call_structured_outputs_api(
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
            result = await call_structured_outputs_api(
                self.client,
                self.model_name,
                prompt,
                response_format,
                **inference_kwargs,
            )
            return result.content
        else:
            return await call_api(
                self.client, self.model_name, prompt, **inference_kwargs
            )


async def call_structured_outputs_api(
    client, model_name, prompt, response_format, **inference_kwargs
):
    completion = await client.beta.chat.completions.parse(
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


async def call_api(client, model_name, prompt, **inference_kwargs):
    completion = await client.chat.completions.create(
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
