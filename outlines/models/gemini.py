"""Integration with Gemini's API."""
import json
from dataclasses import dataclass
from enum import EnumMeta
from typing import Optional, Union

from pydantic import BaseModel

__all__ = ["Gemini"]


@dataclass
class JSON:
    definition: type[BaseModel]


class Gemini:
    def __init__(self, model_name: str, *args, **kwargs):
        import google.generativeai as genai

        self.client = genai.GenerativeModel(model_name, *args, **kwargs)

    def __call__(
        self,
        prompt: str,
        output_type: Optional[Union[JSON, EnumMeta]] = None,
        **inference_kwargs,
    ):
        import google.generativeai as genai

        generation_config = None
        if isinstance(output_type, JSON):
            generation_config = genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=output_type.definition,
            )
        elif isinstance(output_type, EnumMeta):
            generation_config = genai.GenerationConfig(
                response_mime_type="text/x.enum",
                response_schema=output_type,
            )

        completion = self.client.generate_content(
            prompt, generation_config=generation_config, **inference_kwargs
        )

        if isinstance(output_type, JSON):
            return output_type.definition(**json.loads(completion.text))

        return completion.text
