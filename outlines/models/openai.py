"""Integration with OpenAI's API."""
__all__ = ["OpenAI"]


class OpenAI:
    def __init__(self, model_name: str, *args, **kwargs):
        from openai import OpenAI

        self.client = OpenAI(*args, **kwargs)
        self.model_name = model_name

    def __call__(self, prompt: str, **inference_kwargs):
        completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model_name,
            **inference_kwargs,
        )
        return completion.choices[0].message.content
