"""Integration with Anthropic's API."""
__all__ = ["Anthropic"]


class Anthropic:
    def __init__(self, model_name: str, *args, **kwargs):
        from anthropic import Anthropic

        self.client = Anthropic(*args, **kwargs)
        self.model_name = model_name

    def __call__(self, prompt: str, max_tokens=1024, **inference_kwargs):
        completion = self.client.messages.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model_name,
            max_tokens=max_tokens,
            **inference_kwargs,
        )
        return completion.content[0].text
