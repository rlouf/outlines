"""Integration with Gemini's API."""
__all__ = ["Gemini"]


class Gemini:
    def __init__(self, model_name: str, *args, **kwargs):
        import google.generativeai as genai

        self.client = genai.GenerativeModel(model_name, *args, **kwargs)

    def __call__(self, prompt: str, **inference_kwargs):
        completion = self.client.generate_content(prompt)
        return completion.text
