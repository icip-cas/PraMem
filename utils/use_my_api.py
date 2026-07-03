from openai import OpenAI


class LocalClient:
    def __init__(
        self,
        api_key: str = "EMPTY",
        base_url: str = "http://your-model-endpoint/v1",
        timeout: int = 3600,
        model_name: str = "Qwen3-VL-8B-Instruct",
        max_completion_tokens: int = 4096,
    ):
        self.model_name = model_name
        self.max_completion_tokens = max_completion_tokens
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def call_openai(
        self,
        prompt: str,
        model: str = None,
        max_completion_tokens: int = None,
    ) -> str:

        model = model or self.model_name
        max_completion_tokens = max_completion_tokens or self.max_completion_tokens

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=max_completion_tokens,
        )

        return response.choices[0].message.content, response
