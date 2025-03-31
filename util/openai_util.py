"""Tools to generate from OpenAI prompts."""

import re
import os
import base64
import asyncio
import logging
from typing import Any
import urllib.request

import aiolimiter
import openai
from tqdm.asyncio import tqdm_asyncio


# Keyless
from azure.identity import get_bearer_token_provider, AzureCliCredential

logging.basicConfig(level=logging.INFO)

azure_credential = AzureCliCredential()


token_provider = get_bearer_token_provider(
    azure_credential, "https://cognitiveservices.azure.com/.default"
)

# Key
endpoint = os.getenv("ENDPOINT_URL", "https://chatgpt-simulation.openai.azure.com/")  


async def _load_image_in_messages(messages: list[dict[str, str]]):
    for m in messages:
        for c in m["content"]:
            if isinstance(c, dict) and c["type"] == "image_url" and isinstance(c["image_url"], str):
                image = c["image_url"]
                _, ext = os.path.splitext(image)
                with open(image, 'rb') as f:
                    image = base64.b64encode(f.read()).decode("utf-8")
                c["image_url"] = {
                  "url": f"data:image/{ext[1:]};base64,{image}"
                }


async def _throttled_openai_chat_completion_acreate(
    client: openai.AsyncAzureOpenAI,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    n: int,
    limiter: aiolimiter.AsyncLimiter,
) -> dict[str, Any]:
    async with limiter:
        await _load_image_in_messages(messages)
        for _ in range(3):
            try:
                return await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    n=n,
                )
            except Exception as e:
                sleep_time = 10
                m = re.search(r"(\d+) second", str(e))
                if m:
                    sleep_time = int(m.group(1))
                logging.info(f"{str(e)}----> sleep {sleep_time} seconds")
                await asyncio.sleep(sleep_time)
        return None


async def generate_from_openai_chat_completion(
    batch_messages: list[list[dict]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    n: int,
    requests_per_minute: int = 150,
) -> list[dict]:
    """Generate from OpenAI Chat Completion API.

    Args:
        batch_messages: List of messages to generate from.
        prompt_template: Prompt template to use.
        model: Model name.
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        n: Number of responses to generate for each API call.
        top_p: Top p to use.
        context_length: Length of context to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """
    client = openai.AsyncAzureOpenAI(
        azure_endpoint=endpoint,
        api_version="2024-05-01-preview",
        #azure_ad_token_provider=token_provider,
        api_key=os.environ['API_KEY'],
    )
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            client=client,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n=n,
            limiter=limiter,
        )
        for messages in batch_messages
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    await client.close()
    return responses


def run_batch_chat_completion(
    all_messages: list[list[dict]],
    batch_size: int,
    retry: int = 10,
    **kwargs,
    ):
    ids = list(range(len(all_messages)))
    all_responses = [None] * len(ids)
    r = 0
    while r < retry:
        r += 1

        ids = [i for i in ids if all_responses[i] is None]
        if len(ids) == 0:
            break

        logging.warning(f"========== run {r}: {len(ids)} unsolved problems ==========")

        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_messages = [all_messages[j] for j in batch_ids] 
            batch_responses = asyncio.run(
                generate_from_openai_chat_completion(
                    batch_messages=batch_messages,
                    **kwargs,
                )
            )
            for i, response in zip(batch_ids, batch_responses):
                all_responses[i] = response

    return all_responses


if __name__ == "__main__":
    """Example of async Azure OpenAI API calls.
    Make sure the following dependencies are installed:
        - pip install tqdm aiolimiter openai
    """
    import json
    import urllib
    import base64
    
    if 'API_KEY' not in os.environ:
        raise ValueError("Please export your API key with the following syntax:\n\nLinux: export API_KEY=\"<YOUR API KEY>\"\n\nWindows: set API_KEY=\"<YOUR API KEY>\"")

    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

    with urllib.request.urlopen(url) as image_file:
        img = base64.b64encode(image_file.read()).decode("utf-8")
    

    all_messages = [
        [{"role": "user", "content": "Write a poem about asynchronous execution."}],
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpg;base64,{img}",
                            "detail": "high"
                        },
                    },
                ],
            }
        ],
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": url,
                            "detail": "high"
                        },
                    },
                ],
            }
        ]
    ]

    all_responses = run_batch_chat_completion(
        all_messages=all_messages,
        batch_size=128,
        model="gpt-4o",
        temperature=0.3,
        max_tokens=200,
        top_p=1.0,
        n=1
    )

    for r in all_responses:
        print(json.dumps(r.to_dict(), indent=2))