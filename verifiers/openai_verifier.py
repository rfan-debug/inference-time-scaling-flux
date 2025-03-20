import base64
import io
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
import requests
import PIL.Image
from PIL.Image import Image as Pimage

from verifiers.base_verifier import BaseVerifier
from verifiers.shared import script_dir, Grading

MAX_SIZE = 2000


def load_verifier_prompt(path: str) -> str:
    with open(path, "r") as f:
        verifier_prompt = f.read().replace('"""', "")

    return verifier_prompt


def url2pil(url: str) -> Pimage:
    # Send a GET request to fetch the image data
    response = requests.get(url)
    response.raise_for_status()  # Raises an error if the request failed
    img = PIL.Image.open(io.BytesIO(response.content))
    return img


def pil2base64str(pil_image: Pimage, format="JPEG") -> str:
    width, height = pil_image.size
    if height > MAX_SIZE or width > MAX_SIZE:
        pil_image.thumbnail((MAX_SIZE, MAX_SIZE))

    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format=format)
    img_byte_arr = img_byte_arr.getvalue()
    return base64.b64encode(img_byte_arr).decode("utf-8")


class OpenAIVerifier(BaseVerifier):

    def __init__(self,
                 seed: int = 1994,
                 model_name: str = "gpt-4o",
                 max_num_workers: int = 4):

        self.client = openai.OpenAI()
        self.model_name = model_name
        self.system_instruction = load_verifier_prompt(os.path.join(script_dir, "oai_verifier_prompt.txt"))
        self.seed = seed
        self.max_num_workers = max_num_workers

    def prepare_inputs(self,
                       images: list[Pimage | str] | (Pimage | str),
                       prompts: list[str] | str) -> list:
        """Prepare inputs for the API from a given prompt and image."""
        images = images if isinstance(images, list) else [images]
        prompts = prompts if isinstance(prompts, list) else [prompts]

        content_list = []
        # This list is intended to interleave in the following way:
        # [prompt1, image1, prompt2, image2, prompt3, image3, ...]

        for prompt, image in zip(prompts, images):
            if isinstance(image, str) and image.startswith("http"):
                input_image = url2pil(image)
            elif isinstance(image, Pimage):
                input_image = image
            else:
                raise RuntimeError(f"Invalid image type: {type(image)}.")

            base64_image = pil2base64str(input_image)
            image_content = {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "high"
            }
            content = [
                {
                    "type": "text",
                    "text": prompt,
                },
                {
                    "type": "image_url",
                    "image_url": image_content,
                },
            ]
            content_list.extend(content)

        return content_list

    def score(self, inputs, max_new_tokens: int = 3000) -> list[Grading]:

        def call_generate_content(content_list) -> Grading:
            messages = [
                {
                    "role": "system",
                    "content": [{
                        "type": "text",
                        "text": self.system_instruction,
                    }]
                },
                {
                    "role": "user",
                    "content": content_list
                }
            ]
            image_chat_response = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=messages,
                response_format=Grading,
                max_completion_tokens=max_new_tokens,
            )

            return image_chat_response.choices[0].message.parsed

        grouped_inputs = [inputs[i: i + 2] for i in range(0, len(inputs), 2)]
        results = []
        max_workers = len(grouped_inputs)
        if max_workers > self.max_num_workers:
            max_workers = self.max_num_workers
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(call_generate_content, group) for group in grouped_inputs]
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    # Handle exceptions as appropriate.
                    print(f"An error occurred during API call: {e}")
        return results
