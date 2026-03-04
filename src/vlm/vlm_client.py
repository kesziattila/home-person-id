"""OpenAI-compatible HTTP client for local VLM (e.g. llama.cpp with Qwen3-VL)."""

import base64
import logging
from typing import Optional

import cv2
import numpy as np

from src.config import VLMConfig

logger = logging.getLogger(__name__)


class VLMClient:
    """HTTP client for OpenAI-compatible VLM endpoints.

    Encodes images as base64 JPEG and sends them as image_url content parts.
    Compatible with llama.cpp server (llama-server --port 8080).
    """

    def __init__(self, config: VLMConfig):
        self._config = config
        self._client = None

    def _get_client(self):
        """Lazily initialize the OpenAI client."""
        if self._client is None:
            from openai import OpenAI

            # Suppress httpx and openai request/response logging — base64 image
            # data in request bodies makes DEBUG logs extremely large.
            logging.getLogger("httpx").setLevel(logging.INFO)
            logging.getLogger("httpcore").setLevel(logging.INFO)
            logging.getLogger("openai").setLevel(logging.INFO)

            self._client = OpenAI(
                base_url=f"{self._config.url}/v1",
                api_key="dummy",  # llama.cpp does not require a real key
            )
        return self._client

    def _encode_image(self, image: np.ndarray, max_size: Optional[int] = None) -> str:
        """Encode a numpy BGR image to a base64 JPEG string, resizing if needed."""
        if max_size is None:
            max_size = self._config.max_image_size
        if max_size > 0:
            h, w = image.shape[:2]
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer.tobytes()).decode("utf-8")

    def call(
        self,
        prompt: str,
        images: Optional[list[np.ndarray]] = None,
        max_tokens: Optional[int] = None,
        image_max_size: Optional[int] = None,
    ) -> Optional[str]:
        """Call the VLM with text prompt and optional images.

        Args:
            prompt: Text instruction for the model
            images: Optional list of numpy BGR images to include
            max_tokens: Override config max_tokens for this call
            image_max_size: Override config max_image_size for encoding images

        Returns:
            Model response text, or None on failure
        """
        content: list[dict] = []

        if images:
            for img in images:
                b64 = self._encode_image(img, max_size=image_max_size)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    }
                )

        content.append({"type": "text", "text": prompt})

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self._config.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens if max_tokens is not None else self._config.max_tokens,
                timeout=self._config.timeout_sec,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"VLM call failed: {e}")
            return None

    def call_text_only(self, prompt: str) -> Optional[str]:
        """Call the VLM with text only (no images).

        Args:
            prompt: Text prompt

        Returns:
            Model response text, or None on failure
        """
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self._config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self._config.max_tokens,
                timeout=self._config.timeout_sec,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"VLM text call failed: {e}")
            return None
