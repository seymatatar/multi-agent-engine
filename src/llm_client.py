"""Gemini API için sade, asenkron httpx istemcisi."""
import os
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()  # .env dosyasındaki değişkenleri ortama yükle

API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiClient:
    def __init__(self, model: str | None = None, timeout: float = 60.0):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY bulunamadı. .env dosyasını kontrol et.")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.timeout = timeout

    async def generate(
        self,
        contents: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system_instruction: str | None = None,
    ) -> dict[str, Any]:
        """Gemini'ye tek bir istek atar ve ilk adayın (candidate) içeriğini döndürür."""
        payload: dict[str, Any] = {
            "contents": contents,
            # Modelin düşünce özetlerini de al (terminalde göstermek için)
            "generationConfig": {"thinkingConfig": {"includeThoughts": True}},
        }
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        if tools:
            payload["tools"] = [{"functionDeclarations": tools}]  # Function Calling

        url = f"{API_BASE}/{self.model}:generateContent"
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)

        if resp.status_code != 200:
            raise RuntimeError(f"Gemini API hatası {resp.status_code}: {resp.text}")

        data = resp.json()
        candidates = data.get("candidates")
        if not candidates or "content" not in candidates[0]:
            raise RuntimeError(f"Gemini boş/engellenmiş yanıt döndürdü: {data}")
        return candidates[0]["content"]  # {"role": "model", "parts": [...]}
