"""Planla -> Araç -> Gözlem -> Karar döngüsü. Çalıştır: python -m src.agent"""
import asyncio
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError

from .llm_client import GeminiClient

MAX_ITERATIONS = 5  # sonsuz döngü koruması

SYSTEM_PROMPT = (
    "Sen yardımcı bir ajansın. Matematik işlemleri için mutlaka araçları kullan. "
    "Yeterli bilgiye ulaştığında araç çağırmadan kısa ve net nihai cevabı ver."
)


# ---------- Araçlar ----------
class ToplamaArgs(BaseModel):
    a: float
    b: float


def toplama_yap(a: float, b: float) -> float:
    return a + b


# Araç kayıt defteri: isim -> (fonksiyon, argüman şeması)
TOOLS: dict[str, tuple[Callable[..., Any], type[BaseModel]]] = {
    "toplama_yap": (toplama_yap, ToplamaArgs),
}

# Gemini'ye gönderilen Function Calling tanımları
TOOL_DECLARATIONS = [
    {
        "name": "toplama_yap",
        "description": "İki sayıyı toplar.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "a": {"type": "NUMBER", "description": "Birinci sayı"},
                "b": {"type": "NUMBER", "description": "İkinci sayı"},
            },
            "required": ["a", "b"],
        },
    }
]


# ---------- Durum ----------
class AgentState(BaseModel):
    prompt: str
    messages: list[dict[str, Any]] = Field(default_factory=list)  # Gemini "contents"
    iteration: int = 0
    final_answer: str | None = None
    status: str = "running"  # running | done | max_iterations


def run_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Aracı çalıştırır; hata olursa modele hata mesajını döndürür (çökmez)."""
    if name not in TOOLS:
        return {"error": f"Bilinmeyen araç: {name}"}
    func, schema = TOOLS[name]
    try:
        valid = schema(**args)  # pydantic ile argüman doğrulama
        return {"result": func(**valid.model_dump())}
    except (ValidationError, Exception) as e:
        return {"error": str(e)}


async def run_agent(prompt: str) -> str:
    client = GeminiClient()
    state = AgentState(prompt=prompt)
    state.messages.append({"role": "user", "parts": [{"text": prompt}]})
    print(f"\n=== Görev: {prompt} ===")

    while state.iteration < MAX_ITERATIONS:
        state.iteration += 1
        print(f"\n--- Adım {state.iteration}/{MAX_ITERATIONS} ---")

        # 1) PLANLA: modele sor
        content = await client.generate(state.messages, TOOL_DECLARATIONS, SYSTEM_PROMPT)
        # Modelin cevabını olduğu gibi geçmişe ekle (thoughtSignature korunur)
        state.messages.append(content)
        parts = content.get("parts", [])

        calls: list[dict[str, Any]] = []
        texts: list[str] = []
        for p in parts:
            if p.get("thought") and p.get("text"):
                print(f"[Düşünce] {p['text'].strip()}")
            elif "functionCall" in p:
                calls.append(p["functionCall"])
            elif p.get("text"):
                texts.append(p["text"])

        # 4) KARAR: araç çağrısı yoksa nihai cevap budur
        if not calls:
            state.final_answer = "".join(texts).strip()
            state.status = "done"
            print(f"[Karar] Nihai cevap: {state.final_answer}")
            return state.final_answer

        # 2) ARAÇ + 3) GÖZLEM
        response_parts = []
        for call in calls:
            name, args = call["name"], call.get("args", {})
            print(f"[Araç] {name}({args})")
            result = run_tool(name, args)
            print(f"[Gözlem] {result}")
            response_parts.append(
                {"functionResponse": {"name": name, "response": result}}
            )
        # Araç sonuçlarını modele geri ver, döngü devam eder
        state.messages.append({"role": "user", "parts": response_parts})

    state.status = "max_iterations"
    state.final_answer = "Maksimum iterasyona ulaşıldı, görev tamamlanamadı."
    print(f"[Karar] {state.final_answer}")
    return state.final_answer


if __name__ == "__main__":
    asyncio.run(run_agent("3 elmama var 5 ini yedim"))
