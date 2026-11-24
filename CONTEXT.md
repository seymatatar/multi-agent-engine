# Proje: multi-agent-engine
- Amaç: UI'dan bağımsız, saf Python ile geliştirilen asenkron multi-agent motoru.
- Teknolojiler: Python 3.13, httpx (async), pydantic, python-dotenv, pytest-asyncio.
- LLM: Gemini API (anahtar .env dosyasındaki GEMINI_API_KEY üzerinden okunur).
- Klasör Yapısı: Tüm kaynak kodlar `src/` klasörü altında modüler olarak yer alır.
- Hedef Giriş Kapısı: `async run_agent(prompt)` fonksiyonu.
## Mevcut Dosyalar
- `src/llm_client.py`: `.env` dosyasından `GEMINI_API_KEY` okuyan async `httpx` Gemini istemcisi.
- `src/agent.py`: `AgentState`, `max_iterations=5` ve örnek `toplama_yap` aracıyla çalışan ilk ReAct döngüsü.
