"""
Language Learning App — Japanese & Portuguese
FastAPI backend with embedded DeepSeek LLM agent prompts.
Serves a single-page frontend for word/article-based language learning.
"""

import os
import sys
import json
import logging
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ── Logging (stderr so it shows alongside uvicorn logs) ──────────────
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("langapp")

# ── Auto-source DeepSeek API key from Hermes .env ────────────────────
def _load_hermes_env():
    candidates = [
        Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes")) / ".env",
        Path.home() / "AppData" / "Local" / "hermes" / ".env",
    ]
    for env_path in candidates:
        if env_path.exists():
            with open(env_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, val = line.partition("=")
                        key = key.strip()
                        val = val.strip().strip('"').strip("'")
                        if key and not os.environ.get(key):
                            os.environ[key] = val
            logger.info(f"Loaded env from {env_path}")
            break

_load_hermes_env()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
API_BASE = "https://api.deepseek.com/v1"
MODEL = "deepseek-chat"

app = FastAPI(title="Language Learning App", version="1.0.0")

# ── Templates ────────────────────────────────────────────────────────
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

# ═══════════════════════════════════════════════════════════════════════
#  LLM AGENT PROMPTS
# ═══════════════════════════════════════════════════════════════════════

JAPANESE_WORDS_PROMPT = """あなたは日本語教育の専門家です。日本語学習者（中国語話者）のために、初級〜中級レベルの日本語単語を10個生成してください。

【厳守ルール】
- すべての単語と例文は**自然な日本語**で書いてください
- 例文は日本国内で実際に使われる自然な表現にしてください
- 中国語訳は正確かつ自然な中国語で

出力形式（必ずこのJSON形式で返してください。説明文は一切不要です）:

```json
{
  "words": [
    {
      "word": "勉強",
      "reading": "べんきょう",
      "pos": "名詞・スル動詞",
      "meaning_cn": "学习",
      "example_jp": "毎日、図書館で日本語を勉強しています。",
      "example_cn": "我每天在图书馆学习日语。"
    }
  ]
}
```

上の形式に従い、今すぐランダムな初級〜中級レベルの日本語単語を10個生成してください。毎回異なる単語セットを出力すること。"""

PORTUGUESE_WORDS_PROMPT = """Você é um especialista em ensino de português para falantes de chinês.
Gere 10 palavras em português (nível iniciante a intermediário) com frases de exemplo naturais.

【Regras obrigatórias】
- Escreva todas as palavras e frases de exemplo em **português natural**
- As frases de exemplo devem refletir o uso cotidiano no Brasil (ou indique quando houver diferença significativa entre português brasileiro e europeu)
- A tradução para chinês deve ser precisa e natural
- Se houver diferença regional (Brasil vs Portugal), inclua uma nota em "regional_note"

Formato de saída (retorne APENAS este JSON, sem explicações):

```json
{
  "words": [
    {
      "word": "saudade",
      "pos": "substantivo feminino",
      "meaning_cn": "思念，怀念",
      "example_pt": "Tenho saudade da minha família.",
      "example_cn": "我思念我的家人。",
      "regional_note": ""
    },
    {
      "word": "café da manhã",
      "pos": "locução substantiva",
      "meaning_cn": "早餐",
      "example_pt": "O café da manhã está pronto!",
      "example_cn": "早餐准备好了！",
      "regional_note": "Em Portugal: 'pequeno-almoço'"
    }
  ]
}
```

Gere agora 10 palavras aleatórias em português (nível iniciante a intermediário). Cada vez deve gerar um conjunto diferente."""

JAPANESE_ARTICLE_PROMPT = """あなたは日本語のライターです。日本語学習者（初級〜中級）向けに、自然な日本語で短い記事を書いてください。

【厳守ルール】
- **自然な日本語**で書くこと（教科書的な不自然な表現は避ける）
- 日本の文化、日常生活、ニュース、エッセイなど、実用的なテーマ
- 漢字には**ふりがなを振らない**（学習者向けだが、自然な文章として）
- 長さ：200〜400文字程度
- 段落分けをすること

出力形式（必ずこのJSON形式で返してください。説明文は一切不要です）:

```json
{
  "title": "記事のタイトル",
  "content": "記事本文（段落分けあり、自然な日本語）..."
}
```

今すぐ日本語の短い記事を1本書いてください。毎回異なるテーマ・内容にすること。"""

PORTUGUESE_ARTICLE_PROMPT = """Você é um escritor brasileiro. Escreva um artigo curto em português natural para estudantes de português (nível iniciante a intermediário).

【Regras obrigatórias】
- Escreva em **português natural** (evite linguagem de livro didático)
- Temas: cultura brasileira/portuguesa, vida cotidiana, notícias leves, crônicas
- Comprimento: 150-300 palavras
- Use parágrafos
- Indique no campo "region" se o texto é mais típico do Brasil (BR) ou de Portugal (PT)

Formato de saída (retorne APENAS este JSON, sem explicações):

```json
{
  "title": "Título do artigo",
  "content": "Texto do artigo com parágrafos...",
  "region": "BR"
}
```

Escreva agora 1 artigo curto em português. Cada vez deve gerar um tema diferente."""

SUMMARY_PROMPT = """你是一位语言学习助手。请用中文为以下__LANG__文章写一段简洁的概述（100-200字），帮助中文母语者理解文章大意。

要求：
- 用简洁流畅的中文概述文章主要内容
- 可以补充1-2个文章中的关键词/短语（附中文解释）
- 不要逐字翻译，而是提炼核心信息

文章标题：__TITLE__

文章内容：
__CONTENT__

请输出概述："""

# ═══════════════════════════════════════════════════════════════════════
#  LLM CALL HELPER
# ═══════════════════════════════════════════════════════════════════════

async def call_llm(prompt: str, max_tokens: int = 2000, temperature: float = 0.8) -> str:
    """Call DeepSeek API and return response text."""
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY not configured")

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{API_BASE}/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


def extract_json(text: str) -> dict:
    """Extract JSON from LLM response (may contain markdown code fences)."""
    text = text.strip()
    # Remove markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return json.loads(text)

# ═══════════════════════════════════════════════════════════════════════
#  API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL,
        "has_api_key": bool(DEEPSEEK_API_KEY),
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/generate-words")
async def generate_words(request: Request):
    """Generate 10 words for the given language."""
    body = await request.json()
    language = body.get("language", "jp")

    if language == "jp":
        prompt = JAPANESE_WORDS_PROMPT
    elif language == "pt":
        prompt = PORTUGUESE_WORDS_PROMPT
    else:
        return JSONResponse({"success": False, "error": "Invalid language"}, status_code=400)

    try:
        logger.info(f"Generating words for {language}...")
        raw = await call_llm(prompt, max_tokens=3000, temperature=0.9)
        data = extract_json(raw)
        return {"success": True, "data": data}
    except Exception as e:
        logger.exception(f"Failed to generate words: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/api/generate-article")
async def generate_article(request: Request):
    """Generate an article for the given language."""
    body = await request.json()
    language = body.get("language", "jp")

    if language == "jp":
        prompt = JAPANESE_ARTICLE_PROMPT
    elif language == "pt":
        prompt = PORTUGUESE_ARTICLE_PROMPT
    else:
        return JSONResponse({"success": False, "error": "Invalid language"}, status_code=400)

    try:
        logger.info(f"Generating article for {language}...")
        raw = await call_llm(prompt, max_tokens=2000, temperature=0.9)
        data = extract_json(raw)
        return {"success": True, "data": data}
    except Exception as e:
        logger.exception(f"Failed to generate article: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/api/summarize")
async def summarize_article(request: Request):
    """Summarize an article in Chinese."""
    body = await request.json()
    language = body.get("language", "jp")
    title = body.get("title", "")
    content = body.get("content", "")

    lang_name = "日语" if language == "jp" else "葡萄牙语"
    prompt = SUMMARY_PROMPT.replace("__LANG__", lang_name).replace("__TITLE__", title).replace("__CONTENT__", content)

    try:
        logger.info(f"Summarizing article...")
        raw = await call_llm(prompt, max_tokens=800, temperature=0.5)
        return {"success": True, "summary": raw.strip()}
    except Exception as e:
        logger.exception(f"Failed to summarize: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
