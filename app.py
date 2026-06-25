"""
Language Learning App v2 — Japanese (JLPT) & Portuguese (CAPLE)
FastAPI backend with level-based LLM prompts + word lookup.
"""

import os, sys, json, logging
from pathlib import Path
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("langapp")

def _load_hermes_env():
    for env_path in [
        Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes")) / ".env",
        Path.home() / "AppData" / "Local" / "hermes" / ".env",
    ]:
        if env_path.exists():
            with open(env_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, val = line.partition("=")
                        key, val = key.strip(), val.strip().strip('"').strip("'")
                        if key and not os.environ.get(key):
                            os.environ[key] = val
            break
_load_hermes_env()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
API_BASE = "https://api.deepseek.com/v1"
MODEL = "deepseek-chat"

app = FastAPI(title="Language Learning App v2")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

# ═══════════════════════════════════════════════════════════════════════
#  JLPT & CAPLE Level Definitions
# ═══════════════════════════════════════════════════════════════════════

JLPT_LEVELS = {
    "N5": "入门级（N5）：掌握约800个基础词汇，基本问候、数字、时间、家庭成员等。语法限于です/ます体，基本的助词（は、が、を、に、で）。例句用最简单的句型。",
    "N4": "初级（N4）：掌握约1500个词汇，日常会话基础。语法包含て形、た形、ない形、辞書形、可能形等基础活用。例句可使用稍复杂的复合句。",
    "N3": "中级（N3）：掌握约3750个词汇，能理解日常场景中稍复杂的内容。语法包含受身形、使役形、敬语入门、条件形等。例句使用自然流畅的日语。",
    "N2": "中高级（N2）：掌握约6000个词汇，能理解报纸杂志等较复杂内容。语法包含各种高级句型、复合助词、书面语表达。例句贴近真实日本社会场景。",
    "N1": "高级（N1）：掌握约10000个词汇，能理解抽象、学术性内容。使用高度自然的日语，包括谚语、四字熟语、商务日语等。例句体现日语母语者的表达习惯。",
}

CAPLE_LEVELS = {
    "A1": "Iniciante (A1): Vocabulário muito básico — saudações, números, família, objetos cotidianos. Frases muito curtas e simples, apenas presente do indicativo. Vocabulário de alta frequência.",
    "A2": "Elementar (A2): Vocabulário básico para situações cotidianas — compras, transporte, rotina. Frases simples com presente, pretérito perfeito e futuro simples. Expressões frequentes.",
    "B1": "Intermediário (B1): Vocabulário para temas familiares — trabalho, escola, lazer, opiniões. Frases com presente do subjuntivo, pretérito imperfeito. Linguagem mais conectada.",
    "B2": "Intermediário Avançado (B2): Vocabulário mais abstrato — sociedade, cultura, tecnologia. Domínio de todos os tempos verbais, voz passiva, discurso indireto. Expressões idiomáticas comuns.",
    "C1": "Avançado (C1): Vocabulário extenso e preciso — textos acadêmicos, profissionais, literários. Uso natural de expressões idiomáticas, coloquialismos, registros formais e informais.",
    "C2": "Proficiente (C2): Vocabulário próximo ao nativo — nuances, ambiguidades, ironia. Compreensão de todas as sutilezas da língua. Uso de expressões regionais, gírias cultas, linguagem literária.",
}

# ═══════════════════════════════════════════════════════════════════════
#  PROMPT BUILDERS — generate prompts based on language + level
# ═══════════════════════════════════════════════════════════════════════

def build_jp_words_prompt(level: str) -> str:
    desc = JLPT_LEVELS.get(level, JLPT_LEVELS["N3"])
    return f"""あなたは日本語教育の専門家です。JLPT {level} レベルの日本語学習者（中国語話者）向けに、{level}レベルの単語を厳密に10個生成してください。

【レベル要件】
{desc}

【厳守ルール】
- {level}レベルの語彙リスト（JLPT公式出題基準）から単語を選出すること
- すべての単語と例文は**自然な日本語**で書く
- 例文は日本国内で実際に使われる表現にすること
- 例文の文法難易度も{level}に合わせること
- 中国語訳は正確かつ自然な中国語で

出力形式（説明文不要、JSONのみ）:
```json
{{
  "words": [
    {{
      "word": "勉強",
      "reading": "べんきょう",
      "pos": "名詞・スル動詞",
      "meaning_cn": "学习",
      "example_jp": "毎日、図書館で日本語を勉強しています。",
      "example_cn": "我每天在图书馆学习日语。"
    }}
  ]
}}
```

{level}レベルの単語を10個、今すぐ生成してください。毎回異なる単語セットを出力すること。"""


def build_pt_words_prompt(level: str) -> str:
    desc = CAPLE_LEVELS.get(level, CAPLE_LEVELS["B1"])
    return f"""Você é um especialista em ensino de português. Gere 10 palavras em português no nível {level} do QECR/CAPLE para falantes de chinês.

【Requisitos do nível {level}】
{desc}

【Regras obrigatórias】
- Selecione vocabulário adequado ao nível {level}
- Escreva frases de exemplo em português natural
- A complexidade gramatical das frases deve corresponder ao nível {level}
- Indique diferenças regionais (Brasil vs Portugal) em "regional_note" quando relevante
- Tradução para chinês precisa e natural

Formato (APENAS JSON, sem explicações):
```json
{{
  "words": [
    {{
      "word": "saudade",
      "pos": "substantivo feminino",
      "meaning_cn": "思念，怀念",
      "example_pt": "Tenho saudade da minha família.",
      "example_cn": "我思念我的家人。",
      "regional_note": ""
    }}
  ]
}}
```

Gere agora 10 palavras aleatórias no nível {level}. Cada vez gere um conjunto diferente."""


def build_jp_article_prompt(level: str) -> str:
    desc = JLPT_LEVELS.get(level, JLPT_LEVELS["N3"])
    return f"""あなたは日本語のライターです。JLPT {level} レベルの学習者向けに記事を書いてください。

【レベル要件】
{desc}

【ルール】
- {level}レベルの学習者が理解できる語彙・文法で書くこと
- N5/N4向けなら簡単な文型で、N2/N1向けなら自然な日本語で
- テーマ：日本文化、日常生活、ニュース、エッセイ
- N3以上は漢字にふりがな不要。N5/N4は必要に応じて簡単な漢字のみ使用
- 長さ：N5=100〜200字、N4=150〜250字、N3=200〜350字、N2=300〜400字、N1=350〜500字

出力形式（JSONのみ）:
```json
{{
  "title": "記事タイトル",
  "content": "本文...",
  "level": "{level}"
}}
```

今すぐ{level}レベルの短い記事を1本書いてください。"""


def build_pt_article_prompt(level: str) -> str:
    desc = CAPLE_LEVELS.get(level, CAPLE_LEVELS["B1"])
    return f"""Você é um escritor. Escreva um artigo curto em português para estudantes no nível {level} do QECR/CAPLE.

【Requisitos do nível {level}】
{desc}

【Regras】
- Use vocabulário e gramática adequados ao nível {level}
- Temas: cultura, vida cotidiana, notícias leves, crônicas
- Indique no campo "region" se o texto reflete mais o Brasil (BR) ou Portugal (PT)
- Comprimento: A1=60-100 palavras, A2=80-150, B1=120-200, B2=180-300, C1=250-400, C2=300-500

Formato (APENAS JSON):
```json
{{
  "title": "Título do artigo",
  "content": "Texto...",
  "region": "BR",
  "level": "{level}"
}}
```

Escreva agora 1 artigo no nível {level}."""


WORD_LOOKUP_PROMPT = """你是一位__LANG__语言专家。请为以下单词提供详细解释（用中文回答）。

单词：__WORD__
语境（可选）：__CONTEXT__

请按以下格式输出（纯文本，不用JSON）：

【发音】
（日语的用平假名标注读音 / 葡萄牙语的用国际音标或拼音式标注）

【释义】
（中文解释，含多个常用义项）

【用法】
（简要说明该词的使用场景、搭配、语气等）

【例句】
（一个包含该词的自然例句+中文翻译）"""


SUMMARY_PROMPT = """你是一位语言学习助手。请用中文为以下__LANG__文章写一段简洁的概述（100-200字），帮助中文母语者理解文章大意。

要求：用简洁流畅的中文概述文章主要内容；补充1-2个文章中的关键词/短语（附中文解释）；不要逐字翻译，提炼核心信息。

文章标题：__TITLE__
文章内容：
__CONTENT__
请输出概述："""


# ═══════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════

async def call_llm(prompt: str, max_tokens: int = 2000, temperature: float = 0.8) -> str:
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
        return resp.json()["choices"][0]["message"]["content"]


def extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return json.loads(text)


# ═══════════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health():
    return {"status": "ok", "model": MODEL, "has_api_key": bool(DEEPSEEK_API_KEY)}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/generate-words")
async def generate_words(request: Request):
    body = await request.json()
    language = body.get("language", "jp")
    level = body.get("level", "N3" if language == "jp" else "B1")

    if language == "jp":
        prompt = build_jp_words_prompt(level)
    elif language == "pt":
        prompt = build_pt_words_prompt(level)
    else:
        return JSONResponse({"success": False, "error": "Invalid language"}, status_code=400)

    try:
        logger.info(f"Generating words: lang={language} level={level}")
        raw = await call_llm(prompt, max_tokens=3500, temperature=0.9)
        data = extract_json(raw)
        return {"success": True, "data": data, "level": level}
    except Exception as e:
        logger.exception("Failed to generate words")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/api/generate-article")
async def generate_article(request: Request):
    body = await request.json()
    language = body.get("language", "jp")
    level = body.get("level", "N3" if language == "jp" else "B1")

    if language == "jp":
        prompt = build_jp_article_prompt(level)
    elif language == "pt":
        prompt = build_pt_article_prompt(level)
    else:
        return JSONResponse({"success": False, "error": "Invalid language"}, status_code=400)

    try:
        logger.info(f"Generating article: lang={language} level={level}")
        raw = await call_llm(prompt, max_tokens=2500, temperature=0.9)
        data = extract_json(raw)
        return {"success": True, "data": data, "level": level}
    except Exception as e:
        logger.exception("Failed to generate article")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/api/summarize")
async def summarize_article(request: Request):
    body = await request.json()
    language = body.get("language", "jp")
    title = body.get("title", "")
    content = body.get("content", "")

    lang_name = "日语" if language == "jp" else "葡萄牙语"
    prompt = SUMMARY_PROMPT.replace("__LANG__", lang_name).replace("__TITLE__", title).replace("__CONTENT__", content)

    try:
        raw = await call_llm(prompt, max_tokens=800, temperature=0.5)
        return {"success": True, "summary": raw.strip()}
    except Exception as e:
        logger.exception("Failed to summarize")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/api/lookup-word")
async def lookup_word(request: Request):
    """Look up a word's pronunciation, meaning, and usage from an article context."""
    body = await request.json()
    language = body.get("language", "jp")
    word = body.get("word", "")
    context = body.get("context", "")

    lang_name = "日语" if language == "jp" else "葡萄牙语"
    prompt = WORD_LOOKUP_PROMPT.replace("__LANG__", lang_name).replace("__WORD__", word).replace("__CONTEXT__", context or "（无上下文）")

    try:
        raw = await call_llm(prompt, max_tokens=600, temperature=0.3)
        return {"success": True, "lookup": raw.strip()}
    except Exception as e:
        logger.exception("Failed to lookup word")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
