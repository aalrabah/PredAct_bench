import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # reads /home/alrabah2/PredAct_bench/.env

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_student_risk",
        "description": "Get risk prediction for a student",
        "parameters": {
            "type": "object",
            "properties": {"student_id": {"type": "string"}},
            "required": ["student_id"],
        },
    },
}]

MESSAGES = [
    {"role": "system", "content": "You help instructors. Use the tool when asked."},
    {"role": "user", "content": "Please check the risk for student S1234."},
]

PROVIDERS = [
    {
        "name": "OpenAI gpt-4o-mini",
        "base_url": "https://api.openai.com/v1",
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "model": "gpt-4o-mini",
        "extra_body": {},
    },
    {
        "name": "Together Qwen3.5-9B",
        "base_url": "https://api.together.xyz/v1",
        "api_key": os.environ.get("TOGETHER_API_KEY"),
        "model": "Qwen/Qwen3.5-9B",
        "extra_body": {},
    },
    {
        "name": "OpenRouter Qwen3.5-35B-A3B",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.environ.get("OPENROUTER_API_KEY"),
        "model": "qwen/qwen3.5-35b-a3b",
        "extra_body": {"reasoning": {"enabled": False}},
    },
]

for p in PROVIDERS:
    print(f"\n=== {p['name']} ===")
    if not p["api_key"]:
        print(f"❌ MISSING API KEY in .env")
        continue
    try:
        client = OpenAI(api_key=p["api_key"], base_url=p["base_url"])
        kwargs = dict(
            model=p["model"],
            messages=MESSAGES,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.1,
            max_tokens=400,
        )
        if p["extra_body"]:
            kwargs["extra_body"] = p["extra_body"]
        r = client.chat.completions.create(**kwargs)
        msg = r.choices[0].message
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            print(f"✅ Tool call: {tc.function.name}({tc.function.arguments})")
        else:
            print(f"⚠️ No tool call. Text: {(msg.content or '')[:200]}")
    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {e}")
