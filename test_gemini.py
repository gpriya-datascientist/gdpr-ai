import httpx, base64, sys, asyncio
sys.path.insert(0,'.')
from config.settings import get_settings
s = get_settings()

async def test():
    with open('data/industrial/seed_images/bad/2023_8_3_13_29_52_699.jpg','rb') as f:
        img = f.read()
    img_b64 = base64.b64encode(img).decode()
    prompt = '''You are a strict visual quality inspector for BSH industrial metal press tools.
GOOD: smooth curved metal, bright glare/reflections are NORMAL, parallel machining lines are NORMAL.
ANOMALY: dark irregular crack, fracture, or tear that visibly BREAKS the smooth metal surface.
KEY: Bright white = light reflection = GOOD. Dark jagged break in metal = ANOMALY.
Respond ONLY with valid JSON: {"verdict": "GOOD" or "ANOMALY", "confidence": 0.95, "reason": "max 20 words", "defect_type": "crack" or null}'''

    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent"
    payload = {"contents":[{"parts":[{"text":prompt},{"inline_data":{"mime_type":"image/jpeg","data":img_b64}}]}],
               "generationConfig":{"temperature":0.0,"maxOutputTokens":200}}
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, json=payload, params={"key": s.gemini_api_key})
    print("Status:", r.status_code)
    if r.status_code == 200:
        print("Response:", r.json()["candidates"][0]["content"]["parts"][0]["text"])
    else:
        print("Error:", r.text[:300])

asyncio.run(test())
