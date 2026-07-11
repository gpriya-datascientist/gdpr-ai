import httpx, base64, sys, asyncio
sys.path.insert(0,'.')
from config.settings import get_settings
s = get_settings()
key = s.gemini_api_key
print("Key prefix:", key[:15])

with open('data/industrial/seed_images/bad/2022_2_1_12_23_10_670.jpg','rb') as f:
    img = f.read()
img_b64 = base64.b64encode(img).decode()

prompt = 'Inspect this industrial part image for manufacturing defects. Reply ONLY with valid JSON like this: {"verdict": "ANOMALY", "confidence": 0.9, "reason": "crack visible on surface", "defect_type": "crack"}'

payload = {
    "contents": [{"parts": [
        {"text": prompt},
        {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}
    ]}],
    "generationConfig": {"temperature": 0.0, "maxOutputTokens": 300}
}

url = "https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent"
r = httpx.post(url, json=payload, params={"key": key}, timeout=30)
print("Status:", r.status_code)
if r.status_code == 200:
    data = r.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    print("Response:", text[:400])
else:
    print("Error:", r.text[:300])
