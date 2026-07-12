import asyncio, sys, base64
sys.path.insert(0,'.')
from industrial.vision.inspector import VisualInspectionEngine
from config.settings import get_settings
s = get_settings()

engine = VisualInspectionEngine(
    ollama_base_url=s.ollama_base_url,
    ollama_vision_model=s.ollama_vision_model,
    gemini_api_key=s.gemini_api_key or None,
    gemini_model=s.gemini_model,
)

# Test with bad image
with open('data/industrial/seed_images/bad/2022_2_1_12_23_10_670.jpg','rb') as f:
    img = f.read()

async def test():
    # Get raw response from llava directly
    import httpx
    img_b64 = base64.b64encode(img).decode()
    prompt = "Is there a defect in this image? Reply ONLY with JSON: {\"verdict\": \"GOOD\" or \"ANOMALY\", \"confidence\": 0.9, \"reason\": \"brief reason\", \"defect_type\": \"crack\" or null, \"affected_part\": \"unknown\", \"box\": null}"
    payload = {"model": s.ollama_vision_model, "prompt": prompt, "images": [img_b64], "stream": False, "options": {"temperature": 0.0}}
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{s.ollama_base_url}/api/generate", json=payload)
        data = resp.json()
        raw = data.get("response","")
        print("RAW RESPONSE:")
        print(repr(raw[:500]))
        print("\nFIRST 200 chars:")
        print(raw[:200])

asyncio.run(test())
