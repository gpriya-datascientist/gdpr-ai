import sys
sys.path.insert(0,'.')
from industrial.vision.inspector import VisualInspectionEngine

# Test parse directly with the exact raw output llava gives
raw = ' ```json\n\n{\n  "verdict": "GOOD",\n  "confidence": 0.9,\n  "reason": "test",\n  "defect_type": null,\n  "affected_part": null,\n  "box": null\n}\n``` '

print("Testing parse_response...")
result = VisualInspectionEngine._parse_response(raw, 'local', 100.0)
print("Result:", result)
if result:
    print("Verdict:", result.verdict)
    print("Confidence:", result.confidence)
else:
    print("ERROR: Result is None!")

# Now test with bad image
print("\nTesting with bad image...")
import asyncio, base64

async def test_inspect():
    engine = VisualInspectionEngine(
        ollama_base_url='http://localhost:11434',
        ollama_vision_model='llava:7b',
    )
    with open('data/industrial/seed_images/bad/2022_2_1_12_23_10_670.jpg','rb') as f:
        img = f.read()
    result = await engine.inspect(img)
    print("Inspect result:", result)
    print("Verdict:", result.verdict if result else "NONE - result is None!")
    return result

asyncio.run(test_inspect())
