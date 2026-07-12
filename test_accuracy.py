import sys, asyncio
sys.path.insert(0,'.')
from industrial.vision.inspector import VisualInspectionEngine
from config.settings import get_settings
s = get_settings()
engine = VisualInspectionEngine(ollama_base_url=s.ollama_base_url, ollama_vision_model=s.ollama_vision_model)

async def test():
    results = []
    import os
    good_dir = 'data/industrial/seed_images/good'
    bad_dir  = 'data/industrial/seed_images/bad'
    for f in os.listdir(good_dir)[:4]:
        with open(f'{good_dir}/{f}','rb') as fh: img=fh.read()
        r = await engine.inspect(img)
        results.append(('GOOD_IMAGE', f, r.verdict, r.confidence))
    for f in os.listdir(bad_dir)[:4]:
        with open(f'{bad_dir}/{f}','rb') as fh: img=fh.read()
        r = await engine.inspect(img)
        results.append(('BAD_IMAGE', f, r.verdict, r.confidence))
    correct = sum(1 for label,_,v,_ in results if (label=='GOOD_IMAGE' and v=='GOOD') or (label=='BAD_IMAGE' and v=='ANOMALY'))
    print(f"\nResults ({correct}/{len(results)} correct):")
    for label,fname,verdict,conf in results:
        ok = (label=='GOOD_IMAGE' and verdict=='GOOD') or (label=='BAD_IMAGE' and verdict=='ANOMALY')
        print(f"  {'OK' if ok else 'WRONG'} | {label} | {fname[:30]} | predicted={verdict} | conf={conf:.0%}")

asyncio.run(test())
