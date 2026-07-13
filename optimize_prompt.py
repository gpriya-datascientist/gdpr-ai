"""
PressVision Optimization Loop for VaultMind
Automatically improves the inspection prompt by testing against labeled images.
Adapted from PressVisionLoop (AGPL-3.0).
"""
import asyncio
import base64
import json
import os
import sys
import httpx

sys.path.insert(0, '.')
from config.settings import get_settings

GOOD_DIR = 'data/industrial/seed_images/good'
BAD_DIR  = 'data/industrial/seed_images/bad'
MAX_ITERATIONS = 8
TARGET_ACCURACY = 0.875  # 87.5% = 7/8 correct

OPTIMIZER_SYSTEM = """You are an expert at writing vision model prompts for industrial defect detection.
You will be given a prompt and its results on labeled images.
Your job is to rewrite the prompt to improve accuracy — reduce false positives (calling GOOD images ANOMALY)
and false negatives (calling BAD images GOOD).
Return ONLY the new prompt text, nothing else."""


s = get_settings()

def load_images(folder, label):
    images = []
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            with open(os.path.join(folder, f), 'rb') as fh:
                images.append({'label': label, 'name': f, 'data': fh.read()})
    return images

async def test_prompt(prompt_template, images):
    results = []
    for img in images:
        prompt = prompt_template.replace('{part_name}', 'BSH Metal Press Tool')
        prompt = prompt.replace('{description}', 'Check for cracks and deformations.')
        img_b64 = base64.b64encode(img['data']).decode()
        payload = {"model": s.ollama_vision_model, "prompt": prompt,
                   "images": [img_b64], "stream": False, "options": {"temperature": 0.0}}
        try:
            async with httpx.AsyncClient(timeout=90) as client:
                resp = await client.post(f"{s.ollama_base_url}/api/generate", json=payload)
                raw = resp.json().get("response", "").strip()
            verdict = extract_verdict(raw)
        except Exception as e:
            print(f"  Error: {e}")
            verdict = "GOOD"
        correct = (img['label']=='GOOD' and verdict=='GOOD') or (img['label']=='BAD' and verdict=='ANOMALY')
        results.append({'name': img['name'], 'label': img['label'], 'verdict': verdict, 'correct': correct})
        print(f"  [{'OK' if correct else 'WRONG'}] {img['label']} | {img['name'][:35]} | predicted={verdict}")
    score = sum(1 for r in results if r['correct']) / len(results)
    return score, results

def extract_verdict(raw):
    import re
    clean = raw.strip()
    if '```' in clean:
        parts = clean.split('```')
        for p in parts:
            p = p.strip()
            if p.startswith('json'): p = p[4:].strip()
            if '"verdict"' in p: clean = p; break
    match = re.search(r'"verdict"\s*:\s*"(\w+)"', clean, re.IGNORECASE)
    if match:
        v = match.group(1).upper()
        return 'ANOMALY' if 'ANOMALY' in v else 'GOOD'
    if 'ANOMALY' in raw.upper(): return 'ANOMALY'
    return 'GOOD'


async def optimize_prompt_with_groq(current_prompt, results, iteration):
    """Use Groq to rewrite the prompt based on mistakes."""
    wrong = [r for r in results if not r['correct']]
    fp = [r for r in wrong if r['label']=='GOOD' and r['verdict']=='ANOMALY']
    fn = [r for r in wrong if r['label']=='BAD'  and r['verdict']=='GOOD']
    score = sum(1 for r in results if r['correct']) / len(results)

    feedback = f"""Current prompt accuracy: {score*100:.0f}% ({sum(1 for r in results if r['correct'])}/{len(results)})

FALSE POSITIVES (good images wrongly called ANOMALY): {len(fp)}
{chr(10).join([f"- {r['name']}" for r in fp])}

FALSE NEGATIVES (bad images with cracks missed as GOOD): {len(fn)}
{chr(10).join([f"- {r['name']}" for r in fn])}

These are grayscale BSH metal press tool images.
GOOD images: smooth curved metal surface, bright glare, dark background/fixture, machining lines -- all NORMAL.
BAD images: a visible crack or fracture that INTERRUPTS the smooth metal surface continuity.

The key visual difference: in BAD images there is a dark irregular discontinuity IN THE METAL SURFACE ITSELF.
In GOOD images the metal surface is completely smooth and continuous despite background shadows and glare.

Rewrite the prompt to fix these mistakes. Make it very specific about what constitutes a real crack vs normal features.
Return ONLY the new prompt text."""

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {s.groq_api_key}", "Content-Type": "application/json"},
                json={"model": s.groq_model, "messages": [
                    {"role": "system", "content": OPTIMIZER_SYSTEM},
                    {"role": "user", "content": f"Current prompt:\n{current_prompt}\n\nFeedback:\n{feedback}"}
                ], "max_tokens": 1500, "temperature": 0.3}
            )
            new_prompt = resp.json()["choices"][0]["message"]["content"].strip()
            return new_prompt
    except Exception as e:
        print(f"  Optimizer error: {e}")
        return current_prompt

async def main():
    print("Loading images...")
    good_images = load_images(GOOD_DIR, 'GOOD')
    bad_images  = load_images(BAD_DIR,  'BAD')
    all_images  = good_images + bad_images
    print(f"Loaded {len(good_images)} good + {len(bad_images)} bad = {len(all_images)} total\n")

    # Read current prompt from inspector
    with open('industrial/vision/inspector.py', 'r', encoding='utf-8') as f:
        content = f.read()
    start = content.find('INSPECTION_PROMPT_TEMPLATE = """') + len('INSPECTION_PROMPT_TEMPLATE = """')
    end = content.find('"""', start)
    current_prompt = content[start:end].strip()
    print(f"Current prompt length: {len(current_prompt)} chars\n")

    best_prompt = current_prompt
    best_score = 0.0
    best_results = []

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"{'='*60}")
        print(f"ITERATION {iteration}/{MAX_ITERATIONS}")
        print(f"{'='*60}")
        score, results = await test_prompt(current_prompt, all_images)
        print(f"\nScore: {score*100:.0f}% ({sum(1 for r in results if r['correct'])}/{len(results)} correct)")

        if score > best_score:
            best_score = score
            best_prompt = current_prompt
            best_results = results
            print(f"New best score: {best_score*100:.0f}%")

        if score >= TARGET_ACCURACY:
            print(f"\nTarget accuracy {TARGET_ACCURACY*100:.0f}% reached!")
            break

        print(f"\nOptimizing prompt with Groq (iteration {iteration})...")
        current_prompt = await optimize_prompt_with_groq(current_prompt, results, iteration)
        print(f"New prompt length: {len(current_prompt)} chars")

    print(f"\n{'='*60}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"Best score: {best_score*100:.0f}%")
    print(f"{'='*60}")

    # Save best prompt back to inspector.py
    new_content = content[:start] + '\n' + best_prompt + '\n' + content[end:]
    with open('industrial/vision/inspector.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Best prompt saved to industrial/vision/inspector.py")

    # Save report
    with open('optimization_report.txt', 'w', encoding='utf-8') as f:
        f.write(f"Optimization Report\nBest Score: {best_score*100:.0f}%\n\n")
        f.write(f"Best Prompt:\n{best_prompt}\n\nResults:\n")
        for r in best_results:
            f.write(f"{'OK' if r['correct'] else 'WRONG'} | {r['label']} | {r['name']} | {r['verdict']}\n")
    print("Report saved to optimization_report.txt")

if __name__ == '__main__':
    asyncio.run(main())
