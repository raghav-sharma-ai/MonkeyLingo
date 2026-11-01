import os
import json
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
from google import genai
from google.genai import types

class GeminiDubber:
    """
    Advanced localization & performance director using Gemini 3.5 Flash.
    Shapes phrasing, injects breath/pauses via punctuation, and optimizes cadence.
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-3.5-flash", cache_dir: str = "cache"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is missing. Pass it or set it in .env")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def analyze_and_script_scene(
        self,
        segments: List[Dict[str, Any]],
        target_language: str
    ) -> List[Dict[str, Any]]:
        """
        Translates dialogues with high dramatic realism, natural pauses (...),
        conversational fillers, and exact timing constraints.
        """
        cache_file = self.cache_dir / f"dramatic_script_{target_language}.json"
        if cache_file.exists():
            print("[+] Loaded cached dramatic script & prosody data.")
            return json.loads(cache_file.read_text(encoding="utf-8"))

        lines_summary = []
        for s in segments:
            lines_summary.append({
                "id": s["id"],
                "speaker": "Mother" if s["id"] in ["line_001", "line_002"] else "Robert",
                "gender": "female" if s["id"] in ["line_001", "line_002"] else "male",
                "text": s["text"],
                "duration_sec": s["duration"]
            })

        prompt = f"""
You are an award-winning Bollywood film director and dubbing dialoguer localizing an intense cinematic scene into {target_language}.
Characters:
- Robert (Male, 20s): Emotional, passionate, cynical about life being random, pleading with his mother.
- Mother (Female, 50s): Calm, weary, patient but skeptical.

Original scene lines and speaking windows in seconds:
{json.dumps(lines_summary, indent=2)}

YOUR MISSION:
Eliminate the 'documentary/robotic script reading' feel! We need real acting dialogue with:
1. Emotion & Prosody punctuation: Use commas, question marks, and ellipses (...) strategically where the actor should breathe or pause.
2. Colloquial / Natural Indian dialogue: Use authentic conversational phrasing (e.g. "अरे माँ...", "मतलब...", "मेरी बात सुनो...").
3. Strict timing fit: The Hindi phrasing must be spoken naturally within the exact 'duration_sec' without rushing or awkward dragging.
4. Voice Speed Modulation: Suggest a speed factor (e.g. 0.95 for slow thoughtful lines, 1.05 for hurried outbursts).
5. Pitch semitone tweak: Suggest subtle pitch variation (-1.0 to +1.0) to avoid monotonic robot delivery.

Output a valid JSON array of objects with keys:
- "id": string (e.g. "line_000")
- "speaker": "Robert" or "Mother"
- "emotion": string
- "suggested_speed": float (between 0.92 and 1.10)
- "suggested_pitch_semitones": float (between -1.5 and 1.5)
- "dramatic_hindi_script": string (Hindi line with pauses/punctuation)

Output JSON only, no markdown:
"""
        print("[*] Consulting Gemini Dubbing Director for cinematic dialogue and prosody...")
        for attempt in range(5):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
                results = json.loads(response.text)
                cache_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
                return results
            except Exception as e:
                print(f"[!] Gemini attempt {attempt+1} failed ({e}). Retrying...")
                time.sleep(5)
                
        raise RuntimeError("Failed to generate dramatic script.")
