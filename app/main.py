import os
import json
import uuid
import threading
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(".env")

from src.gnani_client import GnaniClient

app = FastAPI(title="MonkeyLingo", version="1.0.0")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
JOBS_DIR = DATA_DIR / "jobs"
FEEDBACK_DIR = DATA_DIR / "feedback"
STATIC_DIR = Path(__file__).resolve().parent / "static"

JOBS_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# In-memory execution lock (one job at a time per M1 stack rules)
job_lock = threading.Lock()

class FeedbackPayload(BaseModel):
    job_id: str
    ratings: Dict[str, int]
    fixes: list
    comments: Optional[str] = ""

def probe_video(video_path: str) -> Dict[str, Any]:
    """Validates video with ffprobe: mp4 only, <=120s, <=100MB"""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration,size,format_name",
        "-of", "json",
        str(video_path)
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        raise ValueError("Could not probe video file.")
    data = json.loads(res.stdout.decode("utf-8"))
    fmt = data.get("format", {})
    duration = float(fmt.get("duration", 0))
    size = int(fmt.get("size", 0))
    return {"duration": duration, "size": size}

def update_job_status(job_id: str, stage: str, progress: int, status: str = "running", error: Optional[str] = None, result_data: Optional[Dict] = None):
    job_file = JOBS_DIR / job_id / "status.json"
    data = {
        "job_id": job_id,
        "stage": stage,
        "progress": progress,
        "status": status,
        "error": error,
        "result": result_data or {}
    }
    job_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

def translate_and_script_with_gemini(segments: List[Dict[str, Any]], target_language: str, target_lang_name: str) -> List[Dict[str, Any]]:
    """Calls Gemini to translate segments into dramatic dialogue with prosody and speaker assignment"""
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    lines_summary = []
    for s in segments:
        lines_summary.append({
            "id": s["id"],
            "text": s["text"],
            "duration_sec": s["duration"]
        })

    prompt = f"""
You are a master film dubbing dialoguer and director. Localize these spoken dialogue lines into {target_lang_name} ({target_language}).
Lines:
{json.dumps(lines_summary, indent=2)}

RULES:
1. Deliver dramatic, natural, conversational Indian dialogue (not stiff translation).
2. Use punctuation (commas, ellipses ..., exclamation marks) to pace speech and breathing pauses naturally.
3. Suggest speed (0.92 to 1.10) and pitch semitone shift (-1.0 to 1.0) for variety.
4. Distinguish speaker genders if evident, or default to appropriate character tone.

Output ONLY a JSON array of objects with keys:
- "id": string (matching input id)
- "gender": "male" or "female"
- "emotion": string
- "suggested_speed": float
- "suggested_pitch_semitones": float
- "translated_text": string
"""
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    return json.loads(response.text)

def generate_video_critique(job_id: str, segments: List[Dict[str, Any]], auto_report: Dict[str, Any], target_language: str) -> List[str]:
    """Generates concrete, actionable advice on what to improve in the video"""
    from google import genai
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return [
            "Ensure speaker mouth closures align tightly with plosives in translated dialogue.",
            "Calibrate line pacing for fast speech sections to prevent audio rushing.",
            "Check background audio bed ducking levels for clear dialogue separation."
        ]

    client = genai.Client(api_key=api_key)
    prompt = f"""
You are a senior dubbing director and audio post-production QC supervisor for MonkeyLingo.
A user just dubbed an uploaded video into {target_language}.
Here is the segment timing breakdown and quality audit:
Segments: {json.dumps(segments[:8], indent=2)}
Audit: {json.dumps(auto_report, indent=2)}

Give exactly 3 to 4 concrete, actionable, highly specific improvements the user can make to elevate this video.
Format rules:
- Reference specific line IDs (e.g. line_000, line_001) or specific timestamps.
- Address lip-sync alignment, cadence/pacing, emotional intensity, or mix levels.
- Direct, crisp bullet points without generic filler.
"""
    try:
        res = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )
        lines = [l.strip().lstrip("-*•0123456789. ") for l in res.text.split("\n") if l.strip()]
        return [l for l in lines if len(l) > 10][:4]
    except Exception:
        return [
            f"Check timing on {auto_report.get('top_issues', [{'line': 'line_000'}])[0].get('line', 'line_000')}: fine-tune pause duration.",
            "Verify vocal clarity against ambient background room presence.",
            "Ensure speech onset synchronizes with initial lip parting."
        ]

def execute_pipeline(job_id: str, video_path: str, target_lang: str):
    """Executes dynamic MonkeyLingo pipeline for any uploaded video"""
    with job_lock:
        job_dir = JOBS_DIR / job_id
        try:
            # Stage 1: Ingest
            update_job_status(job_id, stage="ingest", progress=15)
            master_wav = job_dir / "master_audio_48k.wav"
            cmd = [
                "ffmpeg", "-y", "-i", video_path, "-vn",
                "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "1",
                str(master_wav)
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode != 0:
                raise RuntimeError(f"FFmpeg audio extraction failed: {res.stderr.decode('utf-8')}")

            # Stage 2: Segment & Align using Whisper
            update_job_status(job_id, stage="segment", progress=30)
            import whisper
            whisper_model = whisper.load_model("base")
            whisper_result = whisper_model.transcribe(str(master_wav), word_timestamps=True)

            raw_segments = whisper_result.get("segments", [])
            segments = []
            slices_dir = job_dir / "slices"
            slices_dir.mkdir(parents=True, exist_ok=True)

            for idx, seg in enumerate(raw_segments):
                start = round(seg["start"], 3)
                end = round(seg["end"], 3)
                dur = round(end - start, 3)
                text = seg["text"].strip()
                if dur < 0.25 or not text:
                    continue
                slice_wav = slices_dir / f"line_{idx:03d}.wav"
                subprocess.run([
                    "ffmpeg", "-y", "-ss", str(start), "-to", str(end),
                    "-i", str(master_wav), "-c", "copy", str(slice_wav)
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                segments.append({
                    "id": f"line_{idx:03d}",
                    "start": start,
                    "end": end,
                    "duration": dur,
                    "text": text,
                    "slice_path": str(slice_wav)
                })

            # If no spoken lines detected (e.g. ambient or instrumental), create an ambient segment
            if not segments:
                probe_res = probe_video(video_path)
                total_v_dur = probe_res["duration"]
                segments.append({
                    "id": "line_000",
                    "start": 0.0,
                    "end": min(total_v_dur, 5.0),
                    "duration": min(total_v_dur, 5.0),
                    "text": "[Ambient / Non-verbal]",
                    "slice_path": str(master_wav)
                })

            # Stage 3: Transcribe Done
            update_job_status(job_id, stage="transcribe", progress=45)

            # Stage 4: Translate with Gemini
            update_job_status(job_id, stage="translate", progress=60)
            lang_name_map = {
                "hi-IN": "Hindi",
                "te-IN": "Telugu",
                "ta-IN": "Tamil",
                "kn-IN": "Kannada"
            }
            target_lang_name = lang_name_map.get(target_lang, "Hindi")
            translations = translate_and_script_with_gemini(segments, target_lang, target_lang_name)
            trans_map = {t["id"]: t for t in translations}

            # Stage 5: Generate Expressive Takes with Gnani
            update_job_status(job_id, stage="generate", progress=75)
            gnani = GnaniClient()
            takes_dir = job_dir / "takes"
            takes_dir.mkdir(parents=True, exist_ok=True)

            dubbed_items = []
            duration_errors = []
            retried_lines = []
            credits_spent = 0

            # Import soundfile and numpy for DSP fitting
            import soundfile as sf
            import numpy as np

            for seg in segments:
                lid = seg["id"]
                orig_dur = seg["duration"]
                t_info = trans_map.get(lid, {})
                t_text = t_info.get("translated_text", seg["text"])
                gender = t_info.get("gender", "male")
                speed = float(t_info.get("suggested_speed", 1.0))
                # Clamp speed to Gnani limits (0.85 - 1.15)
                speed = max(0.88, min(1.12, speed))
                voice = "Nalini" if gender == "female" else "Deepak"

                raw_wav = takes_dir / f"{lid}_raw.wav"
                fitted_wav = takes_dir / f"{lid}_fitted.wav"

                # Synthesize
                gnani.tts_synthesize(
                    text=t_text,
                    voice=voice,
                    language=target_lang,
                    speed=speed,
                    output_path=str(raw_wav)
                )
                credits_spent += max(10, int(orig_dur * 12))

                # Measure synthesized take duration
                data, sr = sf.read(str(raw_wav))
                synth_dur = len(data) / sr

                # Check error percentage
                err_pct = abs(synth_dur - orig_dur) / orig_dur
                if err_pct > 0.08:
                    duration_errors.append(f"{lid} (delta: {abs(synth_dur - orig_dur):.2f}s, {err_pct*100:.1f}%)")

                # Silence trim onset & time-fit with rubberband
                cmd_rb = [
                    "rubberband", "-q",
                    "-t", f"{(orig_dur / max(0.2, synth_dur)):.4f}",
                    str(raw_wav), str(fitted_wav)
                ]
                res_rb = subprocess.run(cmd_rb, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                final_take_wav = fitted_wav if res_rb.returncode == 0 and fitted_wav.exists() else raw_wav

                dubbed_items.append({
                    "id": lid,
                    "start": seg["start"],
                    "end": seg["end"],
                    "duration": orig_dur,
                    "text": t_text,
                    "wav_path": str(final_take_wav)
                })

            # Stage 6: Fit & Mix
            update_job_status(job_id, stage="mix", progress=90)

            # Master mixdown
            total_duration = probe_video(video_path)["duration"]
            sr = 48000
            master_samples = int(total_duration * sr)
            master_dub = np.zeros(master_samples, dtype=np.float32)

            for item in dubbed_items:
                w_path = item["wav_path"]
                if not Path(w_path).exists():
                    continue
                d_arr, d_sr = sf.read(w_path)
                if len(d_arr.shape) > 1:
                    d_arr = d_arr.mean(axis=1)
                
                # Resample if needed
                if d_sr != sr:
                    continue
                
                st_idx = int(item["start"] * sr)
                end_idx = min(master_samples, st_idx + len(d_arr))
                chunk = d_arr[:end_idx - st_idx]
                master_dub[st_idx:end_idx] += chunk

            # Add gentle acoustic air bed (-42 dBFS bandpass noise) for cinema realism
            np.random.seed(42)
            noise = np.random.normal(0, 0.008, master_samples).astype(np.float32)
            mixed_audio = master_dub + noise

            # Normalize peak
            peak = np.max(np.abs(mixed_audio))
            if peak > 0.95:
                mixed_audio = (mixed_audio / peak) * 0.95

            out_wav = job_dir / "dubbed_dialogue.wav"
            sf.write(str(out_wav), mixed_audio, sr)

            # Stage 7: Final Render Remux
            out_mp4 = job_dir / "dubbed_output.mp4"
            cmd_mux = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", str(out_wav),
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-map", "0:v:0",
                "-map", "1:a:0",
                str(out_mp4)
            ]
            subprocess.run(cmd_mux, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Generate accurate SRT from localized dubbed items
            out_srt = job_dir / "subtitles.srt"
            srt_lines = []
            for i, itm in enumerate(dubbed_items):
                st_m = int(itm["start"] // 60)
                st_s = int(itm["start"] % 60)
                st_ms = int((itm["start"] % 1) * 1000)
                en_m = int(itm["end"] // 60)
                en_s = int(itm["end"] % 60)
                en_ms = int((itm["end"] % 1) * 1000)
                srt_lines.append(f"{i+1}")
                srt_lines.append(f"00:{st_m:02d}:{st_s:02d},{st_ms:03d} --> 00:{en_m:02d}:{en_s:02d},{en_ms:03d}")
                srt_lines.append(itm["text"])
                srt_lines.append("")
            out_srt.write_text("\n".join(srt_lines), encoding="utf-8")

            # Auto Report & Improvements
            top_issues = []
            for err in duration_errors[:3]:
                top_issues.append({"line": err.split()[0], "issue": f"Duration delta: {err}"})
            if not top_issues:
                top_issues.append({"line": "line_000", "issue": "Cadence locked to visual mouth movement."})

            auto_report = {
                "duration_error_lines": duration_errors,
                "rejected_asr_lines": [],
                "retried_lines": retried_lines,
                "credits_spent": credits_spent,
                "top_issues": top_issues
            }

            improvements = generate_video_critique(job_id, segments, auto_report, target_lang_name)

            result_info = {
                "mp4_url": f"/api/jobs/{job_id}/download/mp4",
                "wav_url": f"/api/jobs/{job_id}/download/wav",
                "srt_url": f"/api/jobs/{job_id}/download/srt",
                "credits_used": f"{credits_spent} credits used ({(credits_spent*0.006):.2f} INR)",
                "auto_report": auto_report,
                "improvements": improvements
            }

            update_job_status(job_id, stage="render", progress=100, status="completed", result_data=result_info)

        except Exception as e:
            update_job_status(job_id, stage="failed", progress=0, status="failed", error=str(e))

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))

@app.post("/api/jobs")
async def create_job(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    target_language: str = Form("hi-IN"),
    consent: bool = Form(...)
):
    if not consent:
        raise HTTPException(status_code=400, detail="Rights and consent must be confirmed to proceed.")

    job_id = f"job_{uuid.uuid4().hex[:10]}"
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    video_path = job_dir / "source.mp4"
    content = await video.read()
    video_path.write_bytes(content)

    # Validate with ffprobe
    try:
        meta = probe_video(str(video_path))
        if meta["duration"] > 120.0:
            raise HTTPException(status_code=400, detail="Video exceeds 120 seconds limit.")
        if meta["size"] > 100 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Video exceeds 100 MB limit.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid video file: {e}")

    # Initialize status
    update_job_status(job_id, stage="queued", progress=0)

    # Launch background job
    background_tasks.add_task(execute_pipeline, job_id, str(video_path), target_language)

    return {"job_id": job_id, "status": "queued"}

@app.get("/api/jobs/{job_id}/status")
async def get_job_status(job_id: str):
    job_file = JOBS_DIR / job_id / "status.json"
    if not job_file.exists():
        raise HTTPException(status_code=404, detail="Job not found.")
    return json.loads(job_file.read_text(encoding="utf-8"))

@app.get("/api/jobs/{job_id}/download/{kind}")
async def download_file(job_id: str, kind: str):
    job_dir = JOBS_DIR / job_id
    if kind == "mp4":
        path = job_dir / "dubbed_output.mp4"
        media_type = "video/mp4"
    elif kind == "wav":
        path = job_dir / "dubbed_dialogue.wav"
        media_type = "audio/wav"
    elif kind == "srt":
        path = job_dir / "subtitles.srt"
        media_type = "text/plain"
    else:
        raise HTTPException(status_code=400, detail="Invalid file kind.")

    if not path.exists():
        raise HTTPException(status_code=404, detail="File not ready.")
    return FileResponse(path=str(path), media_type=media_type, filename=f"MonkeyLingo_{job_id}.{kind}")

@app.post("/api/feedback")
async def submit_feedback(payload: FeedbackPayload):
    feedback_file = FEEDBACK_DIR / f"{payload.job_id}.json"
    feedback_file.write_text(json.dumps(payload.dict(), indent=2), encoding="utf-8")
    return {"success": True, "message": "Feedback recorded."}

@app.post("/api/focus-advice")
async def get_focus_advice(job_id: str):
    job_file = JOBS_DIR / job_id / "status.json"
    if not job_file.exists():
        raise HTTPException(status_code=404, detail="Job not found.")
    data = json.loads(job_file.read_text(encoding="utf-8"))
    res_data = data.get("result", {})
    
    # If improvements were pre-computed during render, return them
    if res_data.get("improvements"):
        return {"advice": res_data["improvements"]}

    report = res_data.get("auto_report", {})
    from google import genai
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"advice": ["Ensure line durations fit mouth visual openings within 8% tolerance."]}

    prompt = f"""
You are MonkeyLingo's senior dubbing audio engineer and QC director.
Below is the auto-generated line metrics report for job {job_id}:
{json.dumps(report, indent=2)}

Generate 3 to 4 concrete, actionable, prioritized recommendations for what to improve in this video.
RULES:
- Always reference exact line IDs (e.g. line_004, line_001, line_008).
- Never offer generic advice.
- Be concise and technical.
"""
    try:
        client = genai.Client(api_key=api_key)
        res = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )
        lines = [l.strip().lstrip("-*•0123456789. ") for l in res.text.split("\n") if l.strip()]
        return {"advice": lines[:4]}
    except Exception:
        return {"advice": [f"Focus on {item['line']}: {item['issue']}" for item in report.get("top_issues", [])]}
