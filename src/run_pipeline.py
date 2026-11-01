import os
import sys
import json
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load credentials
load_dotenv(".env")

from src.ingest import AudioIngest
from src.gnani_client import GnaniClient
from src.gemini_dubber import GeminiDubber
from src.fit_and_mix import AudioFitter

def run_pipeline():
    print("=" * 60)
    print("🎬 MonkeyLingo: Performance-Aware AI Dubbing Pipeline")
    print("=" * 60)

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    video_path = config["scene"]["video_path"]
    target_lang = config["project"]["target_language"]
    target_lang_name = config["project"]["target_language_name"]
    voice = config["tts"]["default_voice"]
    
    print(f"[*] Input Video: {video_path}")
    print(f"[*] Target Language: {target_lang_name} ({target_lang})")
    print(f"[*] TTS Voice: {voice}")

    # Step 1: Ingest & Slicing
    print("\n[Step 1/4] Ingesting video and segmenting dialogue...")
    ingest = AudioIngest(video_path)
    master_wav = ingest.extract_audio()
    
    segments_file = Path("data/scene/segments.json")
    if segments_file.exists():
        with open(segments_file, "r") as f:
            segments = json.load(f)
        print(f"[+] Loaded {len(segments)} cached dialogue lines.")
    else:
        segments = ingest.segment_with_whisper(model_size="base")
        print(f"[+] Identified {len(segments)} spoken dialogue lines.")
        with open(segments_file, "w") as f:
            json.dump(segments, f, indent=2)

    # Step 2: Global Scene Analysis & Budgeted Translation
    print("\n[Step 2/4] Scene performance profiling & translation...")
    gemini = GeminiDubber()
    translations_data = gemini.analyze_and_translate_all(segments, target_lang_name)
    trans_map = {t["id"]: t for t in translations_data}

    # Step 3: Synthesis & Time-Fitting
    print("\n[Step 3/4] Synthesizing expressive takes & applying rubberband fitting...")
    gnani = GnaniClient()
    fitter = AudioFitter()

    out_takes_dir = Path("out/takes")
    out_takes_dir.mkdir(parents=True, exist_ok=True)
    dubbed_items = []

    for seg in segments:
        line_id = seg["id"]
        orig_text = seg["text"]
        dur = seg["duration"]
        
        info = trans_map.get(line_id, {})
        target_trans = info.get("target_translation", orig_text)
        emotion = info.get("emotion", "neutral")
        intensity = info.get("intensity", 3)
        notes = info.get("delivery_notes", "")

        print(f"\n--- {line_id} ({dur:.2f}s) ---")
        print(f"Original: \"{orig_text}\"")
        print(f"Profile:  [{emotion.upper()}, Intensity: {intensity}/5] {notes}")
        print(f"Hindi:    \"{target_trans}\"")

        raw_take = out_takes_dir / f"{line_id}_raw.wav"
        fitted_take = out_takes_dir / f"{line_id}_fitted.wav"

        try:
            gnani.tts_synthesize(
                text=target_trans,
                voice=voice,
                language=target_lang,
                output_path=str(raw_take)
            )

            # Rubberband time-stretching to match original duration
            fitter.time_fit(
                input_wav=str(raw_take),
                target_duration=dur,
                output_wav=str(fitted_take)
            )

            dubbed_items.append({
                "id": line_id,
                "start": seg["start"],
                "wav_path": str(fitted_take)
            })
            print(f"[✓] Take generated and time-fitted to {dur:.2f}s.")
        except Exception as e:
            print(f"[!] Error generating take for {line_id}: {e}")

    # Step 4: Mixdown and Final Render
    print("\n[Step 4/4] Mixing master dialogue track and assembling final video...")
    out_video = "out/final_dubbed_scene.mp4"
    Path("out").mkdir(parents=True, exist_ok=True)
    
    total_dur = fitter.get_duration(str(master_wav))
    fitter.mix_scene(
        video_in=video_path,
        dubbed_dialogues=dubbed_items,
        total_duration=total_dur,
        output_mp4=out_video
    )

    print("\n" + "=" * 60)
    print(f"🎉 Pipeline Complete! Dubbed video exported to: {out_video}")
    print("=" * 60)

if __name__ == "__main__":
    run_pipeline()
