# MonkeyLingo

> **What happens to an iconic character when their voice artist is suddenly gone?**  
> MonkeyLingo preserves character vocal identity across Indian languages — slashing studio dubbing costs by 90% while using facial ML and acoustic modulation to clone the exact emotion, intensity, and lip-cadence of the original performance.

---

## The Billion-Dollar Hook: The "Voice Continuity" Crisis

Imagine watching Season 4 of your favorite series. Your hero walks onto the screen, opens their mouth, and... **it's someone else.**

The timbre is off. The gravel is missing. The subtle smirk in their cadence is gone. 

Within 3 seconds, the illusion shatters.

This is the silent nightmare of film studios, OTT platforms, and anime houses worldwide:
1. **The Inevitable Voice Loss**: Dubbing artists age, fall ill, part ways over contract disputes, or tragically pass away midway through a multi-season franchise. Studios scramble to recast, but audiences instantly reject the replacement. Decades of character equity vanish overnight.
2. **The Budget Wall**: Hiring specialized voice casts across Hindi, Telugu, Tamil, and Malayalam costs thousands of dollars per episode, plus weeks of studio scheduling, ADR stages, and union re-recording fees. As a result, 85% of regional content never gets dubbed at all.
3. **The "Uncanny Valley" of Existing AI Dubs**: Current AI tools either sound like monotone GPS navigators, or they artificially stretch speech into slurred, slow-motion mush just to match the visual timing. No filmmaker with self-respect puts that on screen.

---

## The Breakthrough: MonkeyLingo

**MonkeyLingo is the synthetic ADR and performance-directed dubbing studio for Indian languages.**

We don't just translate words; we preserve character soul, emotion, and visual synchrony:

* **Locked Character Identity for Life**:  
  Lock your character's vocal signature (`Deepak`, `Nalini`, etc.) perpetually. Whether you're producing Episode 1 or Season 10 five years later, your character sounds identical, consistent, and unmistakably *them*.
* **Emotion & Modulation Mirroring via MediaPipe ML**:  
  How do you know if an actor is whispering, weeping, or screaming? MonkeyLingo will pair computer vision with acoustic modulation. By tracking facial landmarks, lip aperture, and jaw tension using **MediaPipe ML**, our AI measures the original actor's visual emotional intensity and modulates the synthesized voice—matching pitch shifts, breath intake, and emotional energy to the original performance in future.
* **Sub-Sentence Lip-Sync (No Slurred Speech)**:  
  Humans don't speak in robot blocks; they breathe. MonkeyLingo segments speech into clause-level breathing intervals and trims digital onset silence. Speech locks to the actor's mouth opening, letting vocal decay breathe naturally into room tone instead of artificially warping audio.
* **From Weeks to 60 Seconds (At 1/50th the Cost)**:  
  Replace weeks of studio booking, sound engineers, and ADR retakes with an automated pipeline that ingests, transcribes, translates with dramatic intent, and mixes a broadcast-ready clip for less than ₹1 per take.
* **Director-in-the-Loop AI Re-Dubbing**:  
  Filmmakers don't surrender creative control. Notice something off? Type a direct note like a real director:  
  *“Make Line 1 sound hurried and cynical”* or *“Tighten the breath before goodbye”* — and watch MonkeyLingo re-synthesize and re-mux the take instantly.
* **Cinema-Grade Zero-Bleed Audio**:  
  Original English vocals are 100% eliminated, replaced by clean Indian-language takes over calibrated studio acoustic air. No digital beeps, no background mumble bleed.

---

## 1. Quick Start (<10 lines)

```bash
# 1. Clone repository
git clone https://github.com/raghav-sharma-ai/MonkeyLingo.git && cd MonkeyLingo

# 2. Setup environment & install dependencies
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3. Configure API keys in .env (GEMINI_API_KEY and GNANI_API_KEY)
cp .env.example .env

# 4. Start local web server
uvicorn app.main:app --port 8000 --reload
```

Open `http://localhost:8000` in your browser.

---

## 2. Core Capabilities

- **Character Consistency:** Dedicates distinct voices per actor (e.g. `Deepak` for Male Lead, `Nalini` for Female Lead) throughout the scene.
- **Sub-Sentence Clause Lip-Sync:** Splits complex thoughts around actor breathing intervals, locking speech onsets without artificial slow-motion stretching.
- **Acoustic Studio Mastering:** Eliminates English audio bleed and digital boundary beeps using synthesized room acoustic air.
- **Round-Trip QA & Auto-Reporting:** Measures line duration fit, WER thresholds, and credit consumption.

---

## 3. Technology Stack

- **Speech Synthesis:** Gnani Timbre (`timbre-v2.5` REST API)
- **Dialogue Intelligence:** Google Gemini (`gemini-3.5-flash-lite`, `gemini-3.7-flash` fallback cascade)
- **Speech Segmentation & Alignment:** Whisper & Silero VAD
- **DSP & Multiplexing:** FFmpeg & Rubberband
- **Web Application:** FastAPI + Vanilla JS / CSS (Light, tool-like design)

---

## 4. Standing Lessons Learned

See [`docs/LESSONS.md`](docs/LESSONS.md) for the complete table of architectural mistakes, fixes, and rules (L1–L20).
