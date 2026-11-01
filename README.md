# MonkeyLingo

> Performance-aware AI dubbing for Indian languages. Upload a video, pick a language, get a dubbed version with character consistency and sub-sentence lip-sync.

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
- **Dialogue Intelligence:** Google Gemini (`gemini-3.5-flash`)
- **Speech Segmentation & Alignment:** Whisper & Silero VAD
- **DSP & Multiplexing:** FFmpeg & Rubberband
- **Web Application:** FastAPI + Vanilla JS / CSS (Light, tool-like design)

---

## 4. Standing Lessons Learned

See [`docs/LESSONS.md`](docs/LESSONS.md) for the complete table of architectural mistakes, fixes, and rules (L1–L14).
