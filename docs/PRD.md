# MonkeyLingo: Performance-Aware AI Dubbing

## 1. Summary
**MonkeyLingo** dubs scenes into Indian languages (Hindi, Telugu, etc.) while preserving the original actors' performance: timing, energy, pacing, pauses, and character distinction. 

An analysis layer segments speech, measures actor delivery, and generates performance profiles. A translation layer rewrites lines under strict syllable budgets. Gnani Timbre synthesizes character-dedicated voices with clause-level pause mapping, and a finishing DSP layer locks speech onset to visual lip movements with zero dialogue bleed.

## 2. Core Architecture
- **Stage 0: Ingestion & Validation** — Extracts 48 kHz mono 16-bit PCM WAV audio from video via FFmpeg.
- **Stage 1: Speech Segmentation & Alignment** — Word-level timestamps and mouth-onset detection using Whisper and acoustic energy profiling.
- **Stage 2: Performance Profiling & Budgeted Translation** — Gemini multimodal analysis extracts emotion, intensity (1–5), and translates dialogue within syllable limits.
- **Stage 3: Character Synthesis** — Gnani Timbre (`timbre-v2.5`) synthesizes takes with character consistency (`Deepak` for male lead, `Nalini` for female lead).
- **Stage 4: Clause-Level Lip-Sync & Mixdown** — Sub-sentence pause injection, acoustic room air bed synthesis (zero 414Hz tone, zero bleed), and video multiplexing.
- **Stage 5: QA & Metric Reporting** — Duration error measurement, round-trip ASR validation, and feedback loop.
