import os
import hashlib
import requests
from pathlib import Path
from typing import Optional, Dict, Any

class GnaniClient:
    """
    Client for Gnani AI / Vachana APIs (Timbre TTS & Prisma STT).
    Features disk-based caching by payload hash.
    """
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.vachana.ai", cache_dir: str = "cache"):
        self.api_key = api_key or os.getenv("GNANI_API_KEY")
        if not self.api_key:
            raise ValueError("Gnani API Key is missing. Set GNANI_API_KEY in environment or pass to constructor.")
        
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.headers = {
            "X-API-Key-ID": self.api_key
        }

    def _hash_payload(self, prefix: str, data: str) -> str:
        h = hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]
        return f"{prefix}_{h}"

    def tts_synthesize(
        self,
        text: str,
        voice: str = "Nalini",
        language: str = "hi-IN",
        model: str = "timbre-v2.5",
        speed: float = 1.0,
        output_path: Optional[str] = None
    ) -> bytes:
        """
        Synthesizes speech using Gnani Timbre REST endpoint.
        Caches response to avoid redundant API credit burns.
        """
        cache_key = self._hash_payload("tts", f"{text}_{voice}_{language}_{model}_{speed}")
        cached_file = self.cache_dir / f"{cache_key}.wav"
        
        if cached_file.exists():
            data = cached_file.read_bytes()
            if output_path:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                Path(output_path).write_bytes(data)
            return data

        url = f"{self.base_url}/api/v1/tts/inference"
        payload = {
            "text": text,
            "voice": voice,
            "model": model,
            "language": language,
            "speed": speed,
            "audio_config": {
                "encoding": "linear_pcm",
                "container": "wav",
                "num_channels": 1,
                "sample_rate": 48000,
                "sample_width": 2
            }
        }
        headers = {**self.headers, "Content-Type": "application/json"}
        
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code != 200:
            raise RuntimeError(f"Gnani TTS failed with status {response.status_code}: {response.text}")
        
        audio_bytes = response.content
        cached_file.write_bytes(audio_bytes)
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(audio_bytes)
            
        return audio_bytes

    def stt_transcribe(
        self,
        audio_file_path: str,
        model: str = "prisma-v3.0",
        language_code: str = "indic-auto"
    ) -> Dict[str, Any]:
        """
        Transcribes speech using Gnani Prisma v3 REST endpoint.
        """
        file_bytes = Path(audio_file_path).read_bytes()
        cache_key = self._hash_payload("stt", f"{hashlib.md5(file_bytes).hexdigest()}_{model}_{language_code}")
        cached_json = self.cache_dir / f"{cache_key}.json"

        if cached_json.exists():
            import json
            return json.loads(cached_json.read_text(encoding="utf-8"))

        url = f"{self.base_url}/stt/v3"
        files = {
            "audio_file": (Path(audio_file_path).name, file_bytes, "audio/wav")
        }
        data = {
            "model": model,
            "language_code": language_code
        }
        
        response = requests.post(url, headers=self.headers, files=files, data=data)
        if response.status_code != 200:
            raise RuntimeError(f"Gnani STT failed with status {response.status_code}: {response.text}")
            
        res_data = response.json()
        import json
        cached_json.write_text(json.dumps(res_data, indent=2, ensure_ascii=False), encoding="utf-8")
        return res_data
