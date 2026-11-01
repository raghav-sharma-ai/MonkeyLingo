from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_home_page():
    res = client.get("/")
    assert res.status_code == 200
    assert "MonkeyLingo" in res.text
    assert "Upload a video, pick a language, get a dubbed version." in res.text

def test_job_validation_no_consent():
    res = client.post("/api/jobs", data={"target_language": "hi-IN", "consent": "false"})
    assert res.status_code in [400, 422]
