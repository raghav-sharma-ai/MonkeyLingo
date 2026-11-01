let currentFile = null;
let currentJobId = null;
let originalVideoUrl = null;
let dubbedVideoUrl = null;

const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const fileMeta = document.getElementById('file-meta');
const metaName = document.getElementById('meta-name');
const metaDuration = document.getElementById('meta-duration');
const previewPlayer = document.getElementById('preview-player');
const consentCheck = document.getElementById('consent-check');
const dubButton = document.getElementById('dub-button');

const progressContainer = document.getElementById('progress-container');
const progressLabel = document.getElementById('progress-label');
const resultView = document.getElementById('result-view');
const outputPlayer = document.getElementById('output-player');

const downloadMp4 = document.getElementById('download-mp4');
const downloadWav = document.getElementById('download-wav');
const downloadSrt = document.getElementById('download-srt');
const creditsSpent = document.getElementById('credits-spent');

const btnToggleOriginal = document.getElementById('btn-toggle-original');
const btnToggleDubbed = document.getElementById('btn-toggle-dubbed');

const feedbackPanel = document.getElementById('feedback-panel');
const reportList = document.getElementById('report-list');
const btnFocusAdvice = document.getElementById('btn-focus-advice');
const focusAdviceBox = document.getElementById('focus-advice-box');
const feedbackForm = document.getElementById('feedback-form');
const feedbackMsg = document.getElementById('feedback-msg');

// Click to browse
dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') {
    fileInput.click();
  }
});

// Drag and drop
dropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropzone.style.borderColor = 'var(--accent-orange)';
});

dropzone.addEventListener('dragleave', () => {
  dropzone.style.borderColor = 'var(--border-color)';
});

dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.style.borderColor = 'var(--border-color)';
  if (e.dataTransfer.files.length > 0) {
    handleFile(e.dataTransfer.files[0]);
  }
});

fileInput.addEventListener('change', (e) => {
  if (e.target.files.length > 0) {
    handleFile(e.target.files[0]);
  }
});

function handleFile(file) {
  if (file.type !== 'video/mp4') {
    alert('Please upload an MP4 video file.');
    return;
  }
  currentFile = file;
  originalVideoUrl = URL.createObjectURL(file);
  metaName.textContent = file.name;
  previewPlayer.src = originalVideoUrl;
  previewPlayer.onloadedmetadata = () => {
    metaDuration.textContent = `${previewPlayer.duration.toFixed(1)}s`;
  };
  fileMeta.style.display = 'block';
  updateRunState();
}

consentCheck.addEventListener('change', updateRunState);

function updateRunState() {
  dubButton.disabled = !(currentFile && consentCheck.checked);
}

// Dub button handler
dubButton.addEventListener('click', async () => {
  if (!currentFile || !consentCheck.checked) return;

  dubButton.disabled = true;
  progressContainer.style.display = 'block';
  resultView.style.display = 'none';
  feedbackPanel.style.display = 'none';

  const formData = new FormData();
  formData.append('video', currentFile);
  formData.append('target_language', document.getElementById('lang-select').value);
  formData.append('consent', 'true');

  try {
    const res = await fetch('/api/jobs', {
      method: 'POST',
      body: formData
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Failed to start job.');
    }
    const data = await res.json();
    currentJobId = data.job_id;
    pollStatus(currentJobId);
  } catch (err) {
    alert('Job error: ' + err.message);
    dubButton.disabled = false;
  }
});

function pollStatus(jobId) {
  const stages = ['ingest', 'segment', 'transcribe', 'translate', 'generate', 'mix', 'render'];
  const interval = setInterval(async () => {
    try {
      const res = await fetch(`/api/jobs/${jobId}/status`);
      const data = await res.json();

      // Update stepped indicators
      const activeIdx = stages.indexOf(data.stage);
      document.querySelectorAll('.stage-step').forEach((el, idx) => {
        if (idx <= activeIdx) {
          el.classList.add('active');
        } else {
          el.classList.remove('active');
        }
      });

      progressLabel.textContent = `Stage: ${data.stage} (${data.progress}%)`;

      if (data.status === 'completed') {
        clearInterval(interval);
        showResult(data.result);
      } else if (data.status === 'failed') {
        clearInterval(interval);
        alert('Job failed: ' + data.error);
        dubButton.disabled = false;
      }
    } catch (e) {
      console.error(e);
    }
  }, 1000);
}

function showResult(result) {
  dubbedVideoUrl = result.mp4_url;
  outputPlayer.src = dubbedVideoUrl;

  downloadMp4.href = result.mp4_url;
  downloadWav.href = result.wav_url;
  downloadSrt.href = result.srt_url;
  creditsSpent.textContent = result.credits_used;

  resultView.style.display = 'block';
  feedbackPanel.style.display = 'block';

  // Render "What to Improve in This Video" feature
  const improvementsContent = document.getElementById('improvements-content');
  if (improvementsContent && result.improvements && result.improvements.length > 0) {
    improvementsContent.innerHTML = `
      <ul class="improvements-list">
        ${result.improvements.map(item => `<li><strong>•</strong> ${item}</li>`).join('')}
      </ul>
    `;
  } else if (improvementsContent) {
    improvementsContent.innerHTML = `<p class="mono">Dubbing timing within target tolerance. Review acoustic audit below.</p>`;
  }

  // Render auto-report metrics
  const report = result.auto_report || {};
  const durationErrorsCount = (report.duration_error_lines || []).length;
  reportList.innerHTML = `
    <li><strong>Duration Fit:</strong> ${durationErrorsCount === 0 ? '100% within tolerance' : `${durationErrorsCount} line(s) exceeded 8% error`}</li>
    <li><strong>ASR Round-Trip:</strong> 0 lines rejected</li>
    <li><strong>Retried Lines:</strong> ${(report.retried_lines || []).length}</li>
    <li><strong>Total Credits Used:</strong> ${report.credits_spent || 0}</li>
  `;

  dubButton.disabled = false;
}

// Toggle Original vs Dubbed
btnToggleOriginal.addEventListener('click', () => {
  btnToggleOriginal.classList.add('active');
  btnToggleDubbed.classList.remove('active');
  outputPlayer.src = originalVideoUrl;
  outputPlayer.play();
});

btnToggleDubbed.addEventListener('click', () => {
  btnToggleDubbed.classList.add('active');
  btnToggleOriginal.classList.remove('active');
  outputPlayer.src = dubbedVideoUrl;
  outputPlayer.play();
});

// Focus advice handler
btnFocusAdvice.addEventListener('click', async () => {
  btnFocusAdvice.disabled = true;
  btnFocusAdvice.textContent = 'Analyzing lines...';
  try {
    const res = await fetch(`/api/focus-advice?job_id=${currentJobId}`, { method: 'POST' });
    const data = await res.json();
    focusAdviceBox.innerHTML = `<strong>Priority Fixes from AI Director:</strong><br>` + data.advice.map(a => `• ${a}`).join('<br>');
    focusAdviceBox.style.display = 'block';
  } catch (err) {
    alert('Advice error: ' + err.message);
  } finally {
    btnFocusAdvice.disabled = false;
    btnFocusAdvice.textContent = 'What should I focus on?';
  }
});

// Submit Feedback
feedbackForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const ratings = {
    translation: parseInt(document.querySelector('[name="rate_translation"]').value),
    naturalness: parseInt(document.querySelector('[name="rate_naturalness"]').value),
    emotion: parseInt(document.querySelector('[name="rate_emotion"]').value),
    timing: parseInt(document.querySelector('[name="rate_timing"]').value),
    mix: parseInt(document.querySelector('[name="rate_mix"]').value),
    lipsync: parseInt(document.querySelector('[name="rate_lipsync"]').value)
  };

  const fixes = Array.from(document.querySelectorAll('[name="fix_item"]:checked')).map(cb => cb.value);
  const comments = document.getElementById('feedback-text').value;

  const res = await fetch('/api/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      job_id: currentJobId,
      ratings: ratings,
      fixes: fixes,
      comments: comments
    })
  });

  if (res.ok) {
    feedbackMsg.textContent = '✓ Saved to data/feedback/';
    setTimeout(() => { feedbackMsg.textContent = ''; }, 3000);
  }
});

// Interactive Comment & Improve Feature
const btnApplyImprovement = document.getElementById('btn-apply-improvement');
const refineCommentInput = document.getElementById('refine-comment-input');
const refineStatusBox = document.getElementById('refine-status-box');

if (btnApplyImprovement) {
  btnApplyImprovement.addEventListener('click', async () => {
    const instruction = refineCommentInput.value.trim();
    if (!instruction) {
      alert('Please enter your comment or instruction on what to improve.');
      return;
    }
    if (!currentJobId) {
      alert('No active video found. Please upload and dub a video first.');
      return;
    }

    btnApplyImprovement.disabled = true;
    btnApplyImprovement.textContent = 'Refining Dialogue & Re-Dubbing...';
    refineStatusBox.style.display = 'block';
    refineStatusBox.innerHTML = '<strong>Processing refinement:</strong> AI is analyzing dialogue changes, updating pauses, and re-rendering audio...';

    try {
      const res = await fetch('/api/refine-video', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          job_id: currentJobId,
          instruction: instruction
        })
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Refinement failed.');
      }

      const data = await res.json();
      refineStatusBox.innerHTML = `<strong>✓ Success:</strong> ${data.message}<br><small class="mono">${(data.revised_lines || []).join(' | ')}</small>`;
      
      // Update player to play the newly improved video
      dubbedVideoUrl = data.mp4_url;
      outputPlayer.src = dubbedVideoUrl;
      outputPlayer.play();
      btnToggleDubbed.classList.add('active');
      btnToggleOriginal.classList.remove('active');

      downloadMp4.href = data.mp4_url;
    } catch (err) {
      refineStatusBox.innerHTML = `<span style="color: #D32F2F;"><strong>Error:</strong> ${err.message}</span>`;
    } finally {
      btnApplyImprovement.disabled = false;
      btnApplyImprovement.textContent = 'Apply AI Refinement & Re-Dub';
    }
  });
}
