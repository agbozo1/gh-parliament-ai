// Same-origin by default (works when FastAPI serves this page directly).
// If you host this page separately (e.g. on Netlify) from the API, set
// this to the API's base URL, e.g. "https://your-api.onrender.com".
const API_BASE = "";

const statusEl = document.getElementById("status");
const statusTextEl = document.getElementById("status-text");
const form = document.getElementById("query-form");
const questionInput = document.getElementById("question");
const submitBtn = document.getElementById("submit-btn");
const advancedToggle = document.getElementById("advanced-toggle");
const advancedPanel = document.getElementById("advanced");
const dateFilterInput = document.getElementById("date-filter");
const loadingEl = document.getElementById("loading");
const errorEl = document.getElementById("error");
const resultEl = document.getElementById("result");
const answerEl = document.getElementById("answer");
const sourcesEl = document.getElementById("sources");
const ingestToggle = document.getElementById("ingest-toggle");
const ingestPanel = document.getElementById("ingest-panel");
const ingestFrom = document.getElementById("ingest-from");
const ingestTo = document.getElementById("ingest-to");
const ingestBtn = document.getElementById("ingest-btn");
const ingestStatusEl = document.getElementById("ingest-status");

async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    statusEl.className = "status status--ok";
    statusTextEl.textContent = `Online — ${data.documents_indexed} document chunk(s) indexed`;
  } catch (err) {
    statusEl.className = "status status--error";
    statusTextEl.textContent = "API unreachable";
  }
}

advancedToggle.addEventListener("click", () => {
  const isHidden = advancedPanel.classList.toggle("hidden");
  advancedToggle.textContent = isHidden ? "+ Filter by date" : "− Hide date filter";
});

function renderSources(sources) {
  sourcesEl.innerHTML = "";
  for (const source of sources || []) {
    const row = document.createElement("div");
    row.className = "source";

    const label = document.createElement("span");
    label.textContent = [source.document, source.date].filter(Boolean).join(" — ");
    row.appendChild(label);

    if (source.url) {
      const link = document.createElement("a");
      link.href = source.url;
      link.target = "_blank";
      link.rel = "noopener";
      link.textContent = "View PDF ↗";
      row.appendChild(link);
    }

    sourcesEl.appendChild(row);
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const question = questionInput.value.trim();
  if (!question) return;

  errorEl.classList.add("hidden");
  resultEl.classList.add("hidden");
  loadingEl.classList.remove("hidden");
  submitBtn.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        date_filter: dateFilterInput.value || null,
      }),
    });

    if (!res.ok) {
      throw new Error(`Request failed (HTTP ${res.status})`);
    }

    const data = await res.json();
    answerEl.textContent = data.answer;
    renderSources(data.sources);
    resultEl.classList.remove("hidden");
  } catch (err) {
    errorEl.textContent = `Something went wrong: ${err.message}`;
    errorEl.classList.remove("hidden");
  } finally {
    loadingEl.classList.add("hidden");
    submitBtn.disabled = false;
  }
});

ingestToggle.addEventListener("click", () => {
  const isHidden = ingestPanel.classList.toggle("hidden");
  ingestToggle.textContent = isHidden ? "+ Ingest new reports" : "− Hide ingest panel";
});

async function pollIngestJob(jobId) {
  const maxAttempts = 60; // ~3 minutes at 3s intervals
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    await new Promise((resolve) => setTimeout(resolve, 3000));

    let data;
    try {
      const res = await fetch(`${API_BASE}/ingest/${jobId}`);
      data = await res.json();
    } catch (err) {
      ingestStatusEl.textContent = `Lost track of the job: ${err.message}`;
      return;
    }

    if (data.status === "completed") {
      ingestStatusEl.textContent =
        `Done — ${data.documents_downloaded} PDF(s) downloaded, ` +
        `${data.chunks_written} chunk(s) indexed.`;
      checkHealth();
      return;
    }
    if (data.status === "failed") {
      ingestStatusEl.textContent = `Failed: ${data.error || "unknown error"}`;
      return;
    }
    ingestStatusEl.textContent = `Status: ${data.status}…`;
  }
  ingestStatusEl.textContent = "Still running in the background — check back later.";
}

ingestBtn.addEventListener("click", async () => {
  const startDate = ingestFrom.value;
  const endDate = ingestTo.value;

  if (!startDate || !endDate) {
    ingestStatusEl.textContent = "Pick both a from date and a to date.";
    return;
  }
  if (endDate < startDate) {
    ingestStatusEl.textContent = "The to date must be on or after the from date.";
    return;
  }

  ingestBtn.disabled = true;
  ingestStatusEl.textContent = "Starting…";

  try {
    const res = await fetch(`${API_BASE}/ingest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ start_date: startDate, end_date: endDate }),
    });

    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    ingestStatusEl.textContent = "Started — this can take a while depending on the range.";
    pollIngestJob(data.job_id);
  } catch (err) {
    ingestStatusEl.textContent = `Error: ${err.message}`;
  } finally {
    ingestBtn.disabled = false;
  }
});

checkHealth();
