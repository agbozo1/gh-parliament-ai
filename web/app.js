// Same-origin by default (works when FastAPI serves this page directly).
// If you host this page separately (e.g. on Netlify) from the API, set
// this to the API's base URL, e.g. "https://your-api.onrender.com".
const API_BASE = "";

const statusEl = document.getElementById("status");
const statusTextEl = document.getElementById("status-text");
const form = document.getElementById("query-form");
const questionInput = document.getElementById("question");
const submitBtn = document.getElementById("submit-btn");
const chatLogEl = document.getElementById("chat-log");

// The index is kept up to date automatically (full backfill once, then a
// daily check for newly published sittings — see pipeline/sync.py). There's
// no ingest control here on purpose: by the time anyone opens this page,
// the data they'd search is expected to already be indexed.
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    statusEl.className = "status status--ok";
    const freshness = data.latest_sitting_date
      ? ` · latest sitting indexed: ${data.latest_sitting_date}`
      : "";
    statusTextEl.textContent = `Online — ${data.documents_indexed} chunk(s) indexed${freshness}`;
  } catch (err) {
    statusEl.className = "status status--error";
    statusTextEl.textContent = "API unreachable";
  }
}

function scrollToLatest() {
  form.scrollIntoView({ behavior: "smooth", block: "end" });
}

function appendUserTurn(question) {
  const turn = document.createElement("div");
  turn.className = "chat-turn chat-turn--user";

  const bubble = document.createElement("div");
  bubble.className = "chat-bubble chat-bubble--user";
  bubble.textContent = question;

  turn.appendChild(bubble);
  chatLogEl.appendChild(turn);
}

function appendAssistantPendingTurn() {
  const turn = document.createElement("div");
  turn.className = "chat-turn chat-turn--assistant";

  const bubble = document.createElement("div");
  bubble.className = "chat-bubble chat-bubble--assistant chat-bubble--pending";
  bubble.textContent = "Searching the record…";

  turn.appendChild(bubble);
  chatLogEl.appendChild(turn);
  return bubble;
}

function renderSourcesInto(container, sources) {
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

    container.appendChild(row);
  }
}

function resolveAssistantTurn(bubble, { answer, sources }) {
  bubble.classList.remove("chat-bubble--pending");
  bubble.textContent = "";

  const answerEl = document.createElement("p");
  answerEl.className = "answer";
  answerEl.textContent = answer;
  bubble.appendChild(answerEl);

  const sourcesEl = document.createElement("div");
  sourcesEl.className = "sources";
  renderSourcesInto(sourcesEl, sources);
  bubble.appendChild(sourcesEl);
}

function failAssistantTurn(bubble, message) {
  bubble.classList.remove("chat-bubble--pending");
  bubble.classList.add("chat-bubble--error");
  bubble.textContent = message;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const question = questionInput.value.trim();
  if (!question) return;

  document.body.classList.add("chat-active");
  appendUserTurn(question);
  const pendingBubble = appendAssistantPendingTurn();
  scrollToLatest();

  questionInput.value = "";
  submitBtn.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    const data = await res.json().catch(() => ({}));

    if (!res.ok) {
      throw new Error(data.detail || `Request failed (HTTP ${res.status})`);
    }

    resolveAssistantTurn(pendingBubble, data);
  } catch (err) {
    failAssistantTurn(pendingBubble, `Something went wrong: ${err.message}`);
  } finally {
    submitBtn.disabled = false;
    scrollToLatest();
  }
});

checkHealth();
