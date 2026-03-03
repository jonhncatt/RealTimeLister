const els = {
  statusPill: document.getElementById("status-pill"),
  latencyChip: document.getElementById("latency-chip"),
  sourceLabel: document.getElementById("source-label"),
  sourceText: document.getElementById("source-text"),
  targetLabel: document.getElementById("target-label"),
  targetText: document.getElementById("target-text"),
  asrMode: document.getElementById("asr-mode"),
  controlNote: document.getElementById("control-note"),
  asrSource: document.getElementById("asr-source"),
  asrBeam: document.getElementById("asr-beam"),
  direction: document.getElementById("direction"),
  translatorState: document.getElementById("translator-state"),
  historyCount: document.getElementById("history-count"),
  historyList: document.getElementById("history-list"),
  startBtn: document.getElementById("start-btn"),
  stopBtn: document.getElementById("stop-btn"),
  clearBtn: document.getElementById("clear-btn"),
};

let currentState = null;

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderHistory(history) {
  if (!history.length) {
    els.historyList.innerHTML = '<div class="history-empty">No transcript yet.</div>';
    return;
  }

  els.historyList.innerHTML = history
    .slice()
    .reverse()
    .map(
      (item) => `
        <article class="history-item">
          <div class="history-meta">
            <span>${escapeHtml(item.timestamp)}</span>
            <span>${escapeHtml(String(item.translate_ms))} ms</span>
          </div>
          <p class="history-source">${escapeHtml(item.source_text)}</p>
          <p class="history-target">${escapeHtml(item.translated_text)}</p>
        </article>
      `
    )
    .join("");
}

function statusClass(level) {
  return `status-${level || "idle"}`;
}

function applyState(state) {
  currentState = state;
  const current = state.current;
  els.statusPill.textContent = state.statusMessage;
  els.statusPill.className = `status-pill ${statusClass(state.statusLevel)}`;
  els.latencyChip.textContent = current ? `${current.translate_ms} ms` : "Waiting";
  els.asrMode.textContent =
    state.asrResolutionMode === "directory" ? "ASR · Fixed Local Dir" : "ASR · Model Name + Cache";
  els.asrSource.textContent = state.asrModelSource;
  els.asrBeam.textContent = String(state.asrBeamSize);
  els.direction.textContent = `${state.sourceLanguage} → ${state.targetLanguage}`;
  els.translatorState.textContent = state.translatorEnabled ? state.translationModel : "ASR only";
  els.sourceLabel.textContent = `${state.sourceLanguage.toUpperCase()} Source`;
  els.targetLabel.textContent = `${state.targetLanguage.toUpperCase()} Translation`;
  els.sourceText.textContent = current?.source_text || "Start the session to begin showing source speech.";
  els.targetText.textContent = current?.translated_text || "Live translation will appear here.";
  els.historyCount.textContent = `${state.history.length} item${state.history.length === 1 ? "" : "s"}`;
  els.controlNote.textContent = state.lastError || state.lastInfo || state.statusMessage;

  els.startBtn.disabled = state.running || state.loading;
  els.stopBtn.disabled = !state.running && !state.loading;
  els.clearBtn.disabled = state.running || state.loading ? false : state.history.length === 0;

  renderHistory(state.history);
}

async function postAction(path) {
  const response = await fetch(path, { method: "POST" });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.message || "Request failed.");
  }
  return payload;
}

function bindControls() {
  els.startBtn.addEventListener("click", async () => {
    try {
      await postAction("/api/start");
    } catch (error) {
      els.controlNote.textContent = error.message;
    }
  });

  els.stopBtn.addEventListener("click", async () => {
    try {
      await postAction("/api/stop");
    } catch (error) {
      els.controlNote.textContent = error.message;
    }
  });

  els.clearBtn.addEventListener("click", async () => {
    try {
      await postAction("/api/clear");
    } catch (error) {
      els.controlNote.textContent = error.message;
    }
  });
}

async function loadInitialState() {
  const response = await fetch("/api/state");
  const payload = await response.json();
  applyState(payload);
}

function connectEvents() {
  const eventSource = new EventSource("/api/events");
  eventSource.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    if (payload.type === "snapshot") {
      applyState(payload.state);
      return;
    }

    if (payload.type === "transcript" && currentState) {
      const history = [...currentState.history, payload.item].slice(-120);
      applyState({ ...currentState, history, current: payload.item });
      return;
    }

    if (payload.type === "status" && currentState) {
      applyState({
        ...currentState,
        statusLevel: payload.level,
        statusMessage: payload.message,
        lastInfo: payload.level === "error" ? currentState.lastInfo : payload.message,
        lastError: payload.level === "error" ? payload.message : currentState.lastError,
      });
      return;
    }

    if (payload.type === "info" && currentState) {
      applyState({ ...currentState, lastInfo: payload.message });
      return;
    }

    if (payload.type === "cleared" && currentState) {
      applyState({ ...currentState, history: [], current: null });
    }
  };

  eventSource.onerror = () => {
    if (currentState) {
      els.controlNote.textContent = "Connection interrupted. Retrying...";
    }
  };
}

bindControls();
loadInitialState().then(connectEvents).catch((error) => {
  els.controlNote.textContent = error.message;
});
