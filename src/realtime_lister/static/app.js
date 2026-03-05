const els = {
  statusPill: document.getElementById("status-pill"),
  latencyChip: document.getElementById("latency-chip"),
  sourceLabel: document.getElementById("source-label"),
  sourceText: document.getElementById("source-text"),
  targetLabel: document.getElementById("target-label"),
  targetText: document.getElementById("target-text"),
  asrMode: document.getElementById("asr-mode"),
  controlNote: document.getElementById("control-note"),
  inputDeviceSelect: document.getElementById("input-device-select"),
  translationPromptInput: document.getElementById("translation-prompt-input"),
  asrSource: document.getElementById("asr-source"),
  asrBeam: document.getElementById("asr-beam"),
  direction: document.getElementById("direction"),
  audioInputState: document.getElementById("audio-input-state"),
  translatorState: document.getElementById("translator-state"),
  speakerSplitState: document.getElementById("speaker-split-state"),
  modelStatusCard: document.getElementById("model-status-card"),
  modelStatusText: document.getElementById("model-status-text"),
  historyCount: document.getElementById("history-count"),
  historyList: document.getElementById("history-list"),
  startBtn: document.getElementById("start-btn"),
  stopBtn: document.getElementById("stop-btn"),
  savePromptBtn: document.getElementById("save-prompt-btn"),
  clearBtn: document.getElementById("clear-btn"),
};

let currentState = null;
let promptFocused = false;
let promptDirty = false;
els.savePromptBtn.disabled = true;

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
            <span>${escapeHtml(item.speaker_label || "Speaker ?")}</span>
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
  const noInputs = Boolean(state.inputDevicesError) || state.inputDevices.length === 0;
  els.statusPill.textContent = state.statusMessage;
  els.statusPill.className = `status-pill ${statusClass(state.statusLevel)}`;
  els.latencyChip.textContent = current ? `${current.translate_ms} ms` : "Waiting";
  els.asrMode.textContent = `ASR · ${state.asrStrategyName || "Unknown Strategy"}`;
  els.asrSource.textContent = state.asrModelSource;
  els.asrBeam.textContent = String(state.asrBeamSize);
  const currentSourceLanguage = current?.source_language || state.sourceLanguage;
  els.direction.textContent = `${currentSourceLanguage} → ${state.targetLanguage}`;
  els.audioInputState.textContent = state.selectedInputDeviceLabel || state.selectedInputDevice;
  els.translatorState.textContent = state.translatorEnabled ? state.translationModel : "ASR only";
  els.speakerSplitState.textContent = state.speakerSplitEnabled
    ? `On (max ${state.speakerMaxSpeakers})`
    : "Off";
  els.modelStatusText.textContent = state.asrModelStatusMessage;
  els.modelStatusCard.className = `status-card ${statusClass(state.asrModelStatusLevel)}`;
  const speakerLabel = current?.speaker_label || "";
  const sourceLangLabel = (currentSourceLanguage || "auto").toUpperCase();
  els.sourceLabel.textContent = speakerLabel
    ? `${sourceLangLabel} Source · ${speakerLabel}`
    : `${sourceLangLabel} Source`;
  els.targetLabel.textContent = `${state.targetLanguage.toUpperCase()} Translation`;
  els.sourceText.textContent = current?.source_text || "Start the session to begin showing source speech.";
  els.targetText.textContent = current?.translated_text || "Live translation will appear here.";
  els.historyCount.textContent = `${state.history.length} item${state.history.length === 1 ? "" : "s"}`;
  els.controlNote.textContent =
    state.inputDevicesError ||
    (noInputs ? "No audio input device available." : "") ||
    state.lastError ||
    state.lastInfo ||
    state.statusMessage;

  els.startBtn.disabled = state.running || state.loading || noInputs || state.asrModelStatusLevel === "error";
  els.stopBtn.disabled = !state.running && !state.loading;
  els.clearBtn.disabled = state.running || state.loading ? false : state.history.length === 0;
  els.inputDeviceSelect.disabled = state.running || state.loading || noInputs;
  if (!promptFocused && !promptDirty) {
    els.translationPromptInput.value = state.translationPromptTemplate || "";
  }
  els.savePromptBtn.disabled = !promptDirty;

  renderInputDevices(state);

  renderHistory(state.history);
}

function renderInputDevices(state) {
  const options = [
    `<option value="auto">Auto</option>`,
    ...state.inputDevices.map(
      (item) =>
        `<option value="${escapeHtml(item.id)}">${escapeHtml(item.name)}${item.isDefault ? " (Default)" : ""}</option>`
    ),
  ];
  const selectedKnown =
    state.selectedInputDevice === "auto" ||
    state.inputDevices.some((item) => item.id === state.selectedInputDevice);
  if (!selectedKnown && state.selectedInputDevice) {
    options.push(
      `<option value="${escapeHtml(state.selectedInputDevice)}">${escapeHtml(state.selectedInputDeviceLabel || state.selectedInputDevice)}</option>`
    );
  }
  els.inputDeviceSelect.innerHTML = options.join("");
  els.inputDeviceSelect.value = state.selectedInputDevice || "auto";
  if (!state.inputDevices.length) {
    els.inputDeviceSelect.innerHTML = `<option value="auto">No input device available</option>`;
    els.inputDeviceSelect.value = "auto";
  }
}

async function postAction(path) {
  const response = await fetch(path, { method: "POST" });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.message || "Request failed.");
  }
  return payload;
}

async function postJson(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
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

  els.inputDeviceSelect.addEventListener("change", async (event) => {
    try {
      await postJson("/api/device", { device: event.target.value });
    } catch (error) {
      els.controlNote.textContent = error.message;
    }
  });

  els.translationPromptInput.addEventListener("focus", () => {
    promptFocused = true;
  });
  els.translationPromptInput.addEventListener("blur", () => {
    promptFocused = false;
  });
  els.translationPromptInput.addEventListener("input", () => {
    if (!currentState) {
      promptDirty = true;
      els.savePromptBtn.disabled = false;
      return;
    }
    promptDirty = els.translationPromptInput.value !== (currentState.translationPromptTemplate || "");
    els.savePromptBtn.disabled = !promptDirty;
  });

  els.savePromptBtn.addEventListener("click", async () => {
    try {
      await postJson("/api/translation-prompt", { template: els.translationPromptInput.value });
      promptDirty = false;
      els.savePromptBtn.disabled = true;
      els.controlNote.textContent = "Translation prompt saved.";
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
