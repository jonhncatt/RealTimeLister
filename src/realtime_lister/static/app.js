const els = {
  statusPill: document.getElementById("status-pill"),
  latencyChip: document.getElementById("latency-chip"),
  sourceLabel: document.getElementById("source-label"),
  sourceText: document.getElementById("source-text"),
  targetLabel: document.getElementById("target-label"),
  targetText: document.getElementById("target-text"),
  asrMode: document.getElementById("asr-mode"),
  nextStepText: document.getElementById("next-step-text"),
  runPlanText: document.getElementById("run-plan-text"),
  modelCheckCard: document.getElementById("model-check-card"),
  modelCheckState: document.getElementById("model-check-state"),
  modelCheckDetail: document.getElementById("model-check-detail"),
  micCheckCard: document.getElementById("mic-check-card"),
  micCheckState: document.getElementById("mic-check-state"),
  micCheckDetail: document.getElementById("mic-check-detail"),
  translatorCheckCard: document.getElementById("translator-check-card"),
  translatorCheckState: document.getElementById("translator-check-state"),
  translatorCheckDetail: document.getElementById("translator-check-detail"),
  controlNote: document.getElementById("control-note"),
  sourceLanguageSelect: document.getElementById("source-language-select"),
  targetLanguageSelect: document.getElementById("target-language-select"),
  inputDeviceSelect: document.getElementById("input-device-select"),
  advancedPanel: document.getElementById("advanced-panel"),
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
  resetPromptBtn: document.getElementById("reset-prompt-btn"),
  clearBtn: document.getElementById("clear-btn"),
};

let currentState = null;
let promptFocused = false;
let promptDirty = false;
els.savePromptBtn.disabled = true;
els.resetPromptBtn.disabled = true;

const LANGUAGE_OPTIONS = [
  { value: "auto", label: "Auto Detect" },
  { value: "zh", label: "Chinese" },
  { value: "en", label: "English" },
  { value: "ja", label: "Japanese" },
  { value: "ko", label: "Korean" },
  { value: "fr", label: "French" },
  { value: "de", label: "German" },
];

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

function setupToneClass(tone) {
  return `setup-tone-${tone || "warn"}`;
}

function languageDisplay(value) {
  if (!value || value === "auto") {
    return "Auto Detect";
  }
  const match = LANGUAGE_OPTIONS.find((item) => item.value === String(value).toLowerCase());
  return match ? match.label : String(value).toUpperCase();
}

function renderLanguageOptions(selectEl, value, { allowAuto }) {
  const baseOptions = LANGUAGE_OPTIONS.filter((item) => allowAuto || item.value !== "auto");
  const normalized = String(value || (allowAuto ? "auto" : "en")).toLowerCase();
  const options = [...baseOptions];
  if (!options.some((item) => item.value === normalized)) {
    options.push({ value: normalized, label: normalized.toUpperCase() });
  }
  selectEl.innerHTML = options
    .map((item) => `<option value="${escapeHtml(item.value)}">${escapeHtml(item.label)}</option>`)
    .join("");
  selectEl.value = normalized;
}

function setGuideItem(card, stateEl, detailEl, tone, stateText, detailText) {
  card.className = `guide-item ${setupToneClass(tone)}`;
  stateEl.textContent = stateText;
  detailEl.textContent = detailText;
}

function renderGuide(state, noInputs) {
  const shortAsrSource = String(state.asrModelSource || "-").split(/[\\/]/).pop();
  const sourcePlan = languageDisplay(state.sourceLanguage);
  const targetPlan = languageDisplay(state.targetLanguage);
  const translatorPlan = state.translatorEnabled ? state.translationModel : "ASR only";
  els.runPlanText.textContent = `${sourcePlan} -> ${targetPlan} · ${shortAsrSource} · ${state.selectedInputDeviceLabel} · ${translatorPlan}`;

  if (state.running) {
    els.nextStepText.textContent = "Session is live. Speak near the microphone.";
  } else if (state.loading) {
    els.nextStepText.textContent = "Loading model and preparing microphone...";
  } else if (state.asrModelStatusLevel === "error") {
    els.nextStepText.textContent = "Next: fix the ASR model first. Set RT_ASR_MODEL_DIR or allow model download.";
  } else if (noInputs) {
    els.nextStepText.textContent = "Next: connect a microphone or choose another input device.";
  } else if (!state.translatorEnabled) {
    els.nextStepText.textContent = "Ready for ASR only. Add OPENAI_API_KEY if you also want translation.";
  } else {
    els.nextStepText.textContent = "Ready. Click Start Listening.";
  }

  const modelTone = state.asrModelStatusLevel === "error" ? "error" : state.asrModelStatusLevel === "ready" ? "ready" : "warn";
  const modelState = modelTone === "ready" ? "Ready" : modelTone === "error" ? "Action Needed" : "Check";
  setGuideItem(els.modelCheckCard, els.modelCheckState, els.modelCheckDetail, modelTone, modelState, state.asrModelStatusMessage);

  const micTone = noInputs ? "error" : "ready";
  const micState = noInputs ? "Missing" : "Ready";
  const micDetail = state.inputDevicesError || state.selectedInputDeviceLabel || "No input device available.";
  setGuideItem(els.micCheckCard, els.micCheckState, els.micCheckDetail, micTone, micState, micDetail);

  const translatorTone = state.translatorEnabled ? "ready" : "warn";
  const translatorState = state.translatorEnabled ? "Ready" : "Optional";
  const translatorDetail = state.translatorEnabled
    ? `Using ${state.translationModel}`
    : "OPENAI_API_KEY not set. Start still works, but only ASR will run.";
  setGuideItem(
    els.translatorCheckCard,
    els.translatorCheckState,
    els.translatorCheckDetail,
    translatorTone,
    translatorState,
    translatorDetail
  );
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

  if (state.loading) {
    els.startBtn.textContent = "Loading...";
  } else if (state.running) {
    els.startBtn.textContent = "Listening...";
  } else if (state.asrModelStatusLevel === "error") {
    els.startBtn.textContent = "Model Not Ready";
  } else if (noInputs) {
    els.startBtn.textContent = "No Microphone";
  } else {
    els.startBtn.textContent = "Start Listening";
  }
  els.startBtn.disabled = state.running || state.loading || noInputs || state.asrModelStatusLevel === "error";
  els.stopBtn.disabled = !state.running && !state.loading;
  els.clearBtn.disabled = state.running || state.loading ? false : state.history.length === 0;
  els.sourceLanguageSelect.disabled = state.running || state.loading;
  els.targetLanguageSelect.disabled = state.running || state.loading;
  els.inputDeviceSelect.disabled = state.running || state.loading || noInputs;
  els.translationPromptInput.disabled = state.running || state.loading;
  els.resetPromptBtn.disabled = state.running || state.loading;
  if (!promptFocused && !promptDirty) {
    els.translationPromptInput.value = state.translationPromptTemplate || "";
  }
  els.savePromptBtn.disabled = state.running || state.loading || !promptDirty;

  renderLanguageOptions(els.sourceLanguageSelect, state.sourceLanguage, { allowAuto: true });
  renderLanguageOptions(els.targetLanguageSelect, state.targetLanguage, { allowAuto: false });
  renderInputDevices(state);
  renderGuide(state, noInputs);
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

  const handleLanguageChange = async () => {
    try {
      await postJson("/api/languages", {
        sourceLanguage: els.sourceLanguageSelect.value,
        targetLanguage: els.targetLanguageSelect.value,
      });
    } catch (error) {
      els.controlNote.textContent = error.message;
    }
  };

  els.sourceLanguageSelect.addEventListener("change", handleLanguageChange);
  els.targetLanguageSelect.addEventListener("change", handleLanguageChange);

  els.translationPromptInput.addEventListener("focus", () => {
    promptFocused = true;
    els.advancedPanel.open = true;
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

  els.resetPromptBtn.addEventListener("click", async () => {
    try {
      await postJson("/api/translation-prompt", { template: "" });
      promptDirty = false;
      els.savePromptBtn.disabled = true;
      els.translationPromptInput.value = currentState?.defaultTranslationPromptTemplate || "";
      els.controlNote.textContent = "Translation prompt reset to default.";
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
