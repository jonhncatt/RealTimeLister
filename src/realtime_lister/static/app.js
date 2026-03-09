const els = {
  statusPill: document.getElementById("status-pill"),
  enginePanel: document.getElementById("engine-panel"),
  engineState: document.getElementById("engine-state"),
  engineDetail: document.getElementById("engine-detail"),
  sessionPanel: document.getElementById("session-panel"),
  sessionState: document.getElementById("session-state"),
  sessionDetail: document.getElementById("session-detail"),
  latencyChip: document.getElementById("latency-chip"),
  sourceLabel: document.getElementById("source-label"),
  sourceText: document.getElementById("source-text"),
  targetLabel: document.getElementById("target-label"),
  targetText: document.getElementById("target-text"),
  asrMode: document.getElementById("asr-mode"),
  engineChip: document.getElementById("engine-chip"),
  sessionChip: document.getElementById("session-chip"),
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
  asrHotwordsInput: document.getElementById("asr-hotwords-input"),
  translationPromptInput: document.getElementById("translation-prompt-input"),
  glossaryInput: document.getElementById("glossary-input"),
  asrSource: document.getElementById("asr-source"),
  asrBeam: document.getElementById("asr-beam"),
  asrHotwordsState: document.getElementById("asr-hotwords-state"),
  asrRuntimeState: document.getElementById("asr-runtime-state"),
  direction: document.getElementById("direction"),
  audioInputState: document.getElementById("audio-input-state"),
  translatorState: document.getElementById("translator-state"),
  glossaryState: document.getElementById("glossary-state"),
  lastActivityState: document.getElementById("last-activity-state"),
  speakerSplitState: document.getElementById("speaker-split-state"),
  modelStatusCard: document.getElementById("model-status-card"),
  modelStatusText: document.getElementById("model-status-text"),
  historyCount: document.getElementById("history-count"),
  historyList: document.getElementById("history-list"),
  startBtn: document.getElementById("start-btn"),
  stopBtn: document.getElementById("stop-btn"),
  savePromptBtn: document.getElementById("save-prompt-btn"),
  resetPromptBtn: document.getElementById("reset-prompt-btn"),
  uploadAsrHotwordsBtn: document.getElementById("upload-asr-hotwords-btn"),
  saveAsrHotwordsBtn: document.getElementById("save-asr-hotwords-btn"),
  clearAsrHotwordsBtn: document.getElementById("clear-asr-hotwords-btn"),
  asrHotwordsFileInput: document.getElementById("asr-hotwords-file-input"),
  uploadGlossaryBtn: document.getElementById("upload-glossary-btn"),
  saveGlossaryBtn: document.getElementById("save-glossary-btn"),
  clearGlossaryBtn: document.getElementById("clear-glossary-btn"),
  glossaryFileInput: document.getElementById("glossary-file-input"),
  clearBtn: document.getElementById("clear-btn"),
};

let currentState = null;
let promptFocused = false;
let promptDirty = false;
let asrHotwordsFocused = false;
let asrHotwordsDirty = false;
let glossaryFocused = false;
let glossaryDirty = false;
els.savePromptBtn.disabled = true;
els.resetPromptBtn.disabled = true;
els.saveAsrHotwordsBtn.disabled = true;
els.saveGlossaryBtn.disabled = true;

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

function normalizeMultilineText(text) {
  return String(text || "").replace(/\r\n?/g, "\n").trim();
}

function termLineCount(text) {
  return normalizeMultilineText(text)
    .split("\n")
    .filter((line) => line.trim()).length;
}

function buildHistoryTurns(history) {
  const turns = [];
  for (const item of history) {
    const sourceText = String(item.source_text || "").trim();
    const targetText = String(item.translated_text || "").trim();
    const speakerLabel = item.speaker_label || "Speaker ?";
    const sourceLanguage = item.source_language || "";
    const targetLanguage = item.target_language || "";
    const last = turns[turns.length - 1];
    const sameTurn =
      last &&
      last.speakerLabel === speakerLabel &&
      last.sourceLanguage === sourceLanguage &&
      last.targetLanguage === targetLanguage &&
      last.segmentCount < 8;

    if (sameTurn) {
      if (sourceText) {
        last.sourceParts.push(sourceText);
      }
      if (targetText) {
        last.targetParts.push(targetText);
      }
      last.segmentCount += 1;
      last.endTimestamp = item.timestamp;
      last.lastTranslateMs = item.translate_ms;
      continue;
    }

    turns.push({
      speakerLabel,
      sourceLanguage,
      targetLanguage,
      startTimestamp: item.timestamp,
      endTimestamp: item.timestamp,
      segmentCount: 1,
      lastTranslateMs: item.translate_ms,
      sourceParts: sourceText ? [sourceText] : [],
      targetParts: targetText ? [targetText] : [],
    });
  }
  return turns;
}

function renderHistory(history, translatorEnabled) {
  if (!history.length) {
    els.historyList.innerHTML = '<div class="history-empty">No transcript yet.</div>';
    return { turnCount: 0, segmentCount: 0 };
  }

  const shouldStick =
    els.historyList.scrollTop + els.historyList.clientHeight >= els.historyList.scrollHeight - 48;
  const turns = buildHistoryTurns(history);

  els.historyList.innerHTML = turns
    .map(
      (turn) => `
        <article class="history-turn">
          <div class="history-turn-meta">
            <span class="history-turn-speaker">${escapeHtml(turn.speakerLabel)}</span>
            <span>${escapeHtml(turn.startTimestamp)}${turn.endTimestamp !== turn.startTimestamp ? ` → ${escapeHtml(turn.endTimestamp)}` : ""}</span>
            <span>${escapeHtml(String(turn.segmentCount))} seg</span>
          </div>
          <div class="history-stream">
            <p class="history-stream-source">
              <span class="history-stream-label">${escapeHtml((turn.sourceLanguage || "auto").toUpperCase())}</span>
              ${escapeHtml(turn.sourceParts.join(" "))}
            </p>
            <p class="history-stream-target ${translatorEnabled ? "" : "mirror-mode"}">
              <span class="history-stream-label">${escapeHtml((turn.targetLanguage || "out").toUpperCase())}</span>
              ${escapeHtml(turn.targetParts.join(" "))}
            </p>
          </div>
        </article>
      `
    )
    .join("");

  if (shouldStick) {
    els.historyList.scrollTop = els.historyList.scrollHeight;
  }
  return { turnCount: turns.length, segmentCount: history.length };
}

function statusClass(level) {
  return `status-${level || "idle"}`;
}

function setupToneClass(tone) {
  return `setup-tone-${tone || "warn"}`;
}

function describeEngineState(state) {
  if (state.loading) {
    return {
      chip: "Model Load",
      value: "Loading Model",
      detail: "ASR engine is being loaded into memory and the microphone is being armed.",
      tone: "loading",
    };
  }
  if (state.asrModelStatusLevel === "error") {
    return {
      chip: "Model Error",
      value: "Model Missing",
      detail: state.asrModelStatusMessage,
      tone: "error",
    };
  }
  if (state.asrModelStatusLevel === "ready") {
    return {
      chip: "Model Ready",
      value: state.running ? "Loaded + Live" : "Loaded + Standby",
      detail: state.asrModelStatusMessage,
      tone: state.running ? "running" : "ok",
    };
  }
  return {
    chip: "Model Check",
    value: "Checking Model",
    detail: state.asrModelStatusMessage || "Inspecting model configuration.",
    tone: "loading",
  };
}

function describeSessionState(state) {
  const last = state.current || state.history[state.history.length - 1] || null;
  if (state.loading) {
    return {
      chip: "Arming",
      value: "Preparing Session",
      detail: "Opening the selected audio input and warming up the ASR pipeline.",
      tone: "loading",
      metric: "Preparing",
    };
  }
  if (state.running) {
    return {
      chip: "Session Live",
      value: "Listening",
      detail: last
        ? `Live capture is active. Last segment at ${last.timestamp} from ${last.speaker_label || "current speaker"}.`
        : "Live capture is active. Waiting for speech to cross the VAD threshold.",
      tone: "running",
      metric: last ? `Live · last ${last.timestamp}` : "Live",
    };
  }
  if (last) {
    return {
      chip: "Standby",
      value: "Idle with History",
      detail: `Last segment at ${last.timestamp} from ${last.speaker_label || "speaker"}. Session is stopped right now.`,
      tone: "idle",
      metric: `Standby · ${state.history.length} seg`,
    };
  }
  return {
    chip: "Session Idle",
    value: "Idle",
    detail: "No active capture. Start the session to load the microphone feed into the ASR engine.",
    tone: "idle",
    metric: "Idle",
  };
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
  const asrHotwordsPlan = state.asrHotwordsLineCount
    ? ` · ${state.asrHotwordsLineCount} ASR hint${state.asrHotwordsLineCount === 1 ? "" : "s"}`
    : "";
  const translatorPlan = state.translatorEnabled ? state.translationModel : "Mirror only";
  const glossaryPlan = state.glossaryLineCount
    ? ` · ${state.glossaryLineCount} term${state.glossaryLineCount === 1 ? "" : "s"}`
    : "";
  els.runPlanText.textContent = `${sourcePlan} -> ${targetPlan} · ${shortAsrSource} · ${state.selectedInputDeviceLabel}${asrHotwordsPlan} · ${translatorPlan}${glossaryPlan}`;

  if (state.running) {
    els.nextStepText.textContent = "Session is live. Speak near the microphone.";
  } else if (state.loading) {
    els.nextStepText.textContent = "Loading model and preparing microphone...";
  } else if (state.asrModelStatusLevel === "error") {
    els.nextStepText.textContent = "Next: fix the ASR model first. Set RT_ASR_MODEL_DIR or allow model download.";
  } else if (noInputs) {
    els.nextStepText.textContent = "Next: connect a microphone or choose another input device.";
  } else if (!state.translatorEnabled) {
    els.nextStepText.textContent = "Ready for local ASR. Add OPENAI_API_KEY only if you want target-language translation.";
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
    ? `Using ${state.translationModel}${state.glossaryLineCount ? ` + ${state.glossaryLineCount} glossary lines` : ""}`
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
  document.body.dataset.translator = state.translatorEnabled ? "on" : "off";
  document.body.dataset.session = state.running ? "running" : state.loading ? "loading" : "idle";
  const current = state.current;
  const last = current || state.history[state.history.length - 1] || null;
  const noInputs = Boolean(state.inputDevicesError) || state.inputDevices.length === 0;
  const engineStatus = describeEngineState(state);
  const sessionStatus = describeSessionState(state);
  els.statusPill.textContent = state.statusMessage;
  els.statusPill.className = `status-pill ${statusClass(state.statusLevel)}`;
  els.engineChip.textContent = engineStatus.chip;
  els.engineChip.className = statusClass(engineStatus.tone);
  els.sessionChip.textContent = sessionStatus.chip;
  els.sessionChip.className = statusClass(sessionStatus.tone);
  els.enginePanel.className = `runtime-row ${statusClass(engineStatus.tone)}`;
  els.engineState.textContent = engineStatus.value;
  els.engineDetail.textContent = engineStatus.detail;
  els.sessionPanel.className = `runtime-row ${statusClass(sessionStatus.tone)}`;
  els.sessionState.textContent = sessionStatus.value;
  els.sessionDetail.textContent = sessionStatus.detail;
  els.latencyChip.textContent = current ? `${current.translate_ms} ms` : state.running ? "Mic Hot" : "Waiting";
  els.asrMode.textContent = `ASR :: ${state.asrStrategyName || "Unknown Strategy"}`;
  els.asrSource.textContent = state.asrModelSource;
  els.asrBeam.textContent = String(state.asrBeamSize);
  els.asrHotwordsState.textContent = state.asrHotwordsLineCount ? `${state.asrHotwordsLineCount} loaded` : "None";
  els.asrRuntimeState.textContent = sessionStatus.metric;
  const currentSourceLanguage = current?.source_language || state.sourceLanguage;
  els.direction.textContent = `${currentSourceLanguage} → ${state.targetLanguage}`;
  els.audioInputState.textContent = state.selectedInputDeviceLabel || state.selectedInputDevice;
  els.translatorState.textContent = state.translatorEnabled ? state.translationModel : "ASR only";
  els.glossaryState.textContent = state.glossaryLineCount ? `${state.glossaryLineCount} loaded` : "None";
  els.lastActivityState.textContent = last
    ? `${last.timestamp} · ${last.speaker_label || "speaker"}`
    : state.running
      ? "Live · waiting for speech"
      : "No activity yet";
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
  els.targetLabel.textContent = state.translatorEnabled
    ? `${state.targetLanguage.toUpperCase()} Translation`
    : "ASR Mirror";
  els.sourceText.textContent = current?.source_text || "Arm the session and speak into the mic. Captured speech will scroll in here first.";
  els.targetText.textContent =
    current?.translated_text ||
    (state.translatorEnabled
      ? "Translation output lands here when the translator is online."
      : "No API key set. This pane mirrors the transcript until translation is enabled.");
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
  els.targetLanguageSelect.disabled = state.running || state.loading || !state.translatorEnabled;
  els.inputDeviceSelect.disabled = state.running || state.loading || noInputs;
  els.translationPromptInput.disabled = state.running || state.loading;
  els.resetPromptBtn.disabled = state.running || state.loading;
  els.asrHotwordsInput.disabled = false;
  els.uploadAsrHotwordsBtn.disabled = false;
  els.clearAsrHotwordsBtn.disabled = false;
  els.glossaryInput.disabled = false;
  els.uploadGlossaryBtn.disabled = false;
  els.clearGlossaryBtn.disabled = false;
  if (!asrHotwordsFocused && !asrHotwordsDirty) {
    els.asrHotwordsInput.value = state.asrHotwords || "";
  }
  if (!promptFocused && !promptDirty) {
    els.translationPromptInput.value = state.translationPromptTemplate || "";
  }
  if (!glossaryFocused && !glossaryDirty) {
    els.glossaryInput.value = state.glossary || "";
  }
  els.savePromptBtn.disabled = state.running || state.loading || !promptDirty;
  els.saveAsrHotwordsBtn.disabled = !asrHotwordsDirty;
  els.saveGlossaryBtn.disabled = !glossaryDirty;

  renderLanguageOptions(els.sourceLanguageSelect, state.sourceLanguage, { allowAuto: true });
  renderLanguageOptions(els.targetLanguageSelect, state.targetLanguage, { allowAuto: false });
  renderInputDevices(state);
  renderGuide(state, noInputs);
  const historySummary = renderHistory(state.history, state.translatorEnabled);
  els.historyCount.textContent = `${historySummary.turnCount} turns · ${historySummary.segmentCount} seg`;
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

  els.asrHotwordsInput.addEventListener("focus", () => {
    asrHotwordsFocused = true;
    els.advancedPanel.open = true;
  });
  els.asrHotwordsInput.addEventListener("blur", () => {
    asrHotwordsFocused = false;
  });
  els.asrHotwordsInput.addEventListener("input", () => {
    if (!currentState) {
      asrHotwordsDirty = true;
      els.saveAsrHotwordsBtn.disabled = false;
      return;
    }
    asrHotwordsDirty =
      normalizeMultilineText(els.asrHotwordsInput.value) !== normalizeMultilineText(currentState.asrHotwords || "");
    els.saveAsrHotwordsBtn.disabled = !asrHotwordsDirty;
  });

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

  els.glossaryInput.addEventListener("focus", () => {
    glossaryFocused = true;
    els.advancedPanel.open = true;
  });
  els.glossaryInput.addEventListener("blur", () => {
    glossaryFocused = false;
  });
  els.glossaryInput.addEventListener("input", () => {
    if (!currentState) {
      glossaryDirty = true;
      els.saveGlossaryBtn.disabled = false;
      return;
    }
    glossaryDirty = normalizeMultilineText(els.glossaryInput.value) !== normalizeMultilineText(currentState.glossary || "");
    els.saveGlossaryBtn.disabled = !glossaryDirty;
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

  els.uploadAsrHotwordsBtn.addEventListener("click", () => {
    els.asrHotwordsFileInput.click();
  });

  els.asrHotwordsFileInput.addEventListener("change", async (event) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    try {
      const hotwords = normalizeMultilineText(await file.text());
      await postJson("/api/asr-hotwords", { hotwords });
      asrHotwordsDirty = false;
      els.saveAsrHotwordsBtn.disabled = true;
      els.asrHotwordsInput.value = hotwords;
      els.controlNote.textContent = `${file.name} imported to ASR hotwords (${termLineCount(hotwords)} lines).`;
    } catch (error) {
      els.controlNote.textContent = error.message;
    } finally {
      event.target.value = "";
    }
  });

  els.saveAsrHotwordsBtn.addEventListener("click", async () => {
    try {
      const hotwords = normalizeMultilineText(els.asrHotwordsInput.value);
      await postJson("/api/asr-hotwords", { hotwords });
      asrHotwordsDirty = false;
      els.saveAsrHotwordsBtn.disabled = true;
      els.controlNote.textContent = hotwords
        ? `ASR hotwords saved (${termLineCount(hotwords)} lines).`
        : "ASR hotwords cleared.";
    } catch (error) {
      els.controlNote.textContent = error.message;
    }
  });

  els.clearAsrHotwordsBtn.addEventListener("click", async () => {
    try {
      await postJson("/api/asr-hotwords", { hotwords: "" });
      asrHotwordsDirty = false;
      els.saveAsrHotwordsBtn.disabled = true;
      els.asrHotwordsInput.value = "";
      els.controlNote.textContent = "ASR hotwords cleared.";
    } catch (error) {
      els.controlNote.textContent = error.message;
    }
  });

  els.uploadGlossaryBtn.addEventListener("click", () => {
    els.glossaryFileInput.click();
  });

  els.glossaryFileInput.addEventListener("change", async (event) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    try {
      const text = normalizeMultilineText(await file.text());
      await postJson("/api/glossary", { glossary: text });
      glossaryDirty = false;
      els.saveGlossaryBtn.disabled = true;
      els.glossaryInput.value = text;
      els.controlNote.textContent = `${file.name} imported (${termLineCount(text)} lines).`;
    } catch (error) {
      els.controlNote.textContent = error.message;
    } finally {
      event.target.value = "";
    }
  });

  els.saveGlossaryBtn.addEventListener("click", async () => {
    try {
      const glossary = normalizeMultilineText(els.glossaryInput.value);
      await postJson("/api/glossary", { glossary });
      glossaryDirty = false;
      els.saveGlossaryBtn.disabled = true;
      els.controlNote.textContent = glossary
        ? `Glossary saved (${termLineCount(glossary)} lines).`
        : "Glossary cleared.";
    } catch (error) {
      els.controlNote.textContent = error.message;
    }
  });

  els.clearGlossaryBtn.addEventListener("click", async () => {
    try {
      await postJson("/api/glossary", { glossary: "" });
      glossaryDirty = false;
      els.saveGlossaryBtn.disabled = true;
      els.glossaryInput.value = "";
      els.controlNote.textContent = "Glossary cleared.";
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
