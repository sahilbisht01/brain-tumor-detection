(async () => {
    const { GoogleGenerativeAI } = await import("https://esm.run/@google/generative-ai");

    // --- STATE ---
    let file = null;
    let resultData = null;
    let simulationMode = false;
    let userApiKey = "";
    let CONFIG_API_KEY = "AIzaSyDDQrauyN82Tpso_Xr97a4wlmG_jgicxBM";

    // --- DOM ELEMENTS ---
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const emptyState = document.getElementById('empty-state');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const btnAnalyze = document.getElementById('btn-analyze');
    const simulationNotice = document.getElementById('simulation-notice');
    const statusPing = document.getElementById('status-ping');
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');

    // Result Elements
    const resultContent = document.getElementById('result-content');
    const resultEmpty = document.getElementById('result-empty');
    const resultLabel = document.getElementById('result-label');
    const resultConfidence = document.getElementById('result-confidence-text');
    const resultBar = document.getElementById('result-bar');
    const resultCard = document.getElementById('result-card');
    const resultIcon = document.getElementById('result-icon');

    // Gemini Elements
    const geminiText = document.getElementById('gemini-text');
    const geminiSkeleton = document.getElementById('gemini-skeleton');
    const geminiError = document.getElementById('gemini-error');
    const geminiLoader = document.getElementById('gemini-loader');
    const modelAttemptName = document.getElementById('model-attempt-name');
    const exportContainer = document.getElementById('export-container');
    const errorTitle = document.getElementById('error-title');
    const errorMsg = document.getElementById('error-msg');
    const apiKeyInput = document.getElementById('api-key-input');
    const btnRetryKey = document.getElementById('btn-retry-key');
    const diagnosticLogContainer = document.getElementById('diagnostic-log-container');
    const diagnosticLog = document.getElementById('diagnostic-log');

    // --- INITIALIZATION ---
    lucide.createIcons();

    // --- EVENT LISTENERS ---
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('bg-cyan-950/30', 'ring-2', 'ring-cyan-500/50');
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.classList.remove('bg-cyan-950/30', 'ring-2', 'ring-cyan-500/50');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('bg-cyan-950/30', 'ring-2', 'ring-cyan-500/50');
        if (e.dataTransfer.files[0]) processFile(e.dataTransfer.files[0]);
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files[0]) processFile(e.target.files[0]);
    });

    document.getElementById('btn-change').addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    document.getElementById('btn-remove').addEventListener('click', (e) => {
        e.stopPropagation();
        clearFile();
    });

    btnAnalyze.addEventListener('click', handleAnalyze);
    btnRetryKey.addEventListener('click', handleRetryWithKey);

    // --- FUNCTIONS ---
    function processFile(selectedFile) {
        file = selectedFile;
        const url = URL.createObjectURL(file);
        imagePreview.src = url;

        emptyState.classList.add('hidden');
        previewContainer.classList.remove('hidden');

        btnAnalyze.disabled = false;
        btnAnalyze.classList.remove('opacity-50', 'cursor-not-allowed', 'bg-slate-800');
        btnAnalyze.classList.add(
            'hover:shadow-[0_0_40px_-10px_rgba(6,182,212,0.5)]',
            'hover:scale-[1.01]',
            'bg-gradient-to-r',
            'from-cyan-500',
            'to-blue-600',
            'cursor-pointer'
        );
        btnAnalyze.querySelector('div').classList.remove('text-slate-500');
        btnAnalyze.querySelector('div').classList.add('text-white');

        resultContent.classList.add('hidden');
        resultEmpty.classList.remove('hidden');
        simulationNotice.classList.add('hidden');
        updateStatus(false);
    }

    function clearFile() {
        file = null;
        fileInput.value = "";
        previewContainer.classList.add('hidden');
        emptyState.classList.remove('hidden');

        btnAnalyze.disabled = true;
        btnAnalyze.classList.add('opacity-50', 'cursor-not-allowed', 'bg-slate-800');
        btnAnalyze.classList.remove(
            'hover:shadow-[0_0_40px_-10px_rgba(6,182,212,0.5)]',
            'hover:scale-[1.01]',
            'bg-gradient-to-r',
            'from-cyan-500',
            'to-blue-600',
            'cursor-pointer'
        );

        resultContent.classList.add('hidden');
        resultEmpty.classList.remove('hidden');
    }

    function updateStatus(isSim) {
        if (isSim) {
            statusPing.classList.replace('bg-emerald-400', 'bg-amber-400');
            statusDot.classList.replace('bg-emerald-500', 'bg-amber-500');
            statusText.textContent = "Simulation Environment";
        } else {
            statusPing.classList.replace('bg-amber-400', 'bg-emerald-400');
            statusDot.classList.replace('bg-amber-500', 'bg-emerald-500');
            statusText.textContent = "Clinical Inference Engine v2.0";
        }
    }

    function addToLog(msg) {
        const div = document.createElement('div');
        div.className = "border-b border-slate-800/50 pb-1 last:border-0";
        div.innerText = `${new Date().toLocaleTimeString()} - ${msg}`;
        diagnosticLog.prepend(div);
    }

    async function handleAnalyze() {
        if (!file) return;

        btnAnalyze.disabled = true;
        const originalBtnContent = btnAnalyze.innerHTML;
        btnAnalyze.innerHTML = `<div class="relative h-full w-full bg-slate-900 rounded-2xl px-8 py-5 flex items-center justify-center gap-3"><i data-lucide="loader-2" class="w-6 h-6 animate-spin text-cyan-400"></i><span class="text-white">Processing Neural Network...</span></div>`;
        lucide.createIcons();

        resultContent.classList.add('hidden');
        resultEmpty.classList.remove('hidden');
        geminiError.classList.add('hidden');
        diagnosticLog.innerHTML = "";

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('http://localhost:5000/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error("Backend failed");

            resultData = await response.json();
            simulationMode = false;
            showResults(resultData);
            generateGeminiResponse(resultData.label, resultData.confidence);
        } catch (error) {
            console.log("Backend unavailable, switching to Simulation");
            simulationMode = true;
            simulationNotice.classList.remove('hidden');
            updateStatus(true);

            setTimeout(() => {
                const classes = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor'];
                const randomClass = classes[Math.floor(Math.random() * classes.length)];
                resultData = {
                    label: randomClass,
                    confidence: `${((0.75 + Math.random() * 0.24) * 100).toFixed(2)}%`
                };
                showResults(resultData);
                generateGeminiResponse(resultData.label, resultData.confidence);
            }, 2000);
        } finally {
            btnAnalyze.innerHTML = originalBtnContent;
            btnAnalyze.disabled = false;
            lucide.createIcons();
        }
    }

    function showResults(data) {
        resultEmpty.classList.add('hidden');
        resultContent.classList.remove('hidden');

        resultLabel.textContent = data.label;
        resultConfidence.textContent = data.confidence;

        resultCard.className = "relative overflow-hidden rounded-[24px] border backdrop-blur-xl p-8 md:p-10 shadow-2xl transition-all duration-500";
        resultBar.className = "h-full rounded-full shadow-[0_0_20px_currentColor] transition-all duration-1000 ease-out relative w-0";

        let colorClass = "";
        let barClass = "";
        let textClass = "";

        if (data.label.includes("No Tumor")) {
            colorClass = "border-emerald-500/30 bg-emerald-950/30";
            barClass = "bg-gradient-to-r from-emerald-500 to-emerald-400";
            textClass = "text-emerald-400";
            resultIcon.setAttribute('data-lucide', 'check-circle-2');
        } else if (data.label.includes("Pituitary")) {
            colorClass = "border-amber-500/30 bg-amber-950/30";
            barClass = "bg-gradient-to-r from-amber-500 to-amber-400";
            textClass = "text-amber-400";
            resultIcon.setAttribute('data-lucide', 'alert-circle');
        } else {
            colorClass = "border-rose-500/30 bg-rose-950/30";
            barClass = "bg-gradient-to-r from-rose-500 to-rose-400";
            textClass = "text-rose-400";
            resultIcon.setAttribute('data-lucide', 'activity');
        }

        resultCard.classList.add(...colorClass.split(" "));
        resultBar.classList.add(...barClass.split(" "));

        resultLabel.className = `text-4xl md:text-5xl font-bold mb-2 tracking-tight ${textClass}`;
        resultConfidence.className = `text-3xl font-bold ${textClass}`;

        lucide.createIcons();

        setTimeout(() => {
            resultBar.style.width = data.confidence;
        }, 100);
    }

    function handleRetryWithKey() {
        const key = apiKeyInput.value.trim();
        if (key) {
            userApiKey = key;
            generateGeminiResponse(resultData.label, resultData.confidence);
        }
    }

    async function generateGeminiResponse(label, confidence) {
        const activeKey = userApiKey || CONFIG_API_KEY;

        geminiText.classList.add('hidden');
        geminiError.classList.add('hidden');
        exportContainer.classList.add('hidden');

        if (!activeKey) {
            showGeminiError("API Key Required", "Please enter your Gemini API Key to generate the report.", false);
            return;
        }

        geminiSkeleton.classList.remove('hidden');
        geminiLoader.classList.remove('hidden');
        addToLog("Starting AI analysis...");

        const genAI = new GoogleGenerativeAI(activeKey);

        const priorityModels = [
            "gemini-1.5-flash",
            "gemini-1.5-flash-001",
            "gemini-1.5-flash-002",
            "gemini-1.5-pro",
            "gemini-1.5-pro-001"
        ];

        const prompt = `Context: You are an advanced medical AI assistant named NeuroScan. Task: Explain the following MRI analysis result to a patient. Data: Detection: ${label}, AI Confidence: ${confidence}. Requirements: Reassuring greeting. Explain "${label}" simply. State clearly: AI SCREENING ONLY, NOT DIAGNOSIS. 3 distinct next steps. Markdown format. Under 150 words.`;

        let success = false;

        for (const modelName of priorityModels) {
            try {
                modelAttemptName.textContent = `Trying ${modelName}...`;
                addToLog(`Attempting ${modelName}...`);

                const model = genAI.getGenerativeModel({ model: modelName });
                const result = await model.generateContent(prompt);
                const response = await result.response;

                displayGeminiText(response.text());
                success = true;
                break;
            } catch (e) {
                addToLog(`Failed ${modelName}: ${e.message.slice(0, 40)}...`);
                if (e.message.includes("429")) {
                    showGeminiError("Rate Limit Exceeded", "Free tier quota exceeded. Please wait 30s.", true);
                    geminiSkeleton.classList.add('hidden');
                    geminiLoader.classList.add('hidden');
                    return;
                }
                await new Promise(r => setTimeout(r, 500));
            }
        }

        if (!success) {
            modelAttemptName.textContent = "Auto-Discovering...";
            addToLog("Standard models failed. Searching for stable alternative...");
            try {
                const listReq = await fetch(`https://generativelanguage.googleapis.com/v1beta/models?key=${activeKey}`);
                if (!listReq.ok) throw new Error("Discovery request failed");

                const listData = await listReq.json();

                const validModel = listData.models?.find(m => {
                    const name = m.name.toLowerCase();
                    const isGen = m.supportedGenerationMethods?.includes("generateContent");
                    const isStable = !name.includes("preview") && !name.includes("exp");
                    return isGen && isStable && (name.includes("flash") || name.includes("pro"));
                });

                if (validModel) {
                    const name = validModel.name.replace("models/", "");
                    addToLog(`Found stable model: ${name}`);
                    const model = genAI.getGenerativeModel({ model: name });
                    const result = await model.generateContent(prompt);
                    const response = await result.response;
                    displayGeminiText(response.text());
                } else {
                    throw new Error("No stable, accessible models found on this key.");
                }
            } catch (finalErr) {
                showGeminiError("Connection Failed", finalErr.message, true);
            }
        }

        geminiSkeleton.classList.add('hidden');
        geminiLoader.classList.add('hidden');
    }

    function displayGeminiText(text) {
        geminiText.innerHTML = text
            .replace(/\*\*(.*?)\*\*/g, '<strong class="font-bold text-cyan-200">$1</strong>')
            .replace(/\n/g, '<br/>');
        geminiText.classList.remove('hidden');
        exportContainer.classList.remove('hidden');
        addToLog("Analysis generated successfully.");
    }

    function showGeminiError(title, msg, showLog) {
        geminiError.classList.remove('hidden');
        errorTitle.textContent = title;
        errorMsg.textContent = msg.replace("API_ERROR: ", "");

        if (showLog) {
            diagnosticLogContainer.classList.remove('hidden');
        } else {
            diagnosticLogContainer.classList.add('hidden');
        }
    }
})();

