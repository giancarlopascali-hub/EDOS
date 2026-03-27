/**
 * EDOS Platform Logic
 * Handles DoE, Bayesian Optimization, and Statistical Analysis.
 */

// --- Global State ---
let currentModule = 'doe'; // 'doe', 'bo', 'sa'
let currentData = null;      // Full dataset loaded from CSV (Array of Arrays)
let currentColumns = [];     // Column names (Array of Strings)
let columnRoles = {};        // Mapping of column names to roles: 'feature' or 'objective'
let objectiveConfigs = {};   // Global store for objective types/targets
let suggestionsData = null;  // Latest results from the optimizer/generator

// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
    // Inject refined inclusion styles
    const style = document.createElement('style');
    style.textContent = `
        .row-disabled { opacity: 0.55; background-color: #f1f5f9; }
        .row-disabled input:not([type="checkbox"]), .row-disabled select { 
            background: #e2e8f0; color: #94a3b8; cursor: not-allowed; pointer-events: none; 
        }
        .row-disabled td { color: #94a3b8; }
    `;
    document.head.appendChild(style);

    initModuleSwitcher();
    initFileUpload();
    initTabs();
    initButtons();
    initSettingsListeners();
});

function initModuleSwitcher() {
    document.querySelectorAll('.module-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const module = btn.dataset.module;
            switchModule(module);
        });
    });
}

function switchModule(module) {
    currentModule = module;
    document.querySelectorAll('.module-btn').forEach(b => b.classList.toggle('active', b.dataset.module === module));
    document.querySelectorAll('.module-view').forEach(v => v.classList.toggle('hidden', v.id !== `${module}-module`));
    
    if (module === 'bo' || module === 'sa') {
        document.getElementById('upload-section').classList.remove('hidden');
        if (currentData) {
            document.getElementById('data-section').classList.remove('hidden');
            if (module === 'bo') {
                renderTrendPlot();
            }
        }
    } else {
        document.getElementById('upload-section').classList.add('hidden');
        document.getElementById('data-section').classList.add('hidden');
    }
    renderSetup();
}

function initTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const parent = btn.closest('section, .module-view');
            parent.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            parent.querySelectorAll('.tab-content').forEach(c => c.classList.add('hidden'));
            btn.classList.add('active');
            document.getElementById(btn.dataset.tab).classList.remove('hidden');
        });
    });
}

function initFileUpload() {
    const fileInput = document.getElementById('file-input');
    
    safeAddListener('file-input', 'change', (e) => {
        if (e.target.files.length > 0) {
            handleFiles(e.target.files);
            e.target.value = ''; // Reset to allow re-selection of same file
        }
    });

    const dropZone = document.getElementById('drop-zone');
    if (dropZone) {
        dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
            handleFiles(e.dataTransfer.files);
        });
    }
}

async function handleFiles(files) {
    if (files.length === 0) return;
    const formData = new FormData();
    formData.append('file', files[0]);

    try {
        const response = await fetch('/upload', { method: 'POST', body: formData });
        const result = await response.json();
        if (result.error) throw new Error(result.error);

        currentData = result.data;
        currentColumns = result.columns;

        columnRoles = {};
        objectiveConfigs = {};
        const objectives = currentColumns.filter((col, idx) => idx === currentColumns.length - 1 || columnRoles[col] === 'objective');
        
        currentColumns.forEach((col, idx) => {
            const isDefaultObj = (idx === currentColumns.length - 1);
            columnRoles[col] = isDefaultObj ? 'objective' : 'feature';
        });

        const objCols = currentColumns.filter(c => columnRoles[c] === 'objective');
        objCols.forEach(col => {
            objectiveConfigs[col] = {
                name: col,
                type: 'maximize',
                target: '',
                importance: (100 / objCols.length).toFixed(1),
                included: true
            };
        });

        renderMainTable();
        renderSetup();
        if (currentModule === 'bo') renderTrendPlot();
        
        document.getElementById('data-section').classList.remove('hidden');
    } catch (err) {
        alert('Error loading file: ' + err.message);
    }
}

function renderMainTable() {
    const head = document.getElementById('main-table-head');
    const body = document.getElementById('main-table-body');
    
    let headHtml = '<tr>';
    currentColumns.forEach(col => {
        const role = columnRoles[col];
        headHtml += `
            <th>
                <div class="col-header-name">${col}</div>
                <div class="type-toggle" onclick="toggleColumnRole('${col}')">
                    <span class="${role === 'feature' ? 'active' : ''}">F</span>
                    <span class="${role === 'objective' ? 'active' : ''}">O</span>
                </div>
            </th>`;
    });
    headHtml += '</tr>';
    head.innerHTML = headHtml;

    let bodyHtml = '';
    currentData.forEach((row, rowIdx) => {
        bodyHtml += '<tr>' + row.map((v, colIdx) => `<td contenteditable="true" onblur="updateCellData(${rowIdx}, ${colIdx}, this.textContent)">${v}</td>`).join('') + '</tr>';
    });
    body.innerHTML = bodyHtml;
}

window.updateCellData = (rowIdx, colIdx, value) => {
    currentData[rowIdx][colIdx] = value;
    if (currentModule === 'bo') renderTrendPlot();
};

window.toggleColumnRole = (colName) => {
    columnRoles[colName] = columnRoles[colName] === 'feature' ? 'objective' : 'feature';
    
    const activeObjs = currentColumns.filter(c => columnRoles[c] === 'objective');
    
    // Ensure all active objectives are in objectiveConfigs
    activeObjs.forEach(col => {
        if (!objectiveConfigs[col]) {
            objectiveConfigs[col] = { name: col, type: 'maximize', target: '', importance: 0, included: true };
        }
    });
    // Remove stale configs
    Object.keys(objectiveConfigs).forEach(col => {
        if (!activeObjs.includes(col)) delete objectiveConfigs[col];
    });

    rebalanceObjectiveConfigs();
    renderMainTable();
    renderSetup();
    if (currentModule === 'bo') renderTrendPlot();
};

function rebalanceRemaining(changedIdOrEl, newValue, selector) {
    const inputs = Array.from(document.querySelectorAll(selector));
    let activeInputs = inputs;
    
    // For BO/SA, only consider included objectives
    if (selector.includes('bo') || selector.includes('sa')) {
        activeInputs = inputs.filter(el => {
            const col = el.dataset.col;
            return objectiveConfigs[col] && objectiveConfigs[col].included;
        });
    }

    if (activeInputs.length <= 1) {
        if (activeInputs.length === 1) activeInputs[0].value = (100).toFixed(1);
        return;
    }

    const changedIdx = activeInputs.findIndex(el => {
        if (changedIdOrEl instanceof HTMLElement) return el === changedIdOrEl;
        return el.dataset.col === changedIdOrEl;
    });
    if (changedIdx === -1) return;

    const total = 1000;
    
    if (changedIdx < activeInputs.length - 1) {
        // 1. Waterfall Down: Rows ABOVE are fixed, Rows BELOW are targets
        let sumAbove = 0;
        for (let i = 0; i < changedIdx; i++) {
            sumAbove += Math.round(parseFloat(activeInputs[i].value) * 10) || 0;
        }

        // Cap user input to fit the remainder
        let changedVal = Math.round(parseFloat(newValue) * 10) || 0;
        if (sumAbove + changedVal > total) {
            changedVal = Math.max(0, total - sumAbove);
            activeInputs[changedIdx].value = (changedVal / 10).toFixed(1);
        }

        const remainingForBelow = total - sumAbove - changedVal;
        const belowRows = activeInputs.slice(changedIdx + 1);
        const share = Math.floor(remainingForBelow / belowRows.length);
        let currentTargetSum = 0;

        belowRows.forEach((el, idx) => {
            let val;
            if (idx === belowRows.length - 1) {
                val = remainingForBelow - currentTargetSum;
            } else {
                val = share;
                currentTargetSum += val;
            }
            if (val < 0) val = 0;
            el.value = (val / 10).toFixed(1);
        });
    } else {
        // 2. Edit Last Row: Rebalance EVERYTHING ABOVE
        let changedVal = Math.round(parseFloat(newValue) * 10) || 0;
        if (changedVal > total) {
            changedVal = total;
            activeInputs[changedIdx].value = (100).toFixed(1);
        }

        const remainderForAbove = total - changedVal;
        const aboveRows = activeInputs.slice(0, changedIdx);
        const share = Math.floor(remainderForAbove / aboveRows.length);
        let currentTargetSum = 0;

        aboveRows.forEach((el, idx) => {
            let val;
            if (idx === aboveRows.length - 1) {
                val = remainderForAbove - currentTargetSum;
            } else {
                val = share;
                currentTargetSum += val;
            }
            if (val < 0) val = 0;
            el.value = (val / 10).toFixed(1);
        });
    }
}

function rebalanceObjectiveConfigs() {
    const objCols = Object.keys(objectiveConfigs);
    const includedCols = objCols.filter(c => objectiveConfigs[c].included);
    
    // First, strictly zero out all non-included objectives
    objCols.forEach(col => {
        if (!objectiveConfigs[col].included) {
            objectiveConfigs[col].importance = (0).toFixed(1);
        }
    });

    if (includedCols.length === 0) return;
    
    // Distribute 100% among included ones
    const total = 1000;
    const share = Math.floor(total / includedCols.length);
    let currentSum = 0;

    includedCols.forEach((col, idx) => {
        if (idx === includedCols.length - 1) {
            objectiveConfigs[col].importance = ((total - currentSum) / 10).toFixed(1);
        } else {
            const val = share;
            objectiveConfigs[col].importance = (val / 10).toFixed(1);
            currentSum += val;
        }
    });
}

function renderSetup() {
    if (currentModule === 'doe') renderDoESetup();
    else if (currentModule === 'bo') renderBOSetup();
    else if (currentModule === 'sa') renderSASetup();
}

function renderDoESetup() {
    // Starts blank and persists manual additions in the DOM unless explicitly cleared.
}

function rebalanceDoEDOM() {
    const inputs = Array.from(document.querySelectorAll('.doe-o-importance'));
    if (inputs.length === 0) return;
    const share = (100 / inputs.length).toFixed(1);
    inputs.forEach(input => input.value = share);
}

window.updateObjectiveName = (oldName, newName) => {
    if (oldName === newName) return;
    if (objectiveConfigs[oldName]) {
        objectiveConfigs[newName] = objectiveConfigs[oldName];
        objectiveConfigs[newName].name = newName;
        delete objectiveConfigs[oldName];
        // Re-render other modules if they care
        renderSetup();
    }
};

function renderBOSetup() {
    const featuresList = document.getElementById('bo-features-list');
    const objectivesList = document.getElementById('bo-objectives-list');
    
    let featuresHtml = '';
    currentColumns.filter(col => columnRoles[col] === 'feature').forEach(col => {
        const colIdx = currentColumns.indexOf(col);
        const vals = currentData.map(r => r[colIdx]).filter(v => v != null);
        const isNum = vals.every(v => !isNaN(parseFloat(v)));
        const range = isNum ? `[${Math.min(...vals)}, ${Math.max(...vals)}]` : [...new Set(vals)].join(', ');

        featuresHtml += `<tr>
            <td>${col}</td>
            <td><select class="bo-f-type" data-col="${col}">
                <option value="continuous" ${isNum ? 'selected' : ''}>continuous</option>
                <option value="discrete">discrete</option>
                <option value="categorical" ${!isNum ? 'selected' : ''}>categorical</option>
            </select></td>
            <td><input type="text" class="bo-f-range" data-col="${col}" value="${range}"></td>
            <td><input type="checkbox" class="bo-f-include" data-col="${col}" checked></td>
        </tr>`;
    });
    featuresList.innerHTML = featuresHtml;

    let objectivesHtml = '';
    const selectedObjs = currentColumns.filter(col => columnRoles[col] === 'objective');
    selectedObjs.forEach(col => {
        const cfg = objectiveConfigs[col] || { type: 'maximize', target: '', importance: (100/selectedObjs.length).toFixed(1), included: true };
        const isDisabled = !cfg.included;
        objectivesHtml += `<tr class="${isDisabled ? 'row-disabled' : ''}">
            <td>${col}</td>
            <td><select class="bo-o-type" data-col="${col}" onchange="syncObjectiveConfig('${col}', 'bo'); toggleRowTarget(this);" ${isDisabled ? 'disabled' : ''}>
                <option value="maximize" ${cfg.type === 'maximize' ? 'selected' : ''}>maximize</option>
                <option value="minimize" ${cfg.type === 'minimize' ? 'selected' : ''}>minimize</option>
                <option value="target" ${cfg.type === 'target' ? 'selected' : ''}>target</option>
            </select></td>
            <td><input type="text" class="bo-o-target ${cfg.type === 'target' ? '' : 'hidden'}" data-col="${col}" value="${cfg.target}" placeholder="Target value" onchange="syncObjectiveConfig('${col}', 'bo', 'target')" ${isDisabled ? 'disabled' : ''}></td>
            <td><input type="number" class="bo-o-importance" data-col="${col}" value="${cfg.importance}" onchange="syncObjectiveConfig('${col}', 'bo', 'importance')" ${isDisabled ? 'disabled' : ''}></td>
            <td><input type="checkbox" class="bo-o-include" data-col="${col}" ${cfg.included ? 'checked' : ''} onchange="syncObjectiveConfig('${col}', 'bo', 'include')"></td>
        </tr>`;
    });
    objectivesList.innerHTML = objectivesHtml;
}

window.syncObjectiveConfig = (col, sourceModule, field) => {
    if (sourceModule === 'doe') return;
    
    const rolePrefix = sourceModule === 'bo' ? 'bo-o' : 'sa-o';
    const row = document.querySelector(`.${rolePrefix}-type[data-col="${col}"]`).closest('tr');
    
    // 1. Sync DOM -> Config FIRST (so rebalance sees correct 'included' flags)
    document.querySelectorAll(`.${rolePrefix}-include`).forEach(cb => {
        const c = cb.dataset.col;
        const r = cb.closest('tr');
        if (objectiveConfigs[c]) {
            objectiveConfigs[c].type = r.querySelector(`.${rolePrefix}-type`).value;
            objectiveConfigs[c].target = r.querySelector(`.${rolePrefix}-target`).value;
            // Only sync importance if we aren't about to auto-rebalance it from scratch
            if (field === 'importance') {
                objectiveConfigs[c].importance = r.querySelector(`.${rolePrefix}-importance`).value;
            }
            objectiveConfigs[c].included = cb.checked;
        }
    });

    // 2. Perform Rebalance logic based on the updated Config
    if (field === 'importance') {
        const selector = `.${rolePrefix}-importance`;
        const val = row.querySelector(selector).value;
        rebalanceRemaining(col, val, selector);
        
        // After DOM-based rebalanceRemaining, sync the derived values back to config
        document.querySelectorAll(selector).forEach(el => {
            if (objectiveConfigs[el.dataset.col]) objectiveConfigs[el.dataset.col].importance = el.value;
        });
    } else if (field === 'include' || field === 'type') {
        rebalanceObjectiveConfigs();
    }

    renderSetup();
    if (currentModule === 'bo') renderTrendPlot();
};

function renderSASetup() {
    const list = document.getElementById('sa-features-list');
    const objList = document.getElementById('sa-objectives-list');
    
    let featHtml = '';
    currentColumns.filter(c => columnRoles[c] === 'feature').forEach(col => {
        // Heuristic: Categorical if it's explicitly non-numeric. 
        // If it's numeric, we always treat as continuous initially, even if it has few values.
        let type = 'continuous';
        const colIdx = currentColumns.indexOf(col);
        const sample = currentData ? currentData.slice(0, 200).map(r => r[colIdx]) : [];
        const isNumeric = sample.every(v => v === null || v === '' || !isNaN(Number(v)));
        if (!isNumeric) {
            type = 'categorical';
        }

        featHtml += `<tr>
            <td>${col}</td>
            <td><select class="sa-f-type" data-col="${col}">
                <option value="continuous" ${type === 'continuous' ? 'selected' : ''}>Numerical</option>
                <option value="discrete">Discrete</option>
                <option value="categorical" ${type === 'categorical' ? 'selected' : ''}>Categorical</option>
            </select></td>
            <td><input type="checkbox" class="sa-f-include" data-col="${col}" checked> Include</td>
        </tr>`;
    });
    list.innerHTML = featHtml;

    let objHtml = '';
    const selectedObjs = currentColumns.filter(col => columnRoles[col] === 'objective');
    selectedObjs.forEach(col => {
        const cfg = objectiveConfigs[col] || { type: 'maximize', target: '', importance: (100/selectedObjs.length).toFixed(1), included: true };
        const isDisabled = !cfg.included;
        objHtml += `<tr class="${isDisabled ? 'row-disabled' : ''}">
            <td>${col}</td>
            <td><select class="sa-o-type" data-col="${col}" onchange="syncObjectiveConfig('${col}', 'sa', 'type'); toggleRowTarget(this)" ${isDisabled ? 'disabled' : ''}>
                <option value="maximize" ${cfg.type === 'maximize' ? 'selected' : ''}>maximize</option>
                <option value="minimize" ${cfg.type === 'minimize' ? 'selected' : ''}>minimize</option>
                <option value="target" ${cfg.type === 'target' ? 'selected' : ''}>target</option>
            </select></td>
            <td><input type="text" class="sa-o-target ${cfg.type === 'target' ? '' : 'hidden'}" data-col="${col}" value="${cfg.target || ''}" placeholder="Target" onchange="syncObjectiveConfig('${col}', 'sa', 'target')" ${isDisabled ? 'disabled' : ''}></td>
            <td><input type="number" class="sa-o-importance" data-col="${col}" value="${cfg.importance}" onchange="syncObjectiveConfig('${col}', 'sa', 'importance')" ${isDisabled ? 'disabled' : ''}></td>
            <td><input type="checkbox" class="sa-o-include" data-col="${col}" ${cfg.included ? 'checked' : ''} onchange="syncObjectiveConfig('${col}', 'sa', 'include')"> Include</td>
        </tr>`;
    });
    objList.innerHTML = objHtml;
}

function safeAddListener(id, event, callback) {
    const el = document.getElementById(id);
    if (el) el.addEventListener(event, callback);
    else console.warn(`Element with ID "${id}" not found. skipping listener.`);
}

function initButtons() {
    // DoE manual add
    safeAddListener('doe-add-feature-btn', 'click', () => {
        const list = document.getElementById('doe-features-list');
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><input type="text" class="doe-f-name" value="Feature_${list.children.length + 1}"></td>
            <td><select class="doe-f-type" onchange="updatePlaceholder(this)">
                <option value="continuous">continuous</option>
                <option value="discrete">discrete</option>
                <option value="categorical">categorical</option>
            </select></td>
            <td><input type="text" class="doe-f-range" placeholder="[min, max]"></td>
            <td><button class="btn secondary small delete-row">×</button></td>`;
        list.appendChild(tr);
        attachDeleteEvents();
    });

    safeAddListener('doe-add-objective-btn', 'click', () => {
        const list = document.getElementById('doe-objectives-list');
        const count = list.children.length + 1;
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><input type="text" class="doe-o-name" value="Objective_${count}"></td>
            <td><select class="doe-o-type" onchange="toggleRowTarget(this)">
                <option value="maximize">maximize</option>
                <option value="minimize">minimize</option>
                <option value="target">target</option>
            </select></td>
            <td><input type="text" class="doe-o-target hidden" placeholder="Target"></td>
            <td><input type="number" class="doe-o-importance" value="0"></td>
            <td><button class="btn secondary small delete-row">×</button></td>`;
        list.appendChild(tr);
        attachDeleteEvents();
        rebalanceDoEDOM();
    });

    safeAddListener('doe-run-btn', 'click', runDoE);
    safeAddListener('run-btn', 'click', runBO);
    safeAddListener('sa-run-btn', 'click', runSA);
    safeAddListener('sa-estimate-btn', 'click', runEstimate);
    safeAddListener('shutdown-btn', 'click', async () => {
        if (confirm('Are you sure you want to shut down the EDOS server?')) {
            try {
                await fetch('/shutdown', { method: 'POST' });
            } catch (_) { /* expected — server is stopping */ }
            document.body.innerHTML = `
                <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100vh;background:#0f172a;color:#e2e8f0;font-family:sans-serif;gap:20px;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M18.4 6.1a9 9 0 1 1-12.8 0"/><line x1="12" y1="2" x2="12" y2="12"/></svg>
                    <h1 style="font-size:2rem;margin:0;">Server Stopped</h1>
                    <p style="color:#64748b;max-width:400px;text-align:center;">The EDOS server has been shut down. You can safely close this browser tab.</p>
                </div>`;
        }
    });

    safeAddListener('export-main-btn', 'click', () => exportData(currentData, currentColumns));
    
    safeAddListener('commit-suggestions-btn', 'click', (e) => {
        if (e.target.disabled) return;
        const rows = document.querySelectorAll('#results-table-body tr');
        const featCols = currentColumns.filter(c => columnRoles[c] === 'feature');
        const objCols = currentColumns.filter(c => columnRoles[c] === 'objective');
        
        let addedCount = 0;
        rows.forEach(tr => {
            const cells = tr.querySelectorAll('td');
            const newRow = new Array(currentColumns.length).fill('');
            
            featCols.forEach((col, i) => {
                newRow[currentColumns.indexOf(col)] = cells[i].textContent.trim();
            });
            
            objCols.forEach((col, i) => {
                const cellIdx = featCols.length + i;
                const val = cells[cellIdx].textContent.trim();
                newRow[currentColumns.indexOf(col)] = (val === 'N/D') ? '' : val;
            });

            currentData.push(newRow);
            addedCount++;
        });

        if (addedCount > 0) {
            renderMainTable();
            renderTrendPlot();
            renderParetoPlot();
            alert(`Added ${addedCount} experiments to the main dataset. Plots refreshed.`);
            document.getElementById('results-section').classList.add('hidden');
        }
    });
    
    safeAddListener('doe-export-btn', 'click', () => {
        if (!suggestionsData || suggestionsData.length === 0) return;
        const rows = document.querySelectorAll('#doe-results-body tr');
        if (rows.length === 0) return;
        const feats = Object.keys(suggestionsData[0]);
        const cols = [...feats, ...currentDoEObjectives.map(o => o.name)];
        
        const exportArr = Array.from(rows).map(tr => {
            const cells = tr.querySelectorAll('td');
            return Array.from(cells).map(td => td.textContent.trim());
        });
        exportData(exportArr, cols);
    });
    
    safeAddListener('export-results-btn', 'click', () => {
        if (!suggestionsData || suggestionsData.length === 0) return;
        const rows = document.querySelectorAll('#results-table-body tr');
        if (rows.length === 0) return;
        const featCols = currentColumns.filter(c => columnRoles[c] === 'feature');
        const objCols = currentColumns.filter(c => columnRoles[c] === 'objective');
        const cols = [...featCols, ...objCols.map(c => `${c} (Result)`)];
        
        const exportArr = Array.from(rows).map(tr => {
            const cells = tr.querySelectorAll('td');
            return Array.from(cells).map(td => td.textContent.trim());
        });
        exportData(exportArr, cols);
    });

    safeAddListener('pareto-export-btn', 'click', () => {
        const objCols = currentColumns.filter(c => columnRoles[c] === 'objective');
        const paretoIndices = detectParetoIndices(currentData, objCols);
        const paretoData = currentData.filter((_, i) => paretoIndices.includes(i));
        exportData(paretoData, currentColumns);
    });
}

function initSettingsListeners() {
    // SA Tweaks visibility handling
    const saModel = document.getElementById('sa-model-select');
    const updateSAParamsVisibility = () => {
        const val = saModel.value;
        document.querySelectorAll('.sa-rf-param').forEach(el => el.classList.toggle('hidden', val !== 'random_forest'));
        document.querySelectorAll('.sa-mlp-param').forEach(el => el.classList.toggle('hidden', val !== 'mlp'));
    };
    saModel.addEventListener('change', updateSAParamsVisibility);
    updateSAParamsVisibility(); // Init on load

    safeAddListener('sa-export-report-btn', 'click', exportSAReport);

    // BO Slider Sync
    const styleSlider = document.getElementById('style-slider');
    const styleNumber = document.getElementById('style-number');
    if (styleSlider && styleNumber) {
        styleSlider.addEventListener('input', (e) => styleNumber.value = e.target.value);
        styleNumber.addEventListener('input', (e) => styleSlider.value = e.target.value);
    }

    // Objective Importance Normalization Listeners
    const attachNorm = (selector) => {
        document.addEventListener('change', (e) => {
            if (e.target.matches(selector)) {
                let inputs = Array.from(document.querySelectorAll(selector));
                
                // For BO/SA, only normalize among INCLUDED objectives
                const isDoe = selector.includes('doe');
                if (!isDoe) {
                    inputs = inputs.filter(el => {
                        const col = el.dataset.col;
                        return objectiveConfigs[col] && objectiveConfigs[col].included;
                    });
                }

                const val = e.target.value;
                rebalanceRemaining(e.target, val, selector);

                // AFTER local DOM update, sync EVERYTHING to global state for BO/SA
                if (!isDoe) {
                    Array.from(document.querySelectorAll(selector)).forEach(el => {
                        const col = el.dataset.col;
                        if (col && objectiveConfigs[col]) {
                            objectiveConfigs[col].importance = el.value;
                        }
                    });
                }
                
                if (!isDoe && currentModule === 'bo') renderTrendPlot();
            }
        });
    };
    attachNorm('.doe-o-importance');
    attachNorm('.bo-o-importance');
    attachNorm('.sa-o-importance');
}

function attachDeleteEvents() {
    document.querySelectorAll('.delete-row').forEach(btn => {
        btn.onclick = () => {
            const tr = btn.closest('tr');
            const nameInput = tr.querySelector('.doe-o-name, .doe-f-name');
            // Check if this is a DoE row
            const isDoeRow = tr.querySelector('.doe-o-name, .doe-f-name, .doe-o-type, .doe-f-type');
            
            if (nameInput && !isDoeRow) {
                const name = nameInput.value;
                if (objectiveConfigs[name]) {
                    delete objectiveConfigs[name];
                    rebalanceObjectiveConfigs();
                }
            }
            tr.remove();
            if (currentModule === 'doe') {
                rebalanceDoEDOM();
            } else {
                renderSetup();
            }
        };
    });
}

window.toggleRowTarget = (select) => {
    const tr = select.closest('tr');
    const targetInput = tr.querySelector('.bo-o-target, .doe-o-target, .sa-o-target');
    if (select.value === 'target') {
        targetInput.classList.remove('hidden');
    } else {
        targetInput.classList.add('hidden');
    }
};

window.updatePlaceholder = (select) => {
    const tr = select.closest('tr');
    const input = tr.querySelector('.doe-f-range');
    const type = select.value;
    if (type === 'continuous') input.placeholder = '[min, max]';
    else if (type === 'discrete') input.placeholder = 'xx, yy, zz,...';
    else if (type === 'categorical') input.placeholder = 'A, B, C,...';
};

async function runDoE() {
    const features = Array.from(document.querySelectorAll('#doe-features-list tr')).map(tr => ({
        name: tr.querySelector('.doe-f-name').value,
        type: tr.querySelector('.doe-f-type').value,
        range: tr.querySelector('.doe-f-range').value
    }));

    // Collect objectives for column rendering
    currentDoEObjectives = Array.from(document.querySelectorAll('#doe-objectives-list tr')).map(tr => ({
        name: tr.querySelector('.doe-o-name').value
    })).filter(o => o.name.trim() !== "");

    const tweaks = {
        model: document.getElementById('doe-model-select').value,
        max_runs: document.getElementById('doe-max-runs').value
    };

    const res = await fetch('/doe', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ features, tweaks })
    });
    const result = await res.json();
    if (result.error) return alert(result.error);
    
    suggestionsData = result.suggestions;
    renderDoEResults(result.metrics);
}

function renderDoEResults(metrics) {
    const section = document.getElementById('doe-results-section');
    const dashboard = document.getElementById('doe-quality-dashboard');
    const head = document.getElementById('doe-results-head');
    const body = document.getElementById('doe-results-body');

    // Render Quality Dashboard
    if (metrics && typeof metrics === 'object') {
        const getFlag = (val, type) => {
            if (type === 'ortho') {
                if (val >= 90) return { text: 'Good', class: 'flag-good' };
                if (val >= 70) return { text: 'Limited', class: 'flag-limited' };
                return { text: 'Poor', class: 'flag-poor' };
            }
            if (type === 'eff') {
                if (val >= 80) return { text: 'Good', class: 'flag-good' };
                if (val >= 50) return { text: 'Limited', class: 'flag-limited' };
                return { text: 'Poor', class: 'flag-poor' };
            }
            if (type === 'res') {
                if (val.includes('V')) return { text: 'Good', class: 'flag-good' };
                if (val.includes('IV')) return { text: 'Limited', class: 'flag-limited' };
                return { text: 'Screening', class: 'flag-poor' };
            }
            if (type === 'curve') {
                if (val === 'Excellent') return { text: 'Good', class: 'flag-good' };
                if (val === 'Partial') return { text: 'Limited', class: 'flag-limited' };
                return { text: 'None', class: 'flag-poor' };
            }
            return { text: 'N/A', class: 'flag-limited' };
        };

        const orthoFlag = getFlag(metrics.orthogonality, 'ortho');
        const effFlag = getFlag(metrics.efficiency, 'eff');
        const resFlag = getFlag(metrics.resolution, 'res');
        const curveFlag = getFlag(metrics.curvature, 'curve');

        dashboard.innerHTML = `
            <div class="quality-card">
                <span class="metric-label">Orthogonality</span>
                <span class="metric-value">${metrics.orthogonality}%</span>
                <span class="quality-flag ${orthoFlag.class}">${orthoFlag.text}</span>
            </div>
            <div class="quality-card">
                <span class="metric-label">D-Efficiency</span>
                <span class="metric-value">${metrics.efficiency}%</span>
                <span class="quality-flag ${effFlag.class}">${effFlag.text}</span>
            </div>
            <div class="quality-card">
                <span class="metric-label">Resolution</span>
                <span class="metric-value">${metrics.resolution}</span>
                <span class="quality-flag ${resFlag.class}">${resFlag.text}</span>
            </div>
            <div class="quality-card">
                <span class="metric-label">Curvature</span>
                <span class="metric-value">${metrics.curvature}</span>
                <span class="quality-flag ${curveFlag.class}">${curveFlag.text}</span>
            </div>
        `;
        dashboard.classList.remove('hidden');
    }

    const feats = Object.keys(suggestionsData[0]);
    
    // Dynamic columns for objectives
    const objHeaders = currentDoEObjectives.length > 0 
        ? currentDoEObjectives.map(o => `<th>${o.name}</th>`).join('')
        : '<th>Target(s)</th>';

    head.innerHTML = '<tr>' + feats.map(f => `<th>${f}</th>`).join('') + objHeaders + '</tr>';
    
    body.innerHTML = suggestionsData.map(row => {
        const featCells = feats.map(f => `<td contenteditable="true" class="feat-input-cell editable-cell" style="background:#f8fafc; border: 1px dashed #94a3b8;">${row[f]}</td>`).join('');
        const objCells = currentDoEObjectives.length > 0
            ? currentDoEObjectives.map(() => `<td contenteditable="true" class="obj-input-cell editable-cell" style="background:#f0fdf4; border: 1px dashed #22c55e;"></td>`).join('')
            : '<td contenteditable="true" class="obj-input-cell editable-cell" style="background:#f0fdf4; border: 1px dashed #22c55e;"></td>';
            
        return '<tr>' + featCells + objCells + '</tr>';
    }).join('');
    
    section.classList.remove('hidden');
    section.scrollIntoView({ behavior: 'smooth' });
}

async function runBO() {
    const features = Array.from(document.querySelectorAll('#bo-features-list tr'))
        .filter(tr => tr.querySelector('.bo-f-include').checked)
        .map(tr => ({
            name: tr.querySelector('.bo-f-type').dataset.col,
            type: tr.querySelector('.bo-f-type').value,
            range: tr.querySelector('.bo-f-range').value
        }));
    
    const objectives = Array.from(document.querySelectorAll('#bo-objectives-list tr'))
        .filter(tr => tr.querySelector('.bo-o-include').checked)
        .map(tr => ({
            name: tr.querySelector('.bo-o-type').dataset.col,
            type: tr.querySelector('.bo-o-type').value,
            target: tr.querySelector('.bo-o-target').value,
            importance: tr.querySelector('.bo-o-importance').value
        }));

    const tweaks = {
        batch_size: document.getElementById('batch-size').value,
        acq_type: document.getElementById('acq-type-select').value,
        kernel: document.getElementById('kernel-select').value,
        exploration: document.getElementById('style-slider').value,
        noiseless: document.getElementById('bo-noiseless').checked,
        avoid_reval: document.getElementById('bo-avoid-reval').checked
    };

    document.getElementById('loading-text').textContent = "Optimizing...";
    document.getElementById('loading-overlay').classList.remove('hidden');

    try {
        const res = await fetch('/optimize', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ data: currentData, columns: currentColumns, features, objectives, tweaks })
        });
        const result = await res.json();
        
        document.getElementById('loading-overlay').classList.add('hidden');
        
        if (result.error) return alert("Optimization Error: " + result.error);
        
        suggestionsData = result.suggestions;
    
    // Merge configurations for plotting (don't clear)
    if (!objectiveConfigs) objectiveConfigs = {};
    objectives.forEach(obj => {
        objectiveConfigs[obj.name] = { ...obj, included: true };
    });

    const section = document.getElementById('results-section');
    const head = document.getElementById('results-table-head');
    const body = document.getElementById('results-table-body');
    const commitBtn = document.getElementById('commit-suggestions-btn');
    
    const featCols = currentColumns.filter(c => columnRoles[c] === 'feature');
    const objCols = currentColumns.filter(c => columnRoles[c] === 'objective');
    
    head.innerHTML = '<tr>' + featCols.map(c => `<th>${c}</th>`).join('') + objCols.map(c => `<th>${c} (Result)</th>`).join('') + '</tr>';
    
    const checkCommitReadiness = () => {
        const objCells = body.querySelectorAll('.obj-input-cell');
        const featCells = body.querySelectorAll('.feat-input-cell');
        let allFilled = true;
        
        objCells.forEach(td => {
            const val = td.textContent.trim();
            if (val === '' || isNaN(parseFloat(val))) allFilled = false;
        });
        
        featCells.forEach(td => {
            const val = td.textContent.trim();
            if (val === '') allFilled = false;
        });

        commitBtn.disabled = !allFilled;
        commitBtn.style.opacity = allFilled ? '1' : '0.5';
        commitBtn.style.cursor = allFilled ? 'pointer' : 'not-allowed';
    };

    const allObjCols = currentColumns.filter(c => columnRoles[c] === 'objective');
    const optimizedCols = objectives.map(o => o.name);

    body.innerHTML = suggestionsData.map(row => {
        const fCells = featCols.map(c => `<td contenteditable="true" class="feat-input-cell editable-cell" style="background:#f8fafc; border: 1px dashed #94a3b8;">${row[c]}</td>`).join('');
        const oCells = allObjCols.map(c => {
            const isOptimized = optimizedCols.includes(c);
            const content = isOptimized ? '' : 'N/D';
            const style = isOptimized 
                ? 'background:#f0fdf4; border: 1px dashed #22c55e;' 
                : 'background:#f1f5f9; border: 1px dashed #94a3b8; color: #64748b; font-style: italic;';
            return `<td contenteditable="true" class="obj-input-cell editable-cell" style="${style}">${content}</td>`;
        }).join('');
        return '<tr>' + fCells + oCells + '</tr>';
    }).join('');
    
    // Add input listeners for real-time validation
    body.querySelectorAll('.obj-input-cell, .feat-input-cell').forEach(cell => {
        cell.addEventListener('input', checkCommitReadiness);
    });
    
    // Initial check
    checkCommitReadiness();

    section.classList.remove('hidden');
    renderTrendPlot();
    section.scrollIntoView({ behavior: 'smooth' });
    } catch (err) {
        document.getElementById('loading-overlay').classList.add('hidden');
        alert("A critical error occurred during optimization: " + err.message);
        console.error(err);
    }
}

async function runSA() {
    const features = Array.from(document.querySelectorAll('.sa-f-include:checked')).map(cb => {
        const row = cb.closest('tr');
        return {
            name: cb.dataset.col,
            type: row.querySelector('.sa-f-type').value,
            range: 'valid'
        };
    });
    // Gather full objective config from the SA objectives table, filtering for "Include"
    const objectives = Array.from(document.querySelectorAll('#sa-objectives-list tr'))
        .filter(tr => tr.querySelector('.sa-o-include').checked)
        .map(tr => ({
            name: tr.querySelector('.sa-o-type').dataset.col,
            type: tr.querySelector('.sa-o-type').value,
            target: tr.querySelector('.sa-o-target').value,
            importance: tr.querySelector('.sa-o-importance').value
        }));
    
    // Store SA objectives for use in rendering plots
    window.saSelectedObjectives = objectives.map(o => o.name);
    
    const tweaks = {
        model: document.getElementById('sa-model-select').value,
        params: {
            n_estimators: document.getElementById('sa-rf-trees').value,
            max_depth: document.getElementById('sa-rf-depth').value,
            mlp_layers: document.getElementById('sa-mlp-layers').value,
            mlp_iter: document.getElementById('sa-mlp-iter').value
        }
    };

    // Store SA features for use in estimator rendering
    window.saSelectedFeatures = Array.from(document.querySelectorAll('.sa-f-include:checked')).map(cb => cb.dataset.col);

    document.getElementById('loading-text').textContent = "Analyzing...";
    document.getElementById('loading-overlay').classList.remove('hidden');
    
    // Clear old estimator results to avoid confusion with new analysis
    document.getElementById('sa-estimate-results').innerHTML = '';
    try {
        const res = await fetch('/sa', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ data: currentData, columns: currentColumns, features, objectives, tweaks })
        });
        const result = await res.json();
        document.getElementById('loading-overlay').classList.add('hidden');
        if (result.error) return alert("SA Error: " + result.error);
        window.lastSAResult = result;
        renderSAResults(result);
    } catch(err) {
        document.getElementById('loading-overlay').classList.add('hidden');
        alert("A critical error occurred during SA: " + err.message);
        console.error(err);
    }
}

function renderSAResults(result) {
    document.getElementById('sa-results-container').classList.remove('hidden');
    // --- Render Numerical Correlation Matrix ---
    const numCorr = result.numerical_correlation;
    const numFeats = result.num_features;
    const objNames = result.extended_obj_names;

    if (numFeats.length > 0) {
        const zNum = objNames.map(obj => numFeats.map(f => (numCorr[f] || {})[obj] ?? 0));
        Plotly.newPlot('sa-num-corr',
            [{ z: zNum, x: numFeats, y: objNames, type: 'heatmap', colorscale: 'RdBu', zmin: -1, zmax: 1,
               text: zNum.map(row => row.map(v => v.toFixed(2))),
               hovertemplate: 'Feature: %{x}<br>Objective: %{y}<br>Correlation: %{z:.3f}<extra></extra>',
               texttemplate: '%{text}', showscale: true }],
            { 
                xaxis: { title: 'Numerical Features', automargin: true },
                yaxis: { title: 'Success Targets', automargin: true },
                margin: { l: 150, b: 80, t: 30, r: 50 },
                height: 400
            });
    } else {
        document.getElementById('sa-num-corr').innerHTML = '<p class="text-secondary"><i>No numerical features available for correlation.</i></p>';
    }

    // --- Render Categorical Interactions ---
    const catContainer = document.getElementById('sa-cat-interactions-container');
    const catControls = document.getElementById('sa-cat-controls');
    catContainer.innerHTML = '';
    catControls.innerHTML = '';

    const interactions = result.cat_interactions;
    const pairs = Object.keys(interactions);

    if (pairs.length > 0) {
        // Create interaction selection controls
        let controlHtml = `<div class="tweak-item">
            <label>Select Pair:</label>
            <select id="sa-cat-pair-select" style="width: 100%;">
                ${pairs.map(p => `<option value="${p}">${p}</option>`).join('')}
            </select>
        </div>
        <div class="tweak-item">
            <label>Select Objective:</label>
            <select id="sa-cat-obj-select" style="width: 100%;">
                ${objNames.map(o => `<option value="${o}">${o}</option>`).join('')}
            </select>
        </div>`;
        catControls.innerHTML = controlHtml;

        const updateInteractionPlot = () => {
            const pair = document.getElementById('sa-cat-pair-select').value;
            const obj = document.getElementById('sa-cat-obj-select').value;
            const data = (interactions[pair] || {})[obj];
            if (!data) {
                catContainer.innerHTML = '<p class="text-secondary"><i>No data found for this combination.</i></p>';
                return;
            }
            
            const trace = {
                z: data.z,
                x: data.x,
                y: data.y,
                type: 'heatmap',
                colorscale: 'RdBu',
                zmin: 0,
                zmax: 1,
                hovertemplate: `<b>Best Achievement</b><br>${pair.split(' vs ')[1]}: %{x}<br>${pair.split(' vs ')[0]}: %{y}<br>Success Score: %{z:.3f}<extra></extra>`
            };
            
            Plotly.newPlot(catContainer, [trace], {
                title: `Optimal Outcome: ${obj} (Max Success)`,
                xaxis: { title: pair.split(' vs ')[1], automargin: true },
                yaxis: { title: pair.split(' vs ')[0], automargin: true },
                margin: { l: 150, b: 80, t: 50, r: 50 },
                height: 500
            });
        };

        document.getElementById('sa-cat-pair-select').addEventListener('change', updateInteractionPlot);
        document.getElementById('sa-cat-obj-select').addEventListener('change', updateInteractionPlot);
        updateInteractionPlot();
    } else {
        catContainer.innerHTML = '<p class="text-secondary"><i>No categorical interactions possible (requires 2+ categorical features).</i></p>';
    }

    // --- Render Parallel Coordinates Plots (one per objective) ---
    const parcoordsContainer = document.getElementById('sa-parcoords-container');
    parcoordsContainer.innerHTML = '';

    // Build the raw dataset: features + objectives for every experiment
    const featCols = (window.saSelectedFeatures || currentColumns.filter(c => columnRoles[c] === 'feature'));
    const objCols = (window.saSelectedObjectives || currentColumns.filter(c => columnRoles[c] === 'objective'));

    if (featCols.length > 0 && objCols.length > 0 && currentData.length > 0) {

        objCols.forEach((objCol, idx) => {
            // Each plot: all features + only its own objective axis
            const plotCols = [...featCols, objCol];

            // Build the dimensions array for Plotly parcoords
            const dimensions = plotCols.map(col => {
                const colIdx = currentColumns.indexOf(col);
                const rawVals = currentData.map(r => r[colIdx]);

                // Detect if categorical (any non-numeric value)
                const numVals = rawVals.map(v => parseFloat(v));
                const isCat = numVals.some(isNaN);

                if (isCat) {
                    const uniqueVals = [...new Set(rawVals.filter(v => v !== '' && v !== null))];
                    const mappedVals = rawVals.map(v => uniqueVals.indexOf(v) >= 0 ? uniqueVals.indexOf(v) : 0);
                    return {
                        label: col,
                        values: mappedVals,
                        tickvals: uniqueVals.map((_, i) => i),
                        ticktext: uniqueVals
                    };
                } else {
                    return {
                        label: col,
                        values: numVals.map(v => isNaN(v) ? 0 : v)
                    };
                }
            });

            // Color by the objective column for this plot
            const objIdx = currentColumns.indexOf(objCol);
            const colorVals = currentData.map(r => parseFloat(r[objIdx]) || 0);

            const pcDiv = document.createElement('div');
            pcDiv.id = `sa-parcoords-${idx}`;
            pcDiv.classList.add('plot-container');
            pcDiv.style.width = '100%';
            pcDiv.style.height = '420px';
            pcDiv.style.marginBottom = '2rem';
            parcoordsContainer.appendChild(pcDiv);

            const finiteColorVals = colorVals.filter(v => isFinite(v));
            const cmin = finiteColorVals.length > 0 ? Math.min(...finiteColorVals) : 0;
            const cmax = finiteColorVals.length > 0 ? Math.max(...finiteColorVals) : 1;

            const pcPlot = Plotly.newPlot(`sa-parcoords-${idx}`, [{
                type: 'parcoords',
                line: {
                    color: colorVals,
                    colorscale: [
                        [0, '#0d0887'], [0.11, '#46039f'], [0.22, '#7201a8'], [0.33, '#9c179e'], 
                        [0.44, '#bd3786'], [0.55, '#d8576b'], [0.66, '#ed7953'], [0.77, '#fb9f3a'], 
                        [0.88, '#fdca26'], [1, '#f0f921']
                    ],
                    showscale: true,
                    reversescale: false,
                    opacity: 0.85,
                    cmin: cmin,
                    cmax: cmax,
                    colorbar: { title: objCol, thickness: 15 }
                },
                unselected: {
                    line: {
                        opacity: 0,
                        color: 'rgba(0,0,0,0)'
                    }
                },
                dimensions: dimensions
            }], {
                title: `Parallel Coordinates — Colored by: ${objCol}`,
                margin: { l: 80, r: 80, t: 120, b: 20 }
            }, { responsive: true });

        });
    } else {
        parcoordsContainer.innerHTML = '<p class="text-secondary"><i>Parallel coordinates require at least one feature and one objective with experiments loaded.</i></p>';
    }

    // --- Render Individual Feature Impact Plots ---
    const impactContainer = document.getElementById('sa-impact-container');
    impactContainer.innerHTML = '';

    objNames.forEach((obj, idx) => {
        const res = result.results[obj];
        if (!res || !res.importance) return;
        
        const imp = res.importance;
        const sortedEntries = Object.entries(imp).sort((a, b) => a[1] - b[1]);
        const yFeats = sortedEntries.map(e => e[0]);
        const xVals = sortedEntries.map(e => e[1]);

        // Create a flex container for two side-by-side plots
        const container = document.createElement('div');
        container.classList.add('impact-row');
        container.style.display = 'flex';
        container.style.flexWrap = 'wrap';
        container.style.gap = '15px';
        container.style.marginBottom = '2.5rem';
        container.style.padding = '1rem';
        container.style.background = '#f8fafc';
        container.style.borderRadius = '12px';
        container.style.border = '1px solid #e2e8f0';
        
        const titleDiv = document.createElement('h3');
        titleDiv.textContent = `Feature Impact: ${obj}`;
        titleDiv.style.width = '100.0%';
        titleDiv.style.margin = '0 0 1rem 0';
        container.appendChild(titleDiv);

        const barId = `sa-bar-${idx}`;
        const barDiv = document.createElement('div');
        barDiv.id = barId;
        barDiv.style.flex = '1';
        barDiv.style.minWidth = '350px';
        barDiv.style.height = '400px';
        container.appendChild(barDiv);

        const shapId = `sa-shap-${idx}`;
        const shapDiv = document.createElement('div');
        shapDiv.id = shapId;
        shapDiv.style.flex = '1.3';
        shapDiv.style.minWidth = '450px';
        shapDiv.style.height = '400px';
        container.appendChild(shapDiv);

        impactContainer.appendChild(container);
        
        const hasShap = res.shap_data && res.shap_data.features && res.shap_data.features.length > 0;
        
        // 1. Bar Chart Trace
        const barTraces = [{
            x: xVals,
            y: yFeats,
            type: 'bar',
            orientation: 'h',
            marker: { color: obj === 'global_success' ? '#FFD700' : 'var(--accent-color)' }
        }];
        
        const barLayout = {
            xaxis: { title: 'Importance (%)', range: [0, 100] },
            yaxis: { automargin: true },
            margin: { l: 120, b: 50, t: 20, r: 20 },
            showlegend: false,
            height: 400
        };

        Plotly.newPlot(barId, barTraces, barLayout, { responsive: true });

        if (hasShap) {
            // 2. SHAP Beeswarm Plot
            const shapTraces = [];
            const sd = res.shap_data;
            sd.features.forEach((feat, i) => {
                const sVals = sd.shap_values[i];
                const fVals = sd.feature_values[i];
                
                let numericFVals = fVals.map(v => parseFloat(v));
                let isCategorical = numericFVals.some(isNaN);
                let colorArray, cmin, cmax, colorscale, reversescale;
                
                if (isCategorical) {
                    const uniqueStr = [...new Set(fVals)];
                    colorArray = fVals.map(v => uniqueStr.indexOf(v));
                    colorscale = 'Portland'; reversescale = false;
                } else {
                    colorArray = numericFVals; colorscale = 'RdBu';
                    cmin = Math.min(...colorArray); cmax = Math.max(...colorArray);
                    reversescale = true;
                }
                
                const yBase = yFeats.indexOf(feat);
                shapTraces.push({
                    x: sVals,
                    y: sVals.map(() => yBase + (Math.random() - 0.5) * 0.4),
                    mode: 'markers',
                    type: 'scatter',
                    marker: {
                        size: 7, opacity: 0.6, color: colorArray,
                        colorscale: colorscale, cmin: cmin, cmax: cmax,
                        reversescale: reversescale, line: {width: 0.5, color: '#fff'}
                    },
                    customdata: fVals,
                    hovertemplate: `<b>Value:</b> %{customdata}<br><b>Impact:</b> %{x:.3f}<extra></extra>`
                });
            });
            
            const shapLayout = {
                xaxis: { title: 'SHAP Value (Impact)', zeroline: true, zerolinecolor: '#94a3b8' },
                yaxis: { 
                    showticklabels: false, // hide since they match the bar chart
                    range: [-0.5, yFeats.length - 0.5]
                },
                margin: { l: 20, b: 50, t: 20, r: 20 },
                showlegend: false,
                height: 400
            };
            Plotly.newPlot(shapId, shapTraces, shapLayout, { responsive: true });
        } else {
            shapDiv.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#94a3b8;">No SHAP data available for this model.</div>';
        }
    });

    const getFittingSuggestion = (status) => {
        switch(status) {
            case 'Good': return 'Reliable model for interpolation.';
            case 'Limited': return 'Moderate predictive power; use with caution.';
            case 'Poor': return 'Low predictive power; consider removing irrelevant features or conducting more experiments to improve data coverage.';
            default: return '';
        }
    };

    let relHtml = `
        <div style="margin-bottom: 10px; font-weight: 600; color: var(--accent-color);">
            Active Model: ${document.getElementById('sa-model-select').options[document.getElementById('sa-model-select').selectedIndex].text}
        </div>
        <table class="setup-table" style="width: 100%; border-collapse: collapse; margin-top: 10px;">
            <thead>
                <tr style="text-align: left; border-bottom: 2px solid var(--border-color);">
                    <th style="padding: 10px;">Objective</th>
                    <th style="padding: 10px;">Predictive Power (R²)</th>
                    <th style="padding: 10px;">Status</th>
                    <th style="padding: 10px;">Suggestion</th>
                </tr>
            </thead>
            <tbody>`;
    objNames.forEach(obj => {
        const r = result.results[obj];
        const statusClass = r.fit_flag.toLowerCase();
        let nameDisplay = `<b>${obj}</b>`;
        if (obj === 'global_success') nameDisplay = `<span style="color: var(--accent-color);">✨ <b>Global Success Index</b></span>`;
        if (obj === 'pareto_optimal') nameDisplay = `<i>Pareto Optimal Flag</i>`;

        relHtml += `
            <tr style="border-bottom: 1px solid var(--border-color);">
                <td style="padding: 10px;">${nameDisplay}</td>
                <td style="padding: 10px;">${r.reliability ?? 'N/A'}</td>
                <td style="padding: 10px;"><span class="badge ${statusClass}">${r.fit_flag}</span></td>
                <td style="padding: 10px; font-size: 0.9rem; color: var(--text-secondary);">${getFittingSuggestion(r.fit_flag)}</td>
            </tr>`;
    });
    document.getElementById('sa-reliability-info').innerHTML = relHtml;

    document.getElementById('sa-cat-interactions-container').style.minHeight = '500px';

    const estDiv = document.getElementById('sa-estimator-inputs');
    const feats = window.saSelectedFeatures || currentColumns.filter(c => columnRoles[c] === 'feature');
    
    estDiv.innerHTML = feats.map(f => {
        // Detect if this feature is categorical: check if any value in currentData for this column is non-numeric
        const colIdx = currentColumns.indexOf(f);
        const uniqueVals = [...new Set(currentData.map(r => r[colIdx]).filter(v => v !== null && v !== ''))]; 
        const isNumeric = uniqueVals.every(v => !isNaN(parseFloat(v)) && isFinite(v));
        
        if (isNumeric) {
            return `
            <div class="tweak-item">
                <label>${f}</label>
                <input type="number" class="sa-est-input" data-col="${f}" step="any">
            </div>`;
        } else {
            const options = uniqueVals.map(v => `<option value="${v}">${v}</option>`).join('');
            return `
            <div class="tweak-item">
                <label>${f}</label>
                <select class="sa-est-input" data-col="${f}">${options}</select>
            </div>`;
        }
    }).join('');
}

async function runEstimate() {
    const inputs = {};
    document.querySelectorAll('.sa-est-input').forEach(el => inputs[el.dataset.col] = el.value);
    
    const featList = (window.saSelectedFeatures || currentColumns.filter(c => columnRoles[c] === 'feature'))
        .map(c => ({name: c, range: 'x'}));
    
    // Use the SA-configured objectives - ONLY those included
    const objList = Array.from(document.querySelectorAll('#sa-objectives-list tr'))
        .filter(tr => tr.querySelector('.sa-o-include').checked)
        .map(tr => ({
            name: tr.querySelector('.sa-o-type').dataset.col,
            type: tr.querySelector('.sa-o-type').value,
            target: tr.querySelector('.sa-o-target').value,
            importance: tr.querySelector('.sa-o-importance').value
        }));
    
    document.getElementById('sa-estimate-results').innerHTML = '<div style="color: var(--text-secondary); animation: pulse 1.5s infinite;"><i>Calculating predictions...</i></div>';

    try {
        const res = await fetch('/estimate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                data: currentData, columns: currentColumns,
                features: featList,
                objectives: objList,
                inputs, 
                model: document.getElementById('sa-model-select').value,
                params: {
                    n_estimators: document.getElementById('sa-rf-trees').value,
                    max_depth: document.getElementById('sa-rf-depth').value,
                    mlp_layers: document.getElementById('sa-mlp-layers').value,
                    mlp_iter: document.getElementById('sa-mlp-iter').value
                }
            })
        });
        const result = await res.json();
        if (result.error) {
            document.getElementById('sa-estimate-results').innerHTML = `<span style="color:red">Error: ${result.error}</span>`;
            return;
        }
        let html = '<div style="margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid var(--border-color);">';
        html += `<b>Global Success Score: <span style="font-size: 1.2rem; color: var(--accent-color);">${result.global_success}%</span></b></div>`;
        html += '<b>Predictions:</b><br>';
        for (const obj in result.predictions) {
            const val = result.predictions[obj];
            const succ = result.success_scores[obj];
            html += `<div style="display: flex; justify-content: space-between; margin: 4px 0;">
                        <span><b>${obj}:</b> ${val}</span>
                        <span style="color: ${succ > 80 ? '#22c55e' : (succ > 50 ? '#f59e0b' : '#ef4444')}">Success: ${succ}%</span>
                     </div>`;
        }
        document.getElementById('sa-estimate-results').innerHTML = html;
    } catch(err) {
        document.getElementById('sa-estimate-results').innerHTML = `<span style="color:red">Request failed: ${err.message}</span>`;
        console.error(err);
    }
}

function renderTrendPlot() {
    if (!currentData || currentData.length === 0) return;
    const objCols = currentColumns.filter(c => columnRoles[c] === 'objective');
    
    const traces = objCols.map(col => {
        const colIdx = currentColumns.indexOf(col);
        const xDist = [];
        const yDist = [];
        
        currentData.forEach((row, i) => {
            const rawVal = row[colIdx];
            const parsed = parseFloat(rawVal);
            if (rawVal !== '' && !isNaN(parsed)) {
                xDist.push(i + 1);
                yDist.push(parsed);
            }
        });

        return {
            x: xDist,
            y: yDist,
            name: col,
            mode: 'lines+markers',
            line: { width: 3 },
            marker: { size: 8 }
        };
    }).filter(t => t.x.length > 0);

    if (traces.length === 0) {
        document.getElementById('bo-visuals').classList.add('hidden');
        return;
    }

    document.getElementById('bo-visuals').classList.remove('hidden');
    
    const layout = { 
        title: objCols.length === 1 ? `Trend: ${objCols[0]}` : 'Objectives Trend',
        autosize: true,
        height: 400,
        margin: { l: 60, r: 40, t: 80, b: 60 },
        legend: { orientation: 'h', y: -0.2 }
    };

    Plotly.newPlot('trend-plot', traces, layout, { responsive: true });
    
    setTimeout(() => {
        Plotly.Plots.resize('trend-plot');
    }, 100);

    // Always try to render Pareto, even if we just updated Trend
    renderParetoPlot();
}

function detectParetoIndices(data, objCols) {
    const dataPoints = data.map(r => {
        const point = {};
        objCols.forEach(col => point[col] = parseFloat(r[currentColumns.indexOf(col)]) || 0);
        return point;
    });

    return dataPoints.map((p1, i) => {
        const isDominated = dataPoints.some((p2, j) => {
            if (i === j) return false;
            let p2AtLeastAsGoodAsP1 = true;
            let p2StrictlyBetterThanP1 = false;

            for (const col of objCols) {
                const config = objectiveConfigs[col] || { type: 'maximize', importance: 100 };
                const v1 = p1[col];
                const v2 = p2[col];
                const target = parseFloat(config.target || 0);
                const weight = parseFloat(config.importance || 0) / 100;

                let v1Score, v2Score;
                if (config.type === 'maximize') { v1Score = v1; v2Score = v2; }
                else if (config.type === 'minimize') { v1Score = -v1; v2Score = -v2; }
                else { v1Score = -Math.abs(v1 - target); v2Score = -Math.abs(v2 - target); }

                v1Score *= weight;
                v2Score *= weight;

                if (v2Score < v1Score) p2AtLeastAsGoodAsP1 = false;
                if (v2Score > v1Score) p2StrictlyBetterThanP1 = true;
            }
            return p2AtLeastAsGoodAsP1 && p2StrictlyBetterThanP1;
        });
        return !isDominated;
    }).map((val, i) => val ? i : -1).filter(idx => idx !== -1);
}

function renderParetoPlot() {
    const objCols = currentColumns.filter(c => columnRoles[c] === 'objective' && (objectiveConfigs[c] ? objectiveConfigs[c].included : true));
    const featCols = currentColumns.filter(c => columnRoles[c] === 'feature');
    const section = document.getElementById('pareto-plot-section');
    const plotEl = document.getElementById('pareto-plot');
    const tableCont = document.getElementById('pareto-table-container');
    const exportBtn = document.getElementById('pareto-export-btn');
    const titleEl = document.getElementById('pareto-title');

    if (objCols.length < 2) {
        section.classList.add('hidden');
        return;
    }

    // Filter data to only include complete experiments
    const completeData = currentData.filter(row => {
        return objCols.every(col => {
            const val = row[currentColumns.indexOf(col)];
            return val !== '' && !isNaN(parseFloat(val));
        });
    });

    if (completeData.length === 0) {
        section.classList.add('hidden');
        return;
    }
    
    section.classList.remove('hidden');
    const paretoIndices = detectParetoIndices(completeData, objCols);
    const isPareto = new Array(completeData.length).fill(false);
    paretoIndices.forEach(idx => isPareto[idx] = true);

    const getHoverText = (idx) => {
        let text = `<b>Experiment ${idx + 1}</b><br><br>`;
        objCols.forEach(col => {
            text += `${col}: ${completeData[idx][currentColumns.indexOf(col)]}<br>`;
        });
        text += `<br><i>Conditions:</i><br>`;
        featCols.forEach(col => {
            text += `${col}: ${completeData[idx][currentColumns.indexOf(col)]}<br>`;
        });
        return text;
    };

    const transformPoints = (indices, colsToUse) => {
        return indices.map(idx => {
            const p = {};
            colsToUse.forEach(col => {
                const cfg = objectiveConfigs[col];
                const rawVal = parseFloat(completeData[idx][currentColumns.indexOf(col)]) || 0;
                let plotVal = rawVal;
                if (cfg && cfg.type === 'target') {
                    plotVal = Math.abs(rawVal - parseFloat(cfg.target || 0));
                }
                const importance = cfg ? parseFloat(cfg.importance || 100) : 100;
                p[col] = plotVal * (importance / 100);
            });
            return p;
        });
    };

    const getAxisTitle = (col) => {
        const cfg = objectiveConfigs[col] || { type: 'maximize', target: 0, importance: 100 };
        let title = `${col}`;
        if (cfg.type === 'target') title = `Dist. to ${cfg.target || 0} (${col})`;
        else title = `${col} (${cfg.type === 'maximize' ? 'Max' : 'Min'})`;
        const imp = parseFloat(cfg.importance || 100);
        if (imp < 100) title += ` [Weighted: ${imp}%]`;
        return title;
    };

    const getAxisLayout = (col) => {
        const cfg = objectiveConfigs[col] || { type: 'maximize' };
        const layout = { title: getAxisTitle(col) };
        if (cfg.type === 'minimize' || cfg.type === 'target') {
            layout.autorange = 'reversed';
        }
        return layout;
    };

    if (objCols.length > 3) {
        // Table View
        titleEl.textContent = "Pareto Optimal Conditions";
        plotEl.classList.add('hidden');
        tableCont.classList.remove('hidden');
        exportBtn.classList.remove('hidden');
        
        const head = document.getElementById('pareto-table-head');
        const body = document.getElementById('pareto-table-body');
        
        head.innerHTML = '<tr>' + currentColumns.map(c => `<th>${c}</th>`).join('') + '</tr>';
        body.innerHTML = paretoIndices.map(idx => {
            const row = completeData[idx];
            return '<tr>' + row.map(v => `<td>${v}</td>`).join('') + '</tr>';
        }).join('');
        
    } else {
        // Plot View (2D or 3D)
        titleEl.textContent = "Pareto Front";
        plotEl.classList.remove('hidden');
        tableCont.classList.add('hidden');
        exportBtn.classList.add('hidden');

        let traces = [];
        if (objCols.length === 2) {
            const dominatedIndices = completeData.map((_, i) => i).filter(i => !isPareto[i]);
            const optimalIndices = paretoIndices;
            
            const dpDominated = transformPoints(dominatedIndices, objCols);
            const dpOptimal = transformPoints(optimalIndices, objCols);

            traces = [
                {
                    x: dpDominated.map(p => p[objCols[0]]),
                    y: dpDominated.map(p => p[objCols[1]]),
                    customdata: dominatedIndices.map(i => getHoverText(i)),
                    hovertemplate: '%{customdata}<extra></extra>',
                    mode: 'markers', name: 'Dominated', type: 'scatter',
                    marker: { size: 8, color: '#94a3b8', opacity: 0.5 }
                },
                {
                    x: dpOptimal.map(p => p[objCols[0]]),
                    y: dpOptimal.map(p => p[objCols[1]]),
                    customdata: optimalIndices.map(i => getHoverText(i)),
                    hovertemplate: '%{customdata}<extra></extra>',
                    mode: 'markers', name: 'Pareto Optimal', type: 'scatter',
                    marker: { size: 12, color: '#F97316', symbol: 'circle', line: { width: 1, color: '#EA580C' } }
                }
            ];

            const layout = { 
                title: `Pareto Front: ${objCols[0]} vs ${objCols[1]}`,
                xaxis: getAxisLayout(objCols[0]),
                yaxis: getAxisLayout(objCols[1]),
                autosize: true, margin: { l: 60, r: 40, t: 80, b: 60 },
                legend: { orientation: 'h', y: -0.2 }
            };
            Plotly.newPlot('pareto-plot', traces, layout, { responsive: true });
        } else {
            // 3D Plot
            const cols = objCols.slice(0, 3);
            const dominatedIndices = completeData.map((_, i) => i).filter(i => !isPareto[i]);
            const optimalIndices = paretoIndices;

            const dpDominated = transformPoints(dominatedIndices, cols);
            const dpOptimal = transformPoints(optimalIndices, cols);

            traces = [
                {
                    x: dpDominated.map(p => p[cols[0]]),
                    y: dpDominated.map(p => p[cols[1]]),
                    z: dpDominated.map(p => p[cols[2]]),
                    customdata: dominatedIndices.map(i => getHoverText(i)),
                    hovertemplate: '%{customdata}<extra></extra>',
                    mode: 'markers', name: 'Dominated', type: 'scatter3d',
                    marker: { size: 4, color: '#94a3b8', opacity: 0.3 }
                },
                {
                    x: dpOptimal.map(p => p[cols[0]]),
                    y: dpOptimal.map(p => p[cols[1]]),
                    z: dpOptimal.map(p => p[cols[2]]),
                    customdata: optimalIndices.map(i => getHoverText(i)),
                    hovertemplate: '%{customdata}<extra></extra>',
                    mode: 'markers', name: 'Pareto Optimal', type: 'scatter3d',
                    marker: { size: 7, color: '#F97316', symbol: 'circle' }
                }
            ];
            const layout = { 
                title: `Pareto Front (3D): ${cols.join(', ')}`,
                scene: { 
                    xaxis: getAxisLayout(cols[0]), 
                    yaxis: getAxisLayout(cols[1]), 
                    zaxis: getAxisLayout(cols[2]) 
                },
                autosize: true, margin: { l: 0, r: 0, t: 80, b: 0 },
                legend: { orientation: 'h', y: -0.1 }
            };
            Plotly.newPlot('pareto-plot', traces, layout, { responsive: true });
        }
        
        setTimeout(() => Plotly.Plots.resize('pareto-plot'), 150);
    }
}

async function exportData(data, columns) {
    const res = await fetch('/export', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ data, columns })
    });
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'export.csv'; a.click();
}

async function exportSAReport() {
    const overlay = document.getElementById('loading-overlay');
    document.getElementById('loading-text').textContent = "Generating Dynamic Report...";
    overlay.classList.remove('hidden');

    try {
        const timestamp = new Date().toLocaleString();
        const activeModel = document.getElementById('sa-model-select').options[document.getElementById('sa-model-select').selectedIndex].text;
        
        // 1. Collect all Plotly data from active plots
        const plotsData = [];
        const capturePlot = (id, title) => {
            const el = document.getElementById(id);
            if (!el || !el.data) return;
            plotsData.push({ id, title, data: el.data, layout: el.layout });
        };

        capturePlot('sa-num-corr', 'Numerical Correlation Matrix');
        
        document.querySelectorAll('[id^="sa-parcoords-"]').forEach(el => {
            const title = el.layout?.title?.text || el.layout?.title || 'Parallel Coordinates';
            capturePlot(el.id, title);
        });

        document.querySelectorAll('[id^="sa-bar-"], [id^="sa-shap-"]').forEach(el => {
            const title = el.layout?.title?.text || el.layout?.title || 'Feature Impact';
            capturePlot(el.id, title);
        });

        const reliabilityHtml = document.getElementById('sa-reliability-info').innerHTML;
        const catInteractions = window.lastSAResult ? window.lastSAResult.cat_interactions : {};
        const objNames = window.lastSAResult ? window.lastSAResult.extended_obj_names : [];

        let reportHtml = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>EDOS Analysis Report - ${timestamp}</title>
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; color: #1e293b; background: #f8fafc; max-width: 1200px; margin: 40px auto; line-height: 1.6; padding: 0 20px; }
        .card { background: white; border-radius: 12px; border: 1px solid #e2e8f0; padding: 25px; margin-bottom: 30px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .header { border-bottom: 2px solid #3b82f6; padding-bottom: 20px; margin-bottom: 40px; display: flex; justify-content: space-between; align-items: flex-end; }
        h1 { color: #0f172a; margin: 0; font-size: 2.2rem; }
        .meta { color: #64748b; font-size: 0.95rem; text-align: right; }
        h2 { color: #1e293b; margin-top: 0; border-left: 5px solid #3b82f6; padding-left: 15px; margin-bottom: 20px; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th { background: #f1f5f9; text-align: left; padding: 12px; border: 1px solid #e2e8f0; font-weight: 600; }
        td { padding: 12px; border: 1px solid #e2e8f0; }
        .badge { padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; }
        .good { background: #dcfce7; color: #166534; }
        .limited { background: #fef9c3; color: #854d0e; }
        .poor { background: #fee2e2; color: #991b1b; }
        .impact-group { display: flex; flex-wrap: wrap; gap: 20px; }
        .plot-box { flex: 1; min-width: 450px; height: 450px; }
        .full-plot { width: 100%; height: 500px; }
        .controls { display: flex; gap: 15px; margin-bottom: 20px; background: #f8fafc; padding: 15px; border-radius: 8px; border: 1px solid #e2e8f0; }
        .tweak-item { flex: 1; display: flex; flex-direction: column; gap: 5px; }
        select { padding: 8px; border-radius: 6px; border: 1px solid #cbd5e1; background: white; font-family: inherit; }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>EDOS Analysis Report</h1>
            <p style="margin: 5px 0 0 0; color: #64748b;">Statistical Analysis Model Results</p>
        </div>
        <div class="meta">
            <b>Model:</b> ${activeModel}<br>
            <b>Date:</b> ${timestamp}
        </div>
    </div>

    <div class="card">
        <h2>Model Reliability & Validation</h2>
        ${reliabilityHtml}
    </div>

    <div class="card" id="num-corr-section" style="display:none;">
        <h2>Numerical Correlation Matrix</h2>
        <div id="sa-num-corr" class="full-plot"></div>
    </div>

    <div class="card" id="cat-section" style="display:none;">
        <h2>Categorical Interaction Analysis</h2>
        <div class="controls">
            <div class="tweak-item">
                <label>Select Pair:</label>
                <select id="report-cat-pair">
                    ${Object.keys(catInteractions).map(p => `<option value="${p}">${p}</option>`).join('')}
                </select>
            </div>
            <div class="tweak-item">
                <label>Select Objective:</label>
                <select id="report-cat-obj">
                    ${objNames.map(o => `<option value="${o}">${o}</option>`).join('')}
                </select>
            </div>
        </div>
        <div id="sa-cat-interactions-container" class="full-plot"></div>
    </div>

    <div id="parcoords-section"></div>
    <div id="impact-section"></div>

    <script>
        const plotsData = ${JSON.stringify(plotsData)};
        const catInteractions = ${JSON.stringify(catInteractions)};
        
        // 1. Render Specific Plots (Correlation)
        const corrPlot = plotsData.find(p => p.id === 'sa-num-corr');
        if (corrPlot) {
            document.getElementById('num-corr-section').style.display = 'block';
            Plotly.newPlot('sa-num-corr', corrPlot.data, corrPlot.layout, { responsive: true });
        }

        // 2. Render Categorical Interaction (Dynamic)
        if (Object.keys(catInteractions).length > 0) {
            document.getElementById('cat-section').style.display = 'block';
            
            const updateCatPlot = () => {
                const pair = document.getElementById('report-cat-pair').value;
                const obj = document.getElementById('report-cat-obj').value;
                const data = (catInteractions[pair] || {})[obj];
                if (!data) return;
                
                const trace = {
                    z: data.z, x: data.x, y: data.y,
                    type: 'heatmap', colorscale: 'RdBu', zmin: 0, zmax: 1
                };
                const layout = {
                    title: \`Optimal Outcome: \${obj}\`,
                    xaxis: { title: pair.split(' vs ')[1], automargin: true },
                    yaxis: { title: pair.split(' vs ')[0], automargin: true },
                    margin: { l: 150, b: 80, t: 50, r: 50 },
                    height: 500
                };
                Plotly.newPlot('sa-cat-interactions-container', [trace], layout, { responsive: true });
            };
            
            document.getElementById('report-cat-pair').addEventListener('change', updateCatPlot);
            document.getElementById('report-cat-obj').addEventListener('change', updateCatPlot);
            updateCatPlot();
        }

        // 3. Render ParCoords list
        const pcSection = document.getElementById('parcoords-section');
        const pcPlots = plotsData.filter(p => p.id.startsWith('sa-parcoords'));
        if (pcPlots.length > 0) {
            const head = document.createElement('h2');
            head.textContent = 'Parallel Coordinates Analysis';
            pcSection.appendChild(head);
            pcPlots.forEach(p => {
                const div = document.createElement('div');
                div.id = p.id;
                div.className = 'card full-plot';
                pcSection.appendChild(div);
                Plotly.newPlot(p.id, p.data, p.layout, { responsive: true });
            });
        }

        // 4. Render Impact Groups
        const impactSection = document.getElementById('impact-section');
        const barPlots = plotsData.filter(p => p.id.startsWith('sa-bar'));
        if (barPlots.length > 0) {
            const head = document.createElement('h2');
            head.textContent = 'Feature Impact Analysis';
            impactSection.appendChild(head);
            barPlots.forEach((p, idx) => {
                const group = document.createElement('div');
                group.className = 'card';
                group.innerHTML = '<h3>' + (p.layout.title?.text || p.layout.title || 'Feature Impact Group') + '</h3><div class="impact-group"></div>';
                const container = group.querySelector('.impact-group');
                
                const barDiv = document.createElement('div');
                barDiv.id = p.id;
                barDiv.className = 'plot-box';
                container.appendChild(barDiv);
                
                const shapPlot = plotsData.find(pd => pd.id === 'sa-shap-' + p.id.split('-')[2]);
                if (shapPlot) {
                    const shapDiv = document.createElement('div');
                    shapDiv.id = shapPlot.id;
                    shapDiv.className = 'plot-box';
                    container.appendChild(shapDiv);
                }
                
                impactSection.appendChild(group);
                Plotly.newPlot(p.id, p.data, p.layout, { responsive: true });
                if (shapPlot) {
                  const finalLayout = Object.assign({}, shapPlot.layout);
                  finalLayout.yaxis = Object.assign({}, finalLayout.yaxis, {showticklabels: true});
                  Plotly.newPlot(shapPlot.id, shapPlot.data, finalLayout, { responsive: true });
                }
            });
        }
    </script>
</body>
</html>`;

        const blob = new Blob([reportHtml], { type: 'text/html' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `EDOS_Report_${new Date().toISOString().slice(0,10)}.html`;
        a.click();
        
    } catch (err) {
        alert("Dynamic report generation failed: " + err.message);
        console.error(err);
    } finally {
        overlay.classList.add('hidden');
    }
}
