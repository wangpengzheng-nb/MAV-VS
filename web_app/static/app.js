const TERMINAL = new Set(['succeeded', 'failed', 'cancelled']);

let currentTaskId = null;
let eventSource = null;
let lastTask = null;
let lastSnapshot = null;
let selectedPhaseId = null;

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('taskForm').addEventListener('submit', startPipeline);
  document.getElementById('queryInput').addEventListener('input', updateQueryCount);
  document.getElementById('proteinInput').addEventListener('change', event => updateFileName(event, 'proteinFileName'));
  document.getElementById('libraryInput').addEventListener('change', event => {
    updateFileName(event, 'libraryFileName');
    updateDefaultLibraryNotice();
  });
  document.getElementById('refreshTasks').addEventListener('click', loadRecentTasks);
  document.getElementById('copyTaskId').addEventListener('click', copyTaskId);
  document.getElementById('resumeBtn').addEventListener('click', resumeCurrentTask);
  document.getElementById('cancelBtn').addEventListener('click', cancelCurrentTask);
  document.getElementById('pauseBtn').addEventListener('click', pauseCurrentTask);
  document.querySelectorAll('.output-tab').forEach(button => button.addEventListener('click', () => switchTab(button.dataset.tab)));
  updateQueryCount();
  updateDefaultLibraryNotice();
  loadHealth();
  loadRecentTasks();
  const remembered = localStorage.getItem('autovs.activeTaskId');
  if (remembered) openTask(remembered, {quiet: true});
});

async function loadHealth() {
  const container = document.getElementById('systemState');
  try {
    const response = await fetch('/api/health');
    if (!response.ok) throw new Error('health check failed');
    const data = await response.json();
    const capabilities = Array.isArray(data.capabilities) ? data.capabilities : [];
    const unavailable = capabilities.filter(item => item.availability === 'unavailable').length;
    const degraded = capabilities.filter(item => item.availability === 'degraded').length;
    const state = data.status === 'unavailable' ? 'unavailable' : (unavailable || degraded ? 'degraded' : 'available');
    container.className = `system-state ${state}`;
    document.getElementById('healthText').textContent = data.status === 'unavailable'
      ? '默认库或核心环境不可用'
      : unavailable || degraded
      ? `环境可运行 · ${unavailable + degraded} 项降级`
      : '计算环境可用';
  } catch (_) {
    container.className = 'system-state unavailable';
    document.getElementById('healthText').textContent = '无法读取环境状态';
  }
}

async function loadRecentTasks() {
  const container = document.getElementById('recentTasks');
  try {
    const response = await fetch('/api/tasks?limit=12');
    if (!response.ok) throw new Error('无法读取任务历史');
    const tasks = (await response.json()).tasks || [];
    if (!tasks.length) {
      container.innerHTML = '<p class="muted-empty">还没有任务记录。</p>';
      return;
    }
    container.innerHTML = tasks.map(task => `
      <button class="recent-task status-${escapeHtml(task.status)} ${task.task_id === currentTaskId ? 'active' : ''}" type="button" data-task-id="${escapeHtml(task.task_id)}">
        <i></i><span class="recent-copy"><strong>${escapeHtml(task.query || '未命名筛选任务')}</strong><small>${escapeHtml(task.task_id)}</small></span>
        <time>${escapeHtml(shortTime(task.updated_at))}</time>
        ${TERMINAL.has(task.status) ? `<span class="recent-task-delete" title="永久删除" data-delete-id="${escapeHtml(task.task_id)}">×</span>` : ''}
      </button>`).join('');
    container.querySelectorAll('.recent-task').forEach(button => button.addEventListener('click', () => openTask(button.dataset.taskId)));
    container.querySelectorAll('.recent-task-delete').forEach(btn => btn.addEventListener('click', event => {
      event.stopPropagation();
      deleteRecentTask(btn.dataset.deleteId);
    }));
  } catch (error) {
    container.innerHTML = `<p class="muted-empty">${escapeHtml(error.message)}</p>`;
  }
}

async function deleteRecentTask(taskId) {
  if (!confirm(`确定要永久删除任务 ${taskId} 及其所有关联文件吗？此操作不可撤销。`)) return;
  try {
    const response = await fetch(`/api/tasks/${encodeURIComponent(taskId)}?permanent=true`, {method: 'DELETE'});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || '删除失败');
    showToast(`任务 ${taskId} 已永久删除`);
    // 如果正在查看被删除的任务，清空工作区
    if (currentTaskId === taskId) {
      disconnectProgress();
      currentTaskId = null;
      lastTask = null;
      selectedPhaseId = null;
      localStorage.removeItem('autovs.activeTaskId');
      document.getElementById('emptyState').hidden = false;
      document.getElementById('taskWorkspace').hidden = true;
    }
    loadRecentTasks();
  } catch (error) {
    showToast(error.message, 'error');
  }
}

async function startPipeline(event) {
  event.preventDefault();
  const targetGene = document.getElementById('targetGeneInput').value.trim();
  const requirements = document.getElementById('queryInput').value.trim();
  const query = `靶点基因: ${targetGene}。${requirements}`;
  const protein = document.getElementById('proteinInput').files[0];
  const library = document.getElementById('libraryInput').files[0];
  const errorBox = document.getElementById('formError');
  errorBox.hidden = true;
  if (!targetGene || targetGene.length < 1) {
    showFormError('请填写靶点基因英文缩写（如 PTGER2、EP2、BCL2）。');
    return;
  }
  if (requirements.length < 5) {
    showFormError('请至少填写 5 个字符的筛选要求。');
    return;
  }
  if (protein && !protein.name.toLowerCase().endsWith('.pdb')) {
    showFormError('蛋白文件必须是预处理后的 .pdb 文件。');
    return;
  }
  if (library && !/\.(smi|smiles)$/i.test(library.name)) {
    showFormError('分子库只接受 .smi 或 .smiles，且每行必须为 molecule_id<TAB>SMILES。');
    return;
  }
  if (document.getElementById('baselineMode').checked && !protein) {
    showFormError('基础链路诊断会跳过调研，因此必须上传预处理后的 PDB。');
    return;
  }
  const form = new FormData();
  form.append('query', query);
  form.append('target_gene', targetGene);
  if (protein) form.append('protein', protein);
  if (library) form.append('library', library);
  form.append('center', document.getElementById('centerInput').value.trim());
  form.append('size', document.getElementById('sizeInput').value.trim());
  form.append('key_residues', document.getElementById('residueInput').value.trim());
  form.append('ligand_id', document.getElementById('ligandInput').value.trim());
  form.append('cpu_only', document.getElementById('cpuOnly').checked ? 'true' : 'false');
  form.append('baseline', document.getElementById('baselineMode').checked ? 'true' : 'false');
  setSubmitState(true);
  try {
    const response = await fetch('/api/tasks', {method: 'POST', body: form});
    const data = await response.json();
    if (!response.ok) throw new Error(formatApiError(data.detail));
    const warning = (data.warnings || [])[0];
    showToast(warning || `任务 ${data.task_id} 已进入队列`);
    await openTask(data.task_id);
    loadRecentTasks();
  } catch (error) {
    showFormError(error.message);
  } finally {
    setSubmitState(false);
  }
}

async function openTask(taskId, {quiet = false} = {}) {
  disconnectProgress();
  selectedPhaseId = null;
  try {
    const response = await fetch(`/api/tasks/${encodeURIComponent(taskId)}`);
    if (!response.ok) throw new Error(response.status === 404 ? '任务记录不存在' : '无法读取任务状态');
    const task = await response.json();
    currentTaskId = task.task_id;
    lastTask = task;
    localStorage.setItem('autovs.activeTaskId', currentTaskId);
    renderFullTask(task);
    if (!TERMINAL.has(task.status)) connectProgress(task.task_id);
    loadRecentTasks();
  } catch (error) {
    if (!quiet) showToast(error.message, 'error');
    localStorage.removeItem('autovs.activeTaskId');
  }
}

function connectProgress(taskId) {
  disconnectProgress();
  const connection = document.getElementById('connectionState');
  connection.textContent = '正在连接实时状态';
  eventSource = new EventSource(`/api/progress/${encodeURIComponent(taskId)}`);
  eventSource.onopen = () => { connection.textContent = '实时状态已连接'; };
  eventSource.onmessage = async event => {
    const snapshot = JSON.parse(event.data);
    renderSnapshot(snapshot);
    if (TERMINAL.has(snapshot.status)) {
      disconnectProgress();
      await refreshCurrentTask();
      loadRecentTasks();
    }
  };
  eventSource.onerror = () => {
    connection.textContent = '连接中断，正在自动重连';
  };
}

function disconnectProgress() {
  if (eventSource) eventSource.close();
  eventSource = null;
}

async function refreshCurrentTask() {
  if (!currentTaskId) return;
  const response = await fetch(`/api/tasks/${encodeURIComponent(currentTaskId)}`);
  if (!response.ok) return;
  const task = await response.json();
  lastTask = task;
  renderFullTask(task);
}

function renderFullTask(task) {
  lastTask = task;
  document.getElementById('taskQuery').textContent = task.request?.query || task.query || '虚拟筛选任务';
  renderInputManifest(task.input_manifest || task.result?.input_manifest);
  renderSnapshot(snapshotFromTask(task));
  renderOutputs(task);
}

function renderInputManifest(manifest) {
  const container = document.getElementById('inputBindingSummary');
  if (!manifest) {
    container.innerHTML = '<span class="binding-chip"><b>INPUT</b> 等待输入绑定</span>';
    return;
  }
  const library = manifest.library_asset || {};
  const target = manifest.target_asset || {};
  const accepted = library.accepted_records == null ? '' : ` · ${Number(library.accepted_records).toLocaleString('zh-CN')} 个`;
  container.innerHTML = `
    <span class="binding-chip"><b>LIBRARY</b> ${escapeHtml(library.source === 'builtin' ? library.version || '内置库' : library.original_filename || '用户库')}${escapeHtml(accepted)}</span>
    <span class="binding-chip"><b>TARGET</b> ${escapeHtml(target.source === 'user' ? target.original_filename || '用户 PDB' : target.pdb_id || '调研后获取')}</span>
    <span class="binding-chip"><b>LOCK</b> 分子库锁定 · ${target.locked ? '结构已锁定' : '等待结构获取'}</span>
    ${(manifest.warnings || []).map(item => `<span class="binding-warning">${escapeHtml(item)}</span>`).join('')}`;
}

function snapshotFromTask(task) {
  const phases = task.progress || [];
  const counted = phases.filter(phase => !(phase.status === 'skipped'
    && (phase.message?.startsWith('基础链路') || phase.message?.startsWith('未包含') || phase.message?.startsWith('已锁定用户'))));
  const completed = counted.filter(phase => ['succeeded', 'failed', 'quarantined', 'cancelled'].includes(phase.status)).length;
  const current = phases.find(phase => phase.status === 'running')
    || [...phases].reverse().find(phase => phase.status === 'failed')
    || [...phases].reverse().find(phase => phase.status === 'succeeded');
  return {
    task_id: task.task_id,
    status: task.status,
    percent: task.status === 'succeeded' ? 100 : Math.floor(100 * completed / Math.max(1, counted.length)),
    current_phase: current,
    phases,
    jobs: task.jobs || [],
    error: task.error || '',
    updated_at: task.updated_at,
  };
}

function renderSnapshot(snapshot) {
  lastSnapshot = snapshot;
  currentTaskId = snapshot.task_id;
  document.getElementById('emptyState').hidden = true;
  document.getElementById('taskWorkspace').hidden = false;
  document.getElementById('taskId').textContent = snapshot.task_id;
  const status = document.getElementById('taskStatus');
  status.textContent = statusLabel(snapshot.status).toUpperCase();
  status.className = `status-pill ${snapshot.status}`;
  document.getElementById('resumeBtn').hidden = !['failed', 'cancelled', 'paused'].includes(snapshot.status);
  document.getElementById('cancelBtn').hidden = ['succeeded', 'failed', 'cancelled', 'paused'].includes(snapshot.status);
  document.getElementById('pauseBtn').hidden = snapshot.status !== 'running';
  document.getElementById('percentText').textContent = `${snapshot.percent || 0}%`;
  document.getElementById('meterFill').style.width = `${Math.max(0, Math.min(100, snapshot.percent || 0))}%`;
  document.querySelector('.meter-track').classList.toggle('running', snapshot.status === 'running');
  const current = snapshot.current_phase;
  document.getElementById('currentPhaseLabel').textContent = current ? current.label : statusLabel(snapshot.status);
  document.getElementById('currentPhaseMessage').textContent = current?.error || current?.message || snapshot.error || '等待下一个阶段。';
  const live = document.getElementById('liveBadge');
  live.classList.toggle('active', snapshot.status === 'running');
  live.innerHTML = snapshot.status === 'running' ? '<i></i> LIVE' : `<i></i> ${escapeHtml(statusLabel(snapshot.status).toUpperCase())}`;
  renderStages(snapshot.phases || []);
  const failed = (snapshot.phases || []).find(phase => phase.status === 'failed');
  if (failed && selectedPhaseId !== failed.phase_id) showPhase(failed);
}

function renderStages(phases) {
  const container = document.getElementById('stageList');
  if (!phases.length) {
    container.innerHTML = '<p class="muted-empty">正在初始化执行时间线…</p>';
    return;
  }
  container.innerHTML = phases.map((phase, index) => `
    <button class="phase-row status-${escapeHtml(phase.status)} ${phase.phase_id === selectedPhaseId ? 'selected' : ''}" type="button" data-phase-id="${escapeHtml(phase.phase_id)}">
      <i class="phase-icon">${phaseIcon(phase.status, index + 1)}</i>
      <span class="phase-copy"><strong>${escapeHtml(phase.label)}</strong><small>${escapeHtml(phase.error || phase.message || statusLabel(phase.status))}</small></span>
      <time class="phase-time">${escapeHtml(shortTime(phase.updated_at))}</time>
    </button>`).join('');
  container.querySelectorAll('.phase-row').forEach(button => button.addEventListener('click', () => {
    const phase = phases.find(item => item.phase_id === button.dataset.phaseId);
    if (phase) showPhase(phase);
  }));
}

async function showPhase(phase) {
  selectedPhaseId = phase.phase_id;
  document.querySelectorAll('.phase-row').forEach(row => row.classList.toggle('selected', row.dataset.phaseId === selectedPhaseId));
  const container = document.getElementById('diagnosticContent');
  const jobId = phase.metadata?.job_id;
  if (!jobId || !currentTaskId) {
    // 规划阶段(调研/策略/投票等)没有job_id, 直接显示摘要, 不请求诊断
    container.innerHTML = renderPhaseSummary(phase, false);
    return;
  }
  container.innerHTML = renderPhaseSummary(phase, true);
  try {
    const response = await fetch(`/api/tasks/${encodeURIComponent(currentTaskId)}/jobs/${encodeURIComponent(jobId)}/diagnostics`);
    if (!response.ok) throw new Error('无法读取该工具步骤的诊断数据');
    const data = await response.json();
    if (selectedPhaseId === phase.phase_id) container.innerHTML = renderDiagnostics(phase, data);
  } catch (error) {
    if (selectedPhaseId === phase.phase_id) container.innerHTML = renderPhaseSummary({...phase, error: phase.error || error.message});
  }
}

function renderPhaseSummary(phase, loading = false) {
  // 展示所有任务级产物（无 job_id），方便用户随时下载中间文件
  const taskArtifacts = (lastTask?.artifacts || []).filter(item => !item.job_id);
  return `
    <div class="diagnostic-head"><span class="diag-status ${escapeHtml(phase.status)}">${escapeHtml(statusLabel(phase.status))}</span><h4>${escapeHtml(phase.label)}</h4><p>${escapeHtml(phase.message || '该阶段尚未产生执行消息。')}</p></div>
    ${phase.error ? `<div class="error-block">${escapeHtml(phase.error)}</div>` : ''}
    <dl class="diag-meta"><dt>阶段 ID</dt><dd>${escapeHtml(phase.phase_id)}</dd><dt>更新时间</dt><dd>${escapeHtml(formatDate(phase.updated_at))}</dd>${phase.metadata?.step_id ? `<dt>工具步骤</dt><dd>${escapeHtml(phase.metadata.step_id)}</dd>` : ''}</dl>
    ${taskArtifacts.length ? `<div class="diag-links"><h5>📥 可下载中间产物</h5>${taskArtifacts.map(artifactLink).join('')}</div>` : ''}
    ${loading ? '<p class="muted-empty">正在读取工具日志…</p>' : ''}`;
}

function renderDiagnostics(phase, data) {
  const job = data.job || {};
  const snippets = data.snippets || [];
  const artifacts = data.artifacts || [];
  return `
    <div class="diagnostic-head"><span class="diag-status ${escapeHtml(job.status || phase.status)}">${escapeHtml(statusLabel(job.status || phase.status))}</span><h4>${escapeHtml(phase.label)}</h4><p>${escapeHtml(phase.message || '')}</p></div>
    ${(phase.error || job.status === 'failed') ? `<div class="error-block">${escapeHtml(phase.error || job.message || '工具执行失败')}</div>` : ''}
    <dl class="diag-meta"><dt>Job ID</dt><dd>${escapeHtml(job.job_id || '')}</dd><dt>Step ID</dt><dd>${escapeHtml(job.step_id || '')}</dd><dt>Action</dt><dd>${escapeHtml(job.action_type || '')}</dd>${job.slurm_job_id ? `<dt>Slurm ID</dt><dd>${escapeHtml(job.slurm_job_id)}</dd>` : ''}<dt>更新时间</dt><dd>${escapeHtml(formatDate(job.updated_at))}</dd></dl>
    ${snippets.length ? `<div class="log-group"><h5>日志与异常详情</h5>${snippets.map(item => `<details class="log-entry" ${/failure/i.test(item.name) ? 'open' : ''}><summary>${escapeHtml(item.name)}${item.truncated ? ' · 仅显示末尾' : ''}</summary><pre>${escapeHtml(item.content)}</pre></details>`).join('')}</div>` : ''}
    ${artifacts.length ? `<div class="diag-links">${artifacts.map(artifactLink).join('')}</div>` : '<p class="muted-empty">该步骤没有登记日志产物。</p>'}`;
}

function renderOutputs(task) {
  const result = task.result;
  const artifacts = task.artifacts || [];
  document.getElementById('artifactCount').textContent = artifacts.length;
  document.getElementById('tabArtifacts').innerHTML = renderArtifacts(task.task_id, artifacts);
  if (!result) {
    document.getElementById('tabOverview').innerHTML = '<div class="output-placeholder">任务运行完成后，这里会显示候选化合物与可复现报告。</div>';
    document.getElementById('tabPocket').innerHTML = '<div class="output-placeholder">等待口袋预检完成。</div>';
    return;
  }
  if (task.status === 'failed') {
    document.getElementById('tabOverview').innerHTML = `<div class="failure-result"><h4>任务在产生科学结果前终止</h4><p>${escapeHtml(task.error || result.error || '请在执行时间线中查看失败阶段。')}</p></div>`;
    document.getElementById('tabPocket').innerHTML = '<div class="output-placeholder">口袋可能尚未通过质量门禁，请查看诊断面板和失败报告。</div>';
    return;
  }
  const hits = result.top_hits || [];
  const pocket = result.pocket_resolution?.selected_pocket;
  const gaps = result.evidence_gaps || [];
  document.getElementById('tabOverview').innerHTML = `
    <div class="result-summary"><div class="metric-card"><span>最终候选</span><strong>${hits.length}</strong></div><div class="metric-card"><span>主口袋置信度</span><strong>${escapeHtml(pocket?.confidence || '—')}</strong></div><div class="metric-card"><span>待补证据</span><strong>${gaps.length}</strong></div></div>
    ${hits.length ? `<table class="result-table"><thead><tr><th>Rank</th><th>Source ID</th><th>Affinity</th><th>PLIP</th><th>Final score</th></tr></thead><tbody>${hits.map(hit => `<tr><td>${escapeHtml(hit.rank || '')}</td><td>${escapeHtml(hit.source_id || '')}</td><td>${escapeHtml(hit.docking_affinity ?? '—')}</td><td>${escapeHtml(hit.plip_score ?? '—')}</td><td>${escapeHtml(hit.final_score ?? '—')}</td></tr>`).join('')}</tbody></table>` : '<div class="output-placeholder">没有可交付的候选化合物。</div>'}`;
  document.getElementById('tabPocket').innerHTML = renderPocket(result.pocket_resolution);
}

function renderPocket(resolution) {
  if (!resolution?.selected_pocket) return '<div class="output-placeholder">没有口袋解析结果。</div>';
  const candidates = [resolution.selected_pocket, ...(resolution.alternate_pockets || [])];
  return `<div class="pocket-grid">${candidates.map((pocket, index) => `
    <article class="pocket-card ${index === 0 ? 'selected' : ''}"><span class="pocket-badge">${index === 0 ? 'SELECTED POCKET' : `ALTERNATE ${index}`}</span><h4>${escapeHtml(pocket.pocket_id)}</h4>
      <dl class="pocket-data"><dt>来源</dt><dd>${escapeHtml(pocket.source)}</dd><dt>置信度</dt><dd>${escapeHtml(pocket.confidence)}</dd><dt>中心</dt><dd>${escapeHtml(formatVector(pocket.center))}</dd><dt>尺寸</dt><dd>${escapeHtml(formatVector(pocket.size))}</dd><dt>残基</dt><dd>${escapeHtml((pocket.residues || []).join(', ') || '—')}</dd></dl>
      ${(pocket.quality_gates || []).length ? `<ul class="gate-list">${pocket.quality_gates.map(gate => `<li><b>${escapeHtml(gate.status)}</b> · ${escapeHtml(gate.name)} — ${escapeHtml(gate.detail)}</li>`).join('')}</ul>` : ''}
      ${(pocket.evidence || []).length ? `<ul class="evidence-list">${pocket.evidence.map(item => `<li>${escapeHtml(item.description)}</li>`).join('')}</ul>` : ''}
    </article>`).join('')}</div>`;
}

function renderArtifacts(taskId, artifacts) {
  if (!artifacts.length) return '<div class="output-placeholder">任务产物将在执行过程中持续登记。</div>';
  return `<div class="artifact-grid">${artifacts.map(item => `
    <article class="artifact-card"><strong title="${escapeHtml(item.name)}">${escapeHtml(item.name)}</strong><small>${escapeHtml(item.format)} · ${escapeHtml(formatBytes(item.size_bytes))}</small><a href="/api/tasks/${encodeURIComponent(taskId)}/artifacts/${item.artifact_id}">下载并检查</a></article>`).join('')}</div>`;
}

async function cancelCurrentTask() {
  if (!currentTaskId) return;
  if (!confirm('确定要取消这个任务吗？正在运行的工具步骤将被终止。')) return;
  const button = document.getElementById('cancelBtn');
  button.disabled = true;
  try {
    const response = await fetch(`/api/tasks/${encodeURIComponent(currentTaskId)}`, {method: 'DELETE'});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || '无法取消任务');
    showToast('任务已取消');
    await refreshCurrentTask();
    loadRecentTasks();
  } catch (error) {
    showToast(error.message, 'error');
  } finally {
    button.disabled = false;
  }
}

async function pauseCurrentTask() {
  if (!currentTaskId) return;
  if (!confirm('确定要暂停该任务吗？已完成阶段将保留，可在修改代码后从断点继续运行。')) return;
  const button = document.getElementById('pauseBtn');
  button.disabled = true;
  try {
    const response = await fetch(`/api/tasks/${encodeURIComponent(currentTaskId)}/pause`, {method: 'POST'});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || '无法暂停任务');
    showToast('已发出暂停信号，任务将在当前阶段完成后暂停');
    await refreshCurrentTask();
    loadRecentTasks();
  } catch (error) {
    showToast(error.message, 'error');
  } finally {
    button.disabled = false;
  }
}

async function resumeCurrentTask() {
  if (!currentTaskId) return;
  const button = document.getElementById('resumeBtn');
  button.disabled = true;
  try {
    const response = await fetch(`/api/tasks/${encodeURIComponent(currentTaskId)}/resume`, {method: 'POST'});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || '无法继续任务');
    showToast('任务已重新进入执行状态');
    await openTask(currentTaskId);
  } catch (error) {
    showToast(error.message, 'error');
  } finally {
    button.disabled = false;
  }
}

function switchTab(name) {
  document.querySelectorAll('.output-tab').forEach(button => {
    const active = button.dataset.tab === name;
    button.classList.toggle('active', active);
    button.setAttribute('aria-selected', active ? 'true' : 'false');
  });
  document.querySelectorAll('.output-view').forEach(view => view.classList.remove('active'));
  const target = document.getElementById(`tab${name.charAt(0).toUpperCase()}${name.slice(1)}`);
  if (target) target.classList.add('active');
}

async function copyTaskId() {
  if (!currentTaskId) return;
  try {
    await navigator.clipboard.writeText(currentTaskId);
    showToast('Task ID 已复制');
  } catch (_) {
    showToast(`Task ID：${currentTaskId}`);
  }
}

function updateQueryCount() {
  document.getElementById('queryCount').textContent = `${document.getElementById('queryInput').value.length} / 3000`;
}
function updateFileName(event, targetId) {
  const file = event.target.files[0];
  document.getElementById(targetId).textContent = file ? `${file.name} · ${formatBytes(file.size)}` : '尚未选择文件';
}
function updateDefaultLibraryNotice() {
  const file = document.getElementById('libraryInput').files[0];
  const notice = document.getElementById('defaultLibraryNotice');
  notice.classList.toggle('user-library', Boolean(file));
  notice.innerHTML = file
    ? '<strong>用户库将被锁定</strong> 策略生成、进化和工具执行不得替换或补充其他库。格式必须为 UTF-8、无表头，每行 <code>molecule_id&lt;TAB&gt;SMILES</code>。'
    : '<strong>默认库已启用</strong> 未上传分子库时，将锁定 PocketXMol curated 87K（87,924 个有效分子）。自定义库必须为 UTF-8、无表头，每行 <code>molecule_id&lt;TAB&gt;SMILES</code>。';
}
function formatApiError(detail) {
  if (!detail) return '任务提交失败';
  if (typeof detail === 'string') return detail;
  if (detail.error_type) {
    const line = detail.line_number ? `第 ${detail.line_number} 行` : '文件';
    return `${line}格式错误（${detail.error_type}）：${detail.detail || ''}。要求：molecule_id<TAB>SMILES。`;
  }
  return JSON.stringify(detail);
}
function setSubmitState(running) {
  const button = document.getElementById('runBtn');
  button.disabled = running;
  button.querySelector('span').textContent = running ? '正在创建任务…' : '启动新筛选';
}
function showFormError(message) {
  const box = document.getElementById('formError');
  box.textContent = message;
  box.hidden = false;
}
function showToast(message, type = 'info') {
  const toast = document.getElementById('toast');
  toast.textContent = message;
  toast.style.background = type === 'error' ? '#8f3035' : '#10242d';
  toast.hidden = false;
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => { toast.hidden = true; }, 3200);
}
function artifactLink(item) {
  const url = item.download_url || `/api/tasks/${encodeURIComponent(currentTaskId)}/artifacts/${item.artifact_id}`;
  return `<a class="artifact-link" href="${url}">${escapeHtml(item.name)} ↓</a>`;
}
function phaseIcon(status, index) {
  if (status === 'succeeded') return '✓';
  if (status === 'failed') return '!';
  if (status === 'running') return '••';
  if (status === 'skipped') return '–';
  return String(index).padStart(2, '0');
}
function statusLabel(status) {
  return ({pending: '等待', running: '运行中', succeeded: '已完成', failed: '失败', skipped: '已跳过', quarantined: '已隔离', cancelled: '已取消', paused: '已暂停'})[status] || status || '未知';
}
function formatVector(values) { return Array.isArray(values) ? values.map(value => Number(value).toFixed(2)).join(', ') : '—'; }
function formatBytes(value) {
  const bytes = Number(value || 0);
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}
function shortTime(value) {
  if (!value) return '—';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString('zh-CN', {month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit'});
}
function formatDate(value) {
  if (!value) return '—';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString('zh-CN', {hour12: false});
}
function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, character => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[character]));
}
