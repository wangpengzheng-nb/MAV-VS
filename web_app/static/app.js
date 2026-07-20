let currentTaskId = null;
let eventSource = null;

async function startPipeline() {
  const query = document.getElementById('queryInput').value.trim();
  const protein = document.getElementById('proteinInput').files[0];
  const library = document.getElementById('libraryInput').files[0];
  if (query.length < 10 || !protein || !library) {
    alert('请填写至少10个字符的任务描述，并上传PDB蛋白和SMI/CSV/SDF分子库'); return;
  }
  const form = new FormData();
  form.append('query', query); form.append('protein', protein); form.append('library', library);
  form.append('center', document.getElementById('centerInput').value.trim());
  form.append('size', document.getElementById('sizeInput').value.trim());
  form.append('key_residues', document.getElementById('residueInput').value.trim());
  form.append('ligand_id', document.getElementById('ligandInput').value.trim());
  form.append('cpu_only', document.getElementById('cpuOnly').checked ? 'true' : 'false');
  form.append('baseline', document.getElementById('baselineMode').checked ? 'true' : 'false');
  const button = document.getElementById('runBtn'); button.disabled = true; button.textContent = '提交中…';
  document.getElementById('progressSection').style.display = 'block';
  try {
    const response = await fetch('/api/tasks', {method: 'POST', body: form});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || '提交失败');
    currentTaskId = data.task_id; document.getElementById('taskId').textContent = currentTaskId;
    connectProgress(currentTaskId);
  } catch (error) { alert(error.message); resetUI(); }
}

function connectProgress(taskId) {
  if (eventSource) eventSource.close();
  eventSource = new EventSource(`/api/progress/${taskId}`);
  eventSource.onmessage = async event => {
    const data = JSON.parse(event.data);
    document.getElementById('pctText').textContent = `${data.percent || 0}%`;
    document.getElementById('stepMsg').textContent = `${data.step || ''}: ${data.message || data.status}`;
    if (data.status === 'done') { eventSource.close(); await fetchResult(taskId); }
    if (data.status === 'error') { eventSource.close(); alert(data.message || '任务失败'); await fetchResult(taskId); }
  };
}

async function fetchResult(taskId) {
  const response = await fetch(`/api/result/${taskId}`); const data = await response.json();
  const container = document.getElementById('reportContent');
  if (data.status !== 'done') {
    const failureReport = data.result?.reports?.failure_report_html || '';
    container.innerHTML = `<h2>任务预检失败</h2><p>${escapeHtml(data.error || '请查看任务状态和日志。')}</p>` +
      (failureReport ? `<p>失败报告: ${escapeHtml(failureReport)}</p>` : '');
  } else {
    const result = data.result || {}; const hits = result.top_hits || []; const pocket = result.pocket_resolution?.selected_pocket || {};
    container.innerHTML = `<h2>口袋预检</h2><p>ID: ${escapeHtml(pocket.pocket_id || '')} | 来源: ${escapeHtml(pocket.source || '')} | 置信度: ${escapeHtml(pocket.confidence || '')}</p>` +
      `<p>中心: ${escapeHtml(JSON.stringify(pocket.center || []))} | 尺寸: ${escapeHtml(JSON.stringify(pocket.size || []))}</p>` +
      `<h2>最终候选 (${hits.length})</h2><p>报告: ${escapeHtml(result.reports?.report_html || '')}</p>` +
      `<table><thead><tr><th>Rank</th><th>Source ID</th><th>Affinity</th><th>Final score</th></tr></thead><tbody>` +
      hits.map(hit => `<tr><td>${escapeHtml(hit.rank)}</td><td>${escapeHtml(hit.source_id)}</td><td>${escapeHtml(hit.docking_affinity)}</td><td>${escapeHtml(hit.final_score)}</td></tr>`).join('') + '</tbody></table>' +
      `<h3>尚缺证据</h3><p>${escapeHtml((result.evidence_gaps || []).join(', ') || '无')}</p>`;
  }
  document.getElementById('resultSection').style.display = 'block'; resetUI();
}

function resetUI() { const button = document.getElementById('runBtn'); button.disabled = false; button.innerHTML = '<span class="btn-icon">🚀</span> 开始虚拟筛选'; }
function escapeHtml(value) { return String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
