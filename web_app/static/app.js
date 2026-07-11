// AutoVS-Agent Web UI v3.0

const STEPS = ["靶点调研", "策略生成", "策略审评", "策略进化", "输出结果"];
let currentTaskId = null;
let eventSource = null;

// =========================================================================
// 流水线启动
// =========================================================================

async function startPipeline() {
  const input = document.getElementById('queryInput');
  const btn = document.getElementById('runBtn');
  const query = input.value.trim();

  if (query.length < 10) {
    alert('请输入至少10个字符的任务描述');
    return;
  }

  // 禁用输入
  btn.disabled = true;
  btn.innerHTML = '<span class="btn-icon">⏳</span> 启动中...';
  input.disabled = true;

  // 显示进度区
  document.getElementById('progressSection').style.display = 'block';
  document.getElementById('resultSection').style.display = 'none';
  resetProgress();

  try {
    const resp = await fetch('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
    });
    if (!resp.ok) {
      const err = await resp.json();
      alert(err.detail || '启动失败');
      resetUI();
      return;
    }
    const data = await resp.json();
    currentTaskId = data.task_id;

    // 连接 SSE 获取进度
    connectProgress(currentTaskId);
    btn.innerHTML = '<span class="btn-icon">🔄</span> 运行中...';
  } catch (e) {
    alert('连接失败: ' + e.message);
    resetUI();
  }
}

// =========================================================================
// SSE 进度推送
// =========================================================================

function connectProgress(taskId) {
  if (eventSource) eventSource.close();

  eventSource = new EventSource(`/api/progress/${taskId}`);

  eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.status === 'error') {
      alert('运行出错: ' + (data.message || '未知错误'));
      resetUI();
      return;
    }

    // 更新进度条
    updateProgress(data.step, data.status, data.percent, data.message);

    // 完成时获取结果
    if (data.status === 'done' && data.percent >= 100) {
      eventSource.close();
      setTimeout(() => fetchResult(taskId), 500);
    }
  };

  eventSource.onerror = () => {
    // SSE 连接断开, 检查是否有结果
    if (currentTaskId) {
      setTimeout(() => checkAndFetchResult(taskId), 1000);
    }
  };
}

// =========================================================================
// 进度条更新
// =========================================================================

function resetProgress() {
  STEPS.forEach(s => {
    const node = document.querySelector(`.pnode[data-step="${s}"]`);
    if (node) node.className = 'pnode';
  });
  document.querySelectorAll('.pconnector').forEach(c => c.className = 'pconnector');
  document.getElementById('pctText').textContent = '0%';
  document.getElementById('stepMsg').textContent = '准备中...';
}

function updateProgress(step, status, percent, message) {
  // 更新百分比
  document.getElementById('pctText').textContent = percent + '%';
  if (message) document.getElementById('stepMsg').textContent = message;

  // 更新节点状态
  const stepIdx = STEPS.indexOf(step);
  STEPS.forEach((s, i) => {
    const node = document.querySelector(`.pnode[data-step="${s}"]`);
    if (!node) return;
    if (i < stepIdx || (i === stepIdx && status === 'done')) {
      node.className = 'pnode done';
    } else if (i === stepIdx && status === 'running') {
      node.className = 'pnode active';
    }
  });

  // 更新连接线
  const connectors = document.querySelectorAll('.pconnector');
  connectors.forEach((c, i) => {
    if (i < stepIdx || (i === stepIdx && status === 'done')) {
      c.className = 'pconnector done';
    }
  });
}

// =========================================================================
// 结果获取与渲染
// =========================================================================

async function fetchResult(taskId) {
  try {
    const resp = await fetch(`/api/result/${taskId}`);
    const data = await resp.json();
    if (data.status === 'done' && data.result) {
      renderResult(data.result);
      document.getElementById('resultSection').style.display = 'block';
    }
  } catch (e) {
    console.error('获取结果失败:', e);
  }
  resetUI();
}

async function checkAndFetchResult(taskId) {
  try {
    const resp = await fetch(`/api/result/${taskId}`);
    const data = await resp.json();
    if (data.status === 'done' && data.result) {
      renderResult(data.result);
      document.getElementById('resultSection').style.display = 'block';
      updateProgress('输出结果', 'done', 100, '完成!');
    }
  } catch (e) {}
  resetUI();
}

function resetUI() {
  const btn = document.getElementById('runBtn');
  btn.disabled = false;
  btn.innerHTML = '<span class="btn-icon">🚀</span> 开始虚拟筛选';
  document.getElementById('queryInput').disabled = false;
}

// =========================================================================
// 结果渲染
// =========================================================================

function renderResult(result) {
  // 调研报告
  renderReport(result.report_md || '', result.target_name || '');

  // 进化策略
  renderStrategies(result.evolved_strategies || []);

  // 排名
  renderRanking(result.ranking || []);
}

function renderReport(md, targetName) {
  const container = document.getElementById('reportContent');
  // 简单 Markdown → HTML
  let html = md
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/^- (.+)$/gm, '<li>$1</li>')
    .replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>')
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n/g, '<br>');
  html = '<p>' + html + '</p>';
  container.innerHTML = html;
}

function renderStrategies(evolved) {
  const container = document.getElementById('strategiesList');
  if (!evolved.length) {
    container.innerHTML = '<p style="color:#94a3b8">暂无进化策略</p>';
    return;
  }

  container.innerHTML = evolved.map((s, i) => `
    <div class="strat-card" id="strat${i}">
      <div class="strat-card-header" onclick="toggleStrat('strat${i}')">
        <div>
          <div class="strat-name">${escHtml(s.name)}</div>
          <div class="strat-tagline">${escHtml(s.tagline || '')} | ${escHtml(s.approach || '')}</div>
        </div>
        <div>
          <span class="strat-badge">v2 进化版</span>
        </div>
      </div>
      <div class="strat-card-body">
        ${s.changelog && s.changelog.length ? `
        <h4>📝 进化日志</h4>
        <ul class="strat-changelog">
          ${s.changelog.map(c => `<li>${escHtml(c)}</li>`).join('')}
        </ul>` : ''}

        <h4>💡 设计原理</h4>
        <p>${escHtml((s.rationale || '').substring(0, 1000))}</p>

        <h4>🔬 筛选步骤 (${s.steps ? s.steps.length : 0}步)</h4>
        ${(s.steps || []).map(st => `
          <div class="strat-step">
            <strong>Step ${st.step_number}: ${escHtml(st.step_name || '')}</strong>
            <div style="font-size:0.85em;color:#94a3b8;margin-top:4px">
              工具: ${escHtml(st.tool || '')} | 指标: ${escHtml(st.metric || '')} | 阈值: ${escHtml(st.threshold || '')}
            </div>
            <div style="margin-top:6px">${escHtml((st.action || '').substring(0, 300))}</div>
          </div>
        `).join('')}

        <h4>✅ 优势</h4>
        <ul>${(s.strengths || []).slice(0, 5).map(x => `<li>${escHtml(x)}</li>`).join('')}</ul>

        <h4>⚠️ 剩余劣势</h4>
        <ul>${(s.weaknesses || []).slice(0, 5).map(x => `<li>${escHtml(x)}</li>`).join('')}</ul>
      </div>
    </div>
  `).join('');
}

function renderRanking(ranking) {
  const container = document.getElementById('rankingContent');
  if (!ranking.length) {
    container.innerHTML = '<p style="color:#94a3b8">暂无排名</p>';
    return;
  }
  const medals = ['🥇', '🥈', '🥉'];
  container.innerHTML = ranking.map(r => `
    <div class="rank-item">
      <div class="rank-pos">${medals[r.rank-1] || r.rank}</div>
      <div class="rank-info">
        <div class="rank-name">${escHtml(r.name)}</div>
        <div class="rank-elo">Elo: ${r.elo?.toFixed(0) || '?'} | 评分: ${r.score?.toFixed(1) || '?'}</div>
      </div>
    </div>
  `).join('');
}

// =========================================================================
// 交互
// =========================================================================

function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.querySelector(`[onclick="switchTab('${name}')"]`).classList.add('active');
  document.getElementById('tab-' + name).classList.add('active');
}

function toggleStrat(id) {
  document.getElementById(id).classList.toggle('open');
}

function escHtml(s) {
  if (!s) return '';
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// 回车提交
document.getElementById('queryInput').addEventListener('keydown', (e) => {
  if (e.ctrlKey && e.key === 'Enter') startPipeline();
});
