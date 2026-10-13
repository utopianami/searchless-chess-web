// ── Chess Web App (wood board + SVG pieces) ──

const INITIAL_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

// Brown-Sand board colors
const LIGHT = '#ebecd0';
const DARK = '#b68860';
const HIGHLIGHT = 'rgba(255, 255, 50, 0.45)';
const LAST_MOVE_FROM = 'rgba(172, 162, 73, 0.55)';
const LAST_MOVE_TO = 'rgba(172, 162, 73, 0.55)';

const PIECE_VALUE = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 };
const PIECE_ORDER = ['q', 'r', 'b', 'n', 'p'];
const START_PIECES = { P: 8, N: 2, B: 2, R: 2, Q: 1, K: 1, p: 8, n: 2, b: 2, r: 2, q: 1, k: 1 };

// Piece name mapping: FEN char -> filename
const PIECE_FILE = {
  K: 'wK', Q: 'wQ', R: 'wR', B: 'wB', N: 'wN', P: 'wP',
  k: 'bK', q: 'bQ', r: 'bR', b: 'bB', n: 'bN', p: 'bP',
};

// ── State ──
let mode = 'ai';
let playerColor = 'w';
let history = [];
let historyIdx = 0;
let selectedSquare = null;
let moveProbs = [];
let allMoveProbs = [];
let whiteWinProb = 0.5;
let isThinking = false;
let gameOver = false;
let boardFlipped = false;
let candidateIdx = -1;
let candidateLocked = false;
let selectedModel = '270M';
let modelCatalog = [];
let positionVersion = 0;
let analysisVersion = 0;
let moveVersion = 0;
let apiError = '';
let analysisPending = false;
let scoreType = 'win_probability';
let lastActivity = Date.now();
let activitySent = 0;
let manuallyReleased = false;
let gpuBusy = false;

// ── Image Loading ──
const pieceImages = {};
let imagesLoaded = 0;
const totalImages = 12;

function loadImages(callback) {
  let loaded = 0;
  let failed = 0;
  function check() {
    if (loaded + failed === totalImages) callback();
  }
  for (const [fenChar, fileName] of Object.entries(PIECE_FILE)) {
    const img = new Image();
    img.onload = () => { loaded++; check(); };
    img.onerror = () => { failed++; console.warn('Failed to load:', fileName); check(); };
    img.src = `/static/pieces/${fileName}.svg`;
    pieceImages[fenChar] = img;
  }
}

// ── DOM ──
const boardCanvas = document.getElementById('board-canvas');
const ctx = boardCanvas.getContext('2d');

function initGame() {
  positionVersion++;
  moveVersion++;
  history = [{ fen: INITIAL_FEN, lastMove: null }];
  historyIdx = 0;
  // Ensure canvas is sized before drawing
  const wrap = document.getElementById('board-wrap');
  const size = wrap.clientWidth || Math.min(window.innerWidth - 12, 500);
  if (!boardCanvas._logicalSize) {
    wrap.style.height = size + 'px';
    const dpr = window.devicePixelRatio || 1;
    boardCanvas.width = size * dpr;
    boardCanvas.height = size * dpr;
    boardCanvas.style.width = size + 'px';
    boardCanvas.style.height = size + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    boardCanvas._logicalSize = size;
  }
  selectedSquare = null;
  moveProbs = [];
  allMoveProbs = [];
  whiteWinProb = 0.5;
  isThinking = false;
  gameOver = false;
  apiError = '';
  candidateIdx = -1;
  boardFlipped = (mode === 'ai') ? playerColor === 'b' : false;
  drawBoard();
  updateUI();
  renderCandidates();
  fetchAnalysis();
  if (mode === 'ai' && playerColor === 'b') {
    requestAIMove();
  }
}

// ── FEN ──
function parseFEN(fen) {
  const parts = fen.split(' ');
  const rows = parts[0].split('/');
  const board = Array(64).fill(null);
  for (let r = 0; r < 8; r++) {
    let c = 0;
    for (const ch of rows[r]) {
      if (ch >= '1' && ch <= '8') c += parseInt(ch);
      else { board[r * 8 + c] = ch; c++; }
    }
  }
  return { board, turn: parts[1] };
}

function currentFEN() { return history[historyIdx].fen; }
function currentTurn() { return parseFEN(currentFEN()).turn; }
function isAtLatest() { return historyIdx === history.length - 1; }
function logicalSize() { return boardCanvas._logicalSize || 400; }

// ── Captured Pieces ──
function getCaptured(fen) {
  const parsed = parseFEN(fen);
  const current = {};
  for (const p of parsed.board) {
    if (p) current[p] = (current[p] || 0) + 1;
  }
  const whiteCaptured = [];
  const blackCaptured = [];
  for (const [p, count] of Object.entries(START_PIECES)) {
    const diff = count - (current[p] || 0);
    for (let i = 0; i < diff; i++) {
      if (p === p.toLowerCase()) whiteCaptured.push(p);
      else blackCaptured.push(p);
    }
  }
  const sortFn = (a, b) => PIECE_VALUE[b.toLowerCase()] - PIECE_VALUE[a.toLowerCase()];
  whiteCaptured.sort(sortFn);
  blackCaptured.sort(sortFn);
  const whitePoints = whiteCaptured.reduce((s, p) => s + PIECE_VALUE[p.toLowerCase()], 0);
  const blackPoints = blackCaptured.reduce((s, p) => s + PIECE_VALUE[p.toLowerCase()], 0);
  return { whiteCaptured, blackCaptured, advantage: whitePoints - blackPoints };
}

function renderCaptured() {
  const { whiteCaptured, blackCaptured, advantage } = getCaptured(currentFEN());

  function renderPieces(captured) {
    const groups = {};
    for (const p of captured) {
      const key = p.toLowerCase();
      if (!groups[key]) groups[key] = [];
      groups[key].push(p);
    }
    let html = '';
    for (const type of PIECE_ORDER) {
      if (groups[type]) {
        html += '<span class="cap-group">';
        for (const p of groups[type]) {
          const fileName = PIECE_FILE[p];
          html += `<img src="/static/pieces/${fileName}.svg" class="cap-icon">`;
        }
        html += '</span>';
      }
    }
    return html;
  }

  document.getElementById('white-pieces').innerHTML = renderPieces(whiteCaptured);
  document.getElementById('black-pieces').innerHTML = renderPieces(blackCaptured);
  document.getElementById('white-adv').textContent = advantage > 0 ? `+${advantage}` : '';
  document.getElementById('black-adv').textContent = advantage < 0 ? `+${-advantage}` : '';
}

// ── Canvas ──
function toCanvasCoords(file, rank) {
  const sq = logicalSize() / 8;
  if (boardFlipped) return { x: (7 - file) * sq, y: rank * sq };
  return { x: file * sq, y: (7 - rank) * sq };
}

function fromCanvasCoords(cx, cy) {
  const sq = logicalSize() / 8;
  let file = Math.floor(cx / sq);
  let rank = 7 - Math.floor(cy / sq);
  if (boardFlipped) { file = 7 - file; rank = 7 - rank; }
  return { file, rank };
}

function squareToFileRank(name) {
  return { file: name.charCodeAt(0) - 97, rank: parseInt(name[1]) - 1 };
}
function fileRankToSquare(file, rank) {
  return String.fromCharCode(97 + file) + (rank + 1);
}

function resizeCanvas() {
  if (!history.length) return;
  const wrap = document.getElementById('board-wrap');
  const size = wrap.clientWidth || Math.min(window.innerWidth - 12, 500);
  const dpr = window.devicePixelRatio || 1;
  if (boardCanvas._logicalSize === size && boardCanvas.width === size * dpr) return;
  wrap.style.height = size + 'px';
  boardCanvas.width = size * dpr;
  boardCanvas.height = size * dpr;
  boardCanvas.style.width = size + 'px';
  boardCanvas.style.height = size + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  boardCanvas._logicalSize = size;
  drawBoard();
}

function drawBoard() {
  const size = logicalSize();
  const sq = size / 8;
  const parsed = parseFEN(currentFEN());
  const lastMove = history[historyIdx].lastMove;

  ctx.clearRect(0, 0, size, size);

  // Squares (wood colors)
  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      const { x, y } = toCanvasCoords(f, r);
      ctx.fillStyle = (f + r) % 2 === 0 ? DARK : LIGHT;
      ctx.fillRect(x, y, sq, sq);
    }
  }

  // Last move highlight
  if (lastMove) {
    for (const sqName of [lastMove.from, lastMove.to]) {
      const { file, rank } = squareToFileRank(sqName);
      const { x, y } = toCanvasCoords(file, rank);
      ctx.fillStyle = sqName === lastMove.from ? LAST_MOVE_FROM : LAST_MOVE_TO;
      ctx.fillRect(x, y, sq, sq);
    }
  }

  // Candidate highlight
  if (candidateIdx >= 0 && candidateIdx < moveProbs.length) {
    const m = moveProbs[candidateIdx];
    for (const sqName of [m.from, m.to]) {
      const { file, rank } = squareToFileRank(sqName);
      const { x, y } = toCanvasCoords(file, rank);
      ctx.fillStyle = 'rgba(66, 135, 245, 0.35)';
      ctx.fillRect(x, y, sq, sq);
    }
  }

  // Selected
  if (selectedSquare !== null) {
    const { file, rank } = squareToFileRank(selectedSquare);
    const { x, y } = toCanvasCoords(file, rank);
    ctx.fillStyle = HIGHLIGHT;
    ctx.fillRect(x, y, sq, sq);

    // Legal move indicators
    const validTargets = allMoveProbs.filter(m => m.from === selectedSquare);
    for (const m of validTargets) {
      const { file: tf, rank: tr } = squareToFileRank(m.to);
      const { x: tx, y: ty } = toCanvasCoords(tf, tr);
      const target = parsed.board[(7 - tr) * 8 + tf];
      if (target) {
        ctx.strokeStyle = 'rgba(0,0,0,0.18)';
        ctx.lineWidth = sq * 0.08;
        ctx.beginPath();
        ctx.arc(tx + sq / 2, ty + sq / 2, sq * 0.46, 0, Math.PI * 2);
        ctx.stroke();
      } else {
        ctx.fillStyle = 'rgba(0,0,0,0.18)';
        ctx.beginPath();
        ctx.arc(tx + sq / 2, ty + sq / 2, sq * 0.15, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  // Pieces (SVG images with fallback)
  const padding = sq * 0.05;
  const pieceSize = sq - padding * 2;
  const FALLBACK = {
    K: '\u2654', Q: '\u2655', R: '\u2656', B: '\u2657', N: '\u2658', P: '\u2659',
    k: '\u265A', q: '\u265B', r: '\u265C', b: '\u265D', n: '\u265E', p: '\u265F',
  };
  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      const piece = parsed.board[(7 - r) * 8 + f];
      if (!piece) continue;
      const { x, y } = toCanvasCoords(f, r);
      const img = pieceImages[piece];
      if (img && img.complete && img.naturalWidth > 0) {
        ctx.drawImage(img, x + padding, y + padding, pieceSize, pieceSize);
      } else {
        // Unicode fallback
        ctx.font = `${sq * 0.8}px serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = piece === piece.toUpperCase() ? '#fff' : '#000';
        ctx.fillText(FALLBACK[piece], x + sq / 2, y + sq / 2);
      }
    }
  }

  // Coordinates
  ctx.font = `bold ${sq * 0.15}px sans-serif`;
  for (let f = 0; f < 8; f++) {
    const df = boardFlipped ? 7 - f : f;
    const { x, y } = toCanvasCoords(f, boardFlipped ? 7 : 0);
    ctx.fillStyle = (df + (boardFlipped ? 7 : 0)) % 2 === 0 ? LIGHT : DARK;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    ctx.fillText(String.fromCharCode(97 + df), x + 2, y + sq - 1);
  }
  for (let r = 0; r < 8; r++) {
    const dr = boardFlipped ? 7 - r : r;
    const { x, y } = toCanvasCoords(boardFlipped ? 0 : 7, r);
    ctx.fillStyle = (dr + (boardFlipped ? 0 : 7)) % 2 === 0 ? LIGHT : DARK;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'top';
    ctx.fillText(String(dr + 1), x + sq - 2, y + 2);
  }
}

// ── Candidates ──
function renderCandidates() {
  const list = document.getElementById('cand-list');
  if (!moveProbs.length) {
    const message = gameOver ? '게임 종료' : manuallyReleased ? 'GPU 해제됨. 재시도로 분석을 다시 시작할 수 있습니다.' : apiError ? '분석을 불러오지 못했습니다. 재시도를 눌러주세요.' : '';
    list.setAttribute('aria-busy', String(!message));
    if (message) {
      const empty = document.createElement('div');
      empty.className = 'cand-empty';
      empty.textContent = message;
      list.replaceChildren(empty);
    } else {
      list.innerHTML = '<div class="cand-placeholder" aria-hidden="true"><span></span><span></span><span></span></div>'.repeat(10);
    }
    return;
  }
  list.setAttribute('aria-busy', 'false');
  const isWhiteTurn = currentTurn() === 'w';
  list.innerHTML = moveProbs.map((m, i) => {
    const pct = (m.prob * 100).toFixed(1);
    const barWidth = Math.max(5, m.norm_prob * 100);
    const rankClass = i === 0 ? 'top1' : i === 1 ? 'top2' : i === 2 ? 'top3' : 'other';
    const barClass = isWhiteTurn ? 'white-turn' : 'black-turn';
    const activeClass = i === candidateIdx ? 'active' : '';
    return `<div class="cand-row ${activeClass}" data-idx="${i}">
      <div class="cand-rank ${rankClass}">${i + 1}</div>
      <div class="cand-move">${m.uci}</div>
      <div class="cand-bar-wrap"><div class="cand-bar ${barClass}" style="width:${barWidth}%"></div></div>
      <div class="cand-pct">${pct}%</div>
    </div>`;
  }).join('');

  list.querySelectorAll('.cand-row').forEach(row => {
    row.addEventListener('click', () => {
      const idx = parseInt(row.dataset.idx);
      if (candidateLocked && candidateIdx === idx) {
        candidateIdx = -1;
        candidateLocked = false;
      } else {
        candidateIdx = idx;
        candidateLocked = true;
      }
      renderCandidates();
      drawBoard();
    });
    row.addEventListener('mouseenter', () => {
      if (!candidateLocked) { candidateIdx = parseInt(row.dataset.idx); drawBoard(); }
    });
    row.addEventListener('mouseleave', () => {
      if (!candidateLocked) { candidateIdx = -1; drawBoard(); }
    });
  });
}

// ── UI ──
function updateUI() {
  document.getElementById('eval-bar').hidden = whiteWinProb === null;
  document.getElementById('cand-header').textContent = scoreType === 'move_probability' ? '다음 수 · 수 선택 확률' : '다음 수 · 예상 승률';
  const wp = Math.round(whiteWinProb * 100);
  const bp = 100 - wp;
  document.getElementById('eval-white').style.flex = Math.max(wp, 3);
  document.getElementById('eval-black').style.flex = Math.max(bp, 3);
  document.getElementById('eval-white-pct').textContent = wp + '%';
  document.getElementById('eval-black-pct').textContent = bp + '%';

  const turn = currentTurn();
  document.getElementById('turn-dot').className = turn === 'b' ? 'black' : '';
  document.getElementById('turn-text').textContent = turn === 'w' ? '백 차례' : '흑 차례';
  document.getElementById('move-counter').textContent = `${historyIdx} / ${history.length - 1}`;
  document.getElementById('color-bar').className = mode === 'ai' ? '' : 'hidden';

  const statusEl = document.getElementById('status');
  if (gameOver) statusEl.textContent = '게임 종료';
  else if (isThinking) statusEl.textContent = 'AI 생각 중...';
  else statusEl.textContent = '';

  const modelStatus = document.getElementById('model-status');
  modelStatus.className = apiError ? 'error' : '';
  modelStatus.textContent = apiError || (manuallyReleased ? 'GPU를 해제했습니다. 다음 분석에서 자동으로 다시 준비합니다.' : analysisPending || isThinking
    ? '모델 분석 중... 처음 선택한 모델은 준비에 시간이 걸릴 수 있습니다.'
    : scoreType === 'move_probability'
      ? '선택 확률을 표시합니다. 이 모델은 승률을 예측하지 않습니다.'
      : '선택한 모델로 대국과 형세를 분석합니다.');
  document.getElementById('btn-retry').hidden = !apiError && !manuallyReleased;

  renderCaptured();
}

// ── API ──
async function fetchAnalysis() {
  if (gameOver) return;
  manuallyReleased = false;
  const version = ++analysisVersion;
  const position = positionVersion;
  const fen = currentFEN();
  const model = selectedModel;
  analysisPending = true;
  apiError = '';
  allMoveProbs = [];
  moveProbs = [];
  updateUI();
  renderCandidates();
  try {
    const res = await fetch('/api/analyse', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fen, model }),
    });
    if (!res.ok) {
      const error = await res.json().catch(() => ({}));
      throw new Error(typeof error.detail === 'string' ? error.detail : `분석 요청 실패 (${res.status}). 재시도해 주세요.`);
    }
    const data = await res.json();
    if (version !== analysisVersion || position !== positionVersion || model !== selectedModel || fen !== currentFEN()) return;
    scoreType = data.score_type;
    if (data.game_over) {
      gameOver = true;
      whiteWinProb = data.white_win_prob;
      moveProbs = [];
      allMoveProbs = [];
    } else {
      allMoveProbs = data.moves;
      moveProbs = data.moves.slice(0, 10);
      whiteWinProb = data.white_win_prob;
    }
    candidateIdx = -1;
    candidateLocked = false;
    updateUI();
    renderCandidates();
  } catch (e) {
    if (version === analysisVersion && position === positionVersion) apiError = e.message;
  } finally {
    if (version === analysisVersion) {
      analysisPending = false;
      updateUI();
      renderCandidates();
      refreshGPU();
    }
  }
}

async function requestAIMove() {
  if (gameOver || isThinking || mode !== 'ai' || currentTurn() === playerColor || !isAtLatest()) return;
  const version = ++moveVersion;
  const position = positionVersion;
  const fen = currentFEN();
  const model = selectedModel;
  isThinking = true;
  apiError = '';
  updateUI();
  try {
    const res = await fetch('/api/move', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fen, model }),
    });
    if (!res.ok) {
      const error = await res.json().catch(() => ({}));
      throw new Error(typeof error.detail === 'string' ? error.detail : `AI 요청 실패 (${res.status}). 재시도해 주세요.`);
    }
    const data = await res.json();
    if (version !== moveVersion || position !== positionVersion || model !== selectedModel || fen !== currentFEN()) return;
    if (data.error) gameOver = true;
    else applyMove(data.uci, { from: data.from, to: data.to });
  } catch (e) {
    if (version === moveVersion && position === positionVersion) apiError = e.message;
  } finally {
    if (version === moveVersion) { isThinking = false; updateUI(); }
  }
}

// ── Move ──
function applyMove(uci, moveInfo) {
  positionVersion++;
  if (!isAtLatest()) history = history.slice(0, historyIdx + 1);
  const newFen = applyMoveToFEN(currentFEN(), uci);
  history.push({ fen: newFen, lastMove: moveInfo });
  historyIdx = history.length - 1;
  selectedSquare = null;
  candidateIdx = -1;
  candidateLocked = false;
  drawBoard();
  updateUI();
  fetchAnalysis();
}

function applyMoveToFEN(fen, uci) {
  const parts = fen.split(' ');
  const board = [];
  const rows = parts[0].split('/');
  for (let r = 0; r < 8; r++) {
    board[r] = []; let c = 0;
    for (const ch of rows[r]) {
      if (ch >= '1' && ch <= '8') { for (let i = 0; i < parseInt(ch); i++) board[r][c++] = null; }
      else board[r][c++] = ch;
    }
  }
  const ff = uci.charCodeAt(0) - 97, fr = 7 - (parseInt(uci[1]) - 1);
  const tf = uci.charCodeAt(2) - 97, tr = 7 - (parseInt(uci[3]) - 1);
  const promo = uci.length === 5 ? uci[4] : null;
  const piece = board[fr][ff];
  const capturedPiece = board[tr][tf];
  board[fr][ff] = null;
  if (piece && piece.toLowerCase() === 'p' && ff !== tf && board[tr][tf] === null) board[fr][tf] = null;
  if (piece && piece.toLowerCase() === 'k' && Math.abs(ff - tf) === 2) {
    if (tf === 6) { board[fr][5] = board[fr][7]; board[fr][7] = null; }
    else if (tf === 2) { board[fr][3] = board[fr][0]; board[fr][0] = null; }
  }
  board[tr][tf] = promo ? (parts[1] === 'w' ? promo.toUpperCase() : promo) : piece;

  let fenBoard = '';
  for (let r = 0; r < 8; r++) {
    let empty = 0;
    for (let c = 0; c < 8; c++) {
      if (!board[r][c]) empty++;
      else { if (empty) { fenBoard += empty; empty = 0; } fenBoard += board[r][c]; }
    }
    if (empty) fenBoard += empty;
    if (r < 7) fenBoard += '/';
  }
  const newTurn = parts[1] === 'w' ? 'b' : 'w';
  let cas = parts[2];
  if (piece === 'K') cas = cas.replace(/[KQ]/g, '');
  if (piece === 'k') cas = cas.replace(/[kq]/g, '');
  if (piece === 'R' && ff === 7 && fr === 7) cas = cas.replace('K', '');
  if (piece === 'R' && ff === 0 && fr === 7) cas = cas.replace('Q', '');
  if (piece === 'r' && ff === 7 && fr === 0) cas = cas.replace('k', '');
  if (piece === 'r' && ff === 0 && fr === 0) cas = cas.replace('q', '');
  if (capturedPiece === 'R' && tf === 7 && tr === 7) cas = cas.replace('K', '');
  if (capturedPiece === 'R' && tf === 0 && tr === 7) cas = cas.replace('Q', '');
  if (capturedPiece === 'r' && tf === 7 && tr === 0) cas = cas.replace('k', '');
  if (capturedPiece === 'r' && tf === 0 && tr === 0) cas = cas.replace('q', '');
  if (!cas) cas = '-';
  let ep = '-';
  if (piece && piece.toLowerCase() === 'p' && Math.abs(fr - tr) === 2)
    ep = String.fromCharCode(97 + ff) + (8 - (fr + tr) / 2);
  const half = (piece && piece.toLowerCase() === 'p' || capturedPiece) ? 0 : parseInt(parts[4]) + 1;
  const full = parts[1] === 'b' ? parseInt(parts[5]) + 1 : parts[5];
  return `${fenBoard} ${newTurn} ${cas} ${ep} ${half} ${full}`;
}

// ── Input ──
function handleBoardClick(e) {
  if (isThinking || gameOver) return;
  if (!isAtLatest() && mode === 'ai') return;
  const rect = boardCanvas.getBoundingClientRect();
  const { file, rank } = fromCanvasCoords(e.clientX - rect.left, e.clientY - rect.top);
  if (file < 0 || file > 7 || rank < 0 || rank > 7) return;
  const sqName = fileRankToSquare(file, rank);
  const parsed = parseFEN(currentFEN());
  const piece = parsed.board[(7 - rank) * 8 + file];
  const turn = parsed.turn;
  if (mode === 'ai' && turn !== playerColor) return;

  if (selectedSquare === null) {
    if (piece) {
      const isW = piece === piece.toUpperCase();
      if ((turn === 'w' && isW) || (turn === 'b' && !isW)) { selectedSquare = sqName; drawBoard(); }
    }
  } else {
    if (sqName === selectedSquare) { selectedSquare = null; drawBoard(); return; }
    if (piece) {
      const isW = piece === piece.toUpperCase();
      if ((turn === 'w' && isW) || (turn === 'b' && !isW)) { selectedSquare = sqName; drawBoard(); return; }
    }
    const isLegal = allMoveProbs.some(m => m.from === selectedSquare && m.to === sqName);
    if (isLegal) {
      const fromSq = squareToFileRank(selectedSquare);
      const mp = parsed.board[(7 - fromSq.rank) * 8 + fromSq.file];
      let uci = selectedSquare + sqName;
      if (mp && mp.toLowerCase() === 'p' && (rank === 7 || rank === 0)) uci += 'q';
      applyMove(uci, { from: selectedSquare, to: sqName });
      if (mode === 'ai' && !gameOver) requestAIMove();
    } else { selectedSquare = null; drawBoard(); }
  }
}

// ── Navigation ──
function goTo(idx) {
  if (idx < 0 || idx >= history.length) return;
  positionVersion++;
  moveVersion++;
  isThinking = false;
  gameOver = false;
  historyIdx = idx;
  selectedSquare = null;
  candidateIdx = -1;
  candidateLocked = false;
  drawBoard();
  updateUI();
  fetchAnalysis();
  requestAIMove();
}

document.getElementById('btn-first').addEventListener('click', () => goTo(0));
document.getElementById('btn-prev').addEventListener('click', () => goTo(historyIdx - 1));
document.getElementById('btn-next').addEventListener('click', () => goTo(historyIdx + 1));
document.getElementById('btn-last').addEventListener('click', () => goTo(history.length - 1));

document.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowLeft') goTo(historyIdx - 1);
  if (e.key === 'ArrowRight') goTo(historyIdx + 1);
  if (e.key === 'Home') goTo(0);
  if (e.key === 'End') goTo(history.length - 1);
});

// ── Mode / Color ──
document.querySelectorAll('.mode-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    mode = btn.dataset.mode;
    initGame();
  });
});
document.querySelectorAll('.color-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.color-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    playerColor = btn.dataset.color;
    initGame();
  });
});

// ── Touch ──
boardCanvas.addEventListener('click', handleBoardClick);
let touchStart = null;
boardCanvas.addEventListener('touchstart', (e) => {
  e.preventDefault();
  touchStart = { x: e.touches[0].clientX, y: e.touches[0].clientY };
});
boardCanvas.addEventListener('touchend', (e) => {
  e.preventDefault();
  if (touchStart) {
    const t = e.changedTouches[0];
    if (Math.abs(t.clientX - touchStart.x) < 20 && Math.abs(t.clientY - touchStart.y) < 20)
      handleBoardClick({ clientX: t.clientX, clientY: t.clientY });
    touchStart = null;
  }
});

// ── Init ──
async function loadModels() {
  const res = await fetch('/api/models');
  if (!res.ok) throw new Error('모델 목록을 불러오지 못했습니다. 재시도해 주세요.');
  const data = await res.json();
  modelCatalog = data.models;
  let stored;
  try { stored = localStorage.getItem('chess-model'); } catch (_) {}
  selectedModel = modelCatalog.some(m => m.id === stored && m.available) ? stored : data.default_model;
  const select = document.getElementById('model-select');
  select.replaceChildren(...modelCatalog.map(m => {
    const option = new Option(m.label + (m.available ? '' : ' · 설치 필요'), m.id);
    option.disabled = !m.available;
    return option;
  }));
  select.value = selectedModel;
  select.disabled = false;
}

document.getElementById('model-select').addEventListener('change', (event) => {
  manuallyReleased = false;
  selectedModel = event.target.value;
  try { localStorage.setItem('chess-model', selectedModel); } catch (_) {}
  positionVersion++;
  moveVersion++;
  isThinking = false;
  gameOver = false;
  selectedSquare = null;
  candidateIdx = -1;
  candidateLocked = false;
  whiteWinProb = null;
  fetchAnalysis();
  drawBoard();
  requestAIMove();
});

document.getElementById('btn-retry').addEventListener('click', async () => {
  manuallyReleased = false;
  try {
    if (!modelCatalog.length) await loadModels();
    fetchAnalysis();
    requestAIMove();
  } catch (e) { apiError = e.message; updateUI(); }
});

async function startApp() {
  window.addEventListener('resize', resizeCanvas);
  new ResizeObserver(resizeCanvas).observe(document.getElementById('board-wrap'));
  try { await loadModels(); } catch (e) { apiError = e.message; }
  const startupError = apiError;
  initGame();
  if (startupError) { apiError = startupError; updateUI(); }
  loadImages(() => { drawBoard(); });
  refreshGPU();
  setInterval(refreshGPU, 5000);
}

async function refreshGPU() {
  if (gpuBusy) return;
  gpuBusy = true;
  try {
    const res = await fetch('/api/gpu');
    if (!res.ok) return;
    const state = await res.json();
    document.getElementById('gpu-status').textContent = state.loaded
      ? `${state.model} · ${state.loading ? 'GPU 준비 중' : `미사용 시 ${Math.ceil(state.remaining_seconds / 60)}분 후 해제`}`
      : 'GPU 해제됨 · 다음 분석 시 자동 준비';
    document.getElementById('btn-unload').disabled = !state.loaded;
    // Status polling itself never renews the lease. Only new visible activity.
    if (state.loaded && !manuallyReleased && document.visibilityState === 'visible' &&
        lastActivity > activitySent && Date.now() - activitySent >= 60000) {
      const sentAt = Date.now();
      const activity = await fetch('/api/gpu/activity', {method: 'POST',
        headers: {'Content-Type': 'application/json'}, body: JSON.stringify({model: selectedModel})});
      if (activity.ok) activitySent = sentAt;
    }
  } catch (_) {
    document.getElementById('gpu-status').textContent = 'GPU 상태 연결 대기 중';
  } finally { gpuBusy = false; }
}

for (const event of ['pointerdown', 'keydown', 'touchstart']) {
  document.addEventListener(event, () => { lastActivity = Date.now(); }, {passive: true});
}
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible') { lastActivity = Date.now(); refreshGPU(); }
});
document.getElementById('btn-unload').addEventListener('click', async () => {
  manuallyReleased = true;
  positionVersion++;
  analysisVersion++;
  moveVersion++;
  isThinking = false;
  analysisPending = false;
  try {
    const res = await fetch('/api/gpu/unload', {method: 'POST'});
    if (!res.ok) throw new Error('GPU 해제에 실패했습니다. 다시 시도해 주세요.');
    apiError = '';
    allMoveProbs = [];
    moveProbs = [];
    updateUI();
    renderCandidates();
    refreshGPU();
  } catch (e) { apiError = e.message; updateUI(); }
});

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', startApp);
} else {
  startApp();
}
