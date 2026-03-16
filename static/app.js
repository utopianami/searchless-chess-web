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
  candidateIdx = -1;
  boardFlipped = (mode === 'ai') ? playerColor === 'b' : false;
  drawBoard();
  updateUI();
  renderCandidates();
  fetchAnalysis();
  if (mode === 'ai' && playerColor === 'b') {
    setTimeout(() => requestAIMove(), 200);
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
  const wrap = document.getElementById('board-wrap');
  const size = wrap.clientWidth || Math.min(window.innerWidth - 12, 500);
  wrap.style.height = size + 'px';
  const dpr = window.devicePixelRatio || 1;
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
    list.innerHTML = '<div style="padding:10px;color:#7c7a76;text-align:center;font-size:12px;">분석 중...</div>';
    return;
  }
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

  renderCaptured();
}

// ── API ──
async function fetchAnalysis() {
  if (gameOver) return;
  try {
    const res = await fetch('/api/analyse', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fen: currentFEN() }),
    });
    const data = await res.json();
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
  } catch (e) { console.error('Analysis error:', e); }
}

async function requestAIMove() {
  if (gameOver || isThinking) return;
  isThinking = true;
  updateUI();
  try {
    const res = await fetch('/api/move', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fen: currentFEN() }),
    });
    const data = await res.json();
    if (data.error) gameOver = true;
    else applyMove(data.uci, { from: data.from, to: data.to });
  } catch (e) { console.error('AI move error:', e); }
  isThinking = false;
  updateUI();
}

// ── Move ──
function applyMove(uci, moveInfo) {
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
  if (!cas) cas = '-';
  let ep = '-';
  if (piece && piece.toLowerCase() === 'p' && Math.abs(fr - tr) === 2)
    ep = String.fromCharCode(97 + ff) + (8 - (fr + tr) / 2);
  const half = (piece && piece.toLowerCase() === 'p' || fen.split(' ')[0] !== fenBoard) ? 0 : parseInt(parts[4]) + 1;
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
      if (mode === 'ai' && !gameOver) setTimeout(() => requestAIMove(), 100);
    } else { selectedSquare = null; drawBoard(); }
  }
}

// ── Navigation ──
function goTo(idx) {
  if (idx < 0 || idx >= history.length) return;
  historyIdx = idx;
  selectedSquare = null;
  candidateIdx = -1;
  candidateLocked = false;
  drawBoard();
  updateUI();
  fetchAnalysis();
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
function startApp() {
  window.addEventListener('resize', resizeCanvas);
  initGame();
  loadImages(() => { drawBoard(); });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', startApp);
} else {
  startApp();
}
