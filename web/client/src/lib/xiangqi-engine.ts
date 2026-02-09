/**
 * 中国象棋游戏引擎 - 前端TypeScript版本
 * 
 * 棋盘: 10行 x 9列
 * 红方(正数): 帅1 仕2 相3 马4 车5 炮6 兵7
 * 黑方(负数): 将-1 士-2 象-3 马-4 车-5 炮-6 卒-7
 */

export const ROWS = 10;
export const COLS = 9;

export const EMPTY = 0;
export const R_KING = 1;
export const R_ADVISOR = 2;
export const R_BISHOP = 3;
export const R_KNIGHT = 4;
export const R_ROOK = 5;
export const R_CANNON = 6;
export const R_PAWN = 7;

export const PIECE_NAMES: Record<number, string> = {
  0: '', 1: '帅', 2: '仕', 3: '相', 4: '马', 5: '车', 6: '炮', 7: '兵',
  [-1]: '将', [-2]: '士', [-3]: '象', [-4]: '马', [-5]: '车', [-6]: '炮', [-7]: '卒'
};

export const PIECE_CHARS: Record<number, string> = {
  1: '帅', 2: '仕', 3: '相', 4: '馬', 5: '車', 6: '炮', 7: '兵',
  [-1]: '将', [-2]: '士', [-3]: '象', [-4]: '馬', [-5]: '車', [-6]: '砲', [-7]: '卒'
};

export type Board = number[][];
export type Move = [number, number, number, number]; // [fromRow, fromCol, toRow, toCol]

export function createInitialBoard(): Board {
  const board: Board = Array.from({ length: ROWS }, () => Array(COLS).fill(0));

  // 红方（底部 row 0-4）
  board[0] = [R_ROOK, R_KNIGHT, R_BISHOP, R_ADVISOR, R_KING, R_ADVISOR, R_BISHOP, R_KNIGHT, R_ROOK];
  board[2][1] = R_CANNON; board[2][7] = R_CANNON;
  board[3][0] = R_PAWN; board[3][2] = R_PAWN; board[3][4] = R_PAWN;
  board[3][6] = R_PAWN; board[3][8] = R_PAWN;

  // 黑方（顶部 row 5-9）
  board[9] = [-R_ROOK, -R_KNIGHT, -R_BISHOP, -R_ADVISOR, -R_KING, -R_ADVISOR, -R_BISHOP, -R_KNIGHT, -R_ROOK];
  board[7][1] = -R_CANNON; board[7][7] = -R_CANNON;
  board[6][0] = -R_PAWN; board[6][2] = -R_PAWN; board[6][4] = -R_PAWN;
  board[6][6] = -R_PAWN; board[6][8] = -R_PAWN;

  return board;
}

export function cloneBoard(board: Board): Board {
  return board.map(row => [...row]);
}

function inBoard(r: number, c: number): boolean {
  return r >= 0 && r < ROWS && c >= 0 && c < COLS;
}

function isOwnPiece(board: Board, r: number, c: number, player: number): boolean {
  if (!inBoard(r, c)) return false;
  return (player === 1 && board[r][c] > 0) || (player === -1 && board[r][c] < 0);
}

function isEnemyPiece(board: Board, r: number, c: number, player: number): boolean {
  if (!inBoard(r, c)) return false;
  return (player === 1 && board[r][c] < 0) || (player === -1 && board[r][c] > 0);
}

function isEmptyOrEnemy(board: Board, r: number, c: number, player: number): boolean {
  if (!inBoard(r, c)) return false;
  return !isOwnPiece(board, r, c, player);
}

function findKing(board: Board, player: number): [number, number] | null {
  const target = player === 1 ? R_KING : -R_KING;
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      if (board[r][c] === target) return [r, c];
    }
  }
  return null;
}

function kingsFacing(board: Board): boolean {
  const rk = findKing(board, 1);
  const bk = findKing(board, -1);
  if (!rk || !bk) return false;
  if (rk[1] !== bk[1]) return false;
  const col = rk[1];
  const minR = Math.min(rk[0], bk[0]);
  const maxR = Math.max(rk[0], bk[0]);
  for (let r = minR + 1; r < maxR; r++) {
    if (board[r][col] !== EMPTY) return false;
  }
  return true;
}

function generateKingMoves(board: Board, r: number, c: number, player: number): [number, number][] {
  const moves: [number, number][] = [];
  const palace = player === 1
    ? [[0,3],[0,4],[0,5],[1,3],[1,4],[1,5],[2,3],[2,4],[2,5]]
    : [[7,3],[7,4],[7,5],[8,3],[8,4],[8,5],[9,3],[9,4],[9,5]];
  const palaceSet = new Set(palace.map(p => `${p[0]},${p[1]}`));

  for (const [dr, dc] of [[-1,0],[1,0],[0,-1],[0,1]]) {
    const nr = r + dr, nc = c + dc;
    if (palaceSet.has(`${nr},${nc}`) && isEmptyOrEnemy(board, nr, nc, player)) {
      moves.push([nr, nc]);
    }
  }
  return moves;
}

function generateAdvisorMoves(board: Board, r: number, c: number, player: number): [number, number][] {
  const moves: [number, number][] = [];
  const palace = player === 1
    ? [[0,3],[0,5],[1,4],[2,3],[2,5]]
    : [[7,3],[7,5],[8,4],[9,3],[9,5]];
  const palaceSet = new Set(palace.map(p => `${p[0]},${p[1]}`));

  for (const [dr, dc] of [[-1,-1],[-1,1],[1,-1],[1,1]]) {
    const nr = r + dr, nc = c + dc;
    if (palaceSet.has(`${nr},${nc}`) && isEmptyOrEnemy(board, nr, nc, player)) {
      moves.push([nr, nc]);
    }
  }
  return moves;
}

function generateBishopMoves(board: Board, r: number, c: number, player: number): [number, number][] {
  const moves: [number, number][] = [];
  for (const [dr, dc] of [[-2,-2],[-2,2],[2,-2],[2,2]]) {
    const nr = r + dr, nc = c + dc;
    const br = r + dr / 2, bc = c + dc / 2;
    if (!inBoard(nr, nc)) continue;
    if (player === 1 && nr > 4) continue;
    if (player === -1 && nr < 5) continue;
    if (board[br][bc] !== EMPTY) continue;
    if (isEmptyOrEnemy(board, nr, nc, player)) moves.push([nr, nc]);
  }
  return moves;
}

function generateKnightMoves(board: Board, r: number, c: number, player: number): [number, number][] {
  const moves: [number, number][] = [];
  const knightDirs: [number, number, number, number][] = [
    [-2,-1,-1,0],[-2,1,-1,0],[2,-1,1,0],[2,1,1,0],
    [-1,-2,0,-1],[-1,2,0,1],[1,-2,0,-1],[1,2,0,1]
  ];
  for (const [dr, dc, br, bc] of knightDirs) {
    const nr = r + dr, nc = c + dc;
    if (!inBoard(nr, nc)) continue;
    if (board[r + br][c + bc] !== EMPTY) continue;
    if (isEmptyOrEnemy(board, nr, nc, player)) moves.push([nr, nc]);
  }
  return moves;
}

function generateRookMoves(board: Board, r: number, c: number, player: number): [number, number][] {
  const moves: [number, number][] = [];
  for (const [dr, dc] of [[-1,0],[1,0],[0,-1],[0,1]]) {
    let nr = r + dr, nc = c + dc;
    while (inBoard(nr, nc)) {
      if (board[nr][nc] === EMPTY) {
        moves.push([nr, nc]);
      } else if (isEnemyPiece(board, nr, nc, player)) {
        moves.push([nr, nc]);
        break;
      } else {
        break;
      }
      nr += dr; nc += dc;
    }
  }
  return moves;
}

function generateCannonMoves(board: Board, r: number, c: number, player: number): [number, number][] {
  const moves: [number, number][] = [];
  for (const [dr, dc] of [[-1,0],[1,0],[0,-1],[0,1]]) {
    let nr = r + dr, nc = c + dc;
    while (inBoard(nr, nc) && board[nr][nc] === EMPTY) {
      moves.push([nr, nc]);
      nr += dr; nc += dc;
    }
    if (inBoard(nr, nc)) {
      nr += dr; nc += dc;
      while (inBoard(nr, nc)) {
        if (board[nr][nc] !== EMPTY) {
          if (isEnemyPiece(board, nr, nc, player)) moves.push([nr, nc]);
          break;
        }
        nr += dr; nc += dc;
      }
    }
  }
  return moves;
}

function generatePawnMoves(board: Board, r: number, c: number, player: number): [number, number][] {
  const moves: [number, number][] = [];
  if (player === 1) {
    if (r + 1 < ROWS && isEmptyOrEnemy(board, r + 1, c, player)) moves.push([r + 1, c]);
    if (r >= 5) {
      if (c - 1 >= 0 && isEmptyOrEnemy(board, r, c - 1, player)) moves.push([r, c - 1]);
      if (c + 1 < COLS && isEmptyOrEnemy(board, r, c + 1, player)) moves.push([r, c + 1]);
    }
  } else {
    if (r - 1 >= 0 && isEmptyOrEnemy(board, r - 1, c, player)) moves.push([r - 1, c]);
    if (r <= 4) {
      if (c - 1 >= 0 && isEmptyOrEnemy(board, r, c - 1, player)) moves.push([r, c - 1]);
      if (c + 1 < COLS && isEmptyOrEnemy(board, r, c + 1, player)) moves.push([r, c + 1]);
    }
  }
  return moves;
}

function generatePieceMoves(board: Board, r: number, c: number, player: number): [number, number][] {
  const piece = Math.abs(board[r][c]);
  switch (piece) {
    case 1: return generateKingMoves(board, r, c, player);
    case 2: return generateAdvisorMoves(board, r, c, player);
    case 3: return generateBishopMoves(board, r, c, player);
    case 4: return generateKnightMoves(board, r, c, player);
    case 5: return generateRookMoves(board, r, c, player);
    case 6: return generateCannonMoves(board, r, c, player);
    case 7: return generatePawnMoves(board, r, c, player);
    default: return [];
  }
}

function isInCheck(board: Board, player: number): boolean {
  const kingPos = findKing(board, player);
  if (!kingPos) return true;

  const enemy = -player;
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      if ((enemy === 1 && board[r][c] > 0) || (enemy === -1 && board[r][c] < 0)) {
        const targets = generatePieceMoves(board, r, c, enemy);
        if (targets.some(([tr, tc]) => tr === kingPos[0] && tc === kingPos[1])) {
          return true;
        }
      }
    }
  }
  return false;
}

export function getLegalMoves(board: Board, player: number): Move[] {
  const moves: Move[] = [];
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      if ((player === 1 && board[r][c] > 0) || (player === -1 && board[r][c] < 0)) {
        const targets = generatePieceMoves(board, r, c, player);
        for (const [tr, tc] of targets) {
          const newBoard = cloneBoard(board);
          newBoard[tr][tc] = newBoard[r][c];
          newBoard[r][c] = EMPTY;
          if (kingsFacing(newBoard)) continue;
          if (!isInCheck(newBoard, player)) {
            moves.push([r, c, tr, tc]);
          }
        }
      }
    }
  }
  return moves;
}

export function makeMove(board: Board, move: Move): Board {
  const newBoard = cloneBoard(board);
  const [fr, fc, tr, tc] = move;
  newBoard[tr][tc] = newBoard[fr][fc];
  newBoard[fr][fc] = EMPTY;
  return newBoard;
}

export type GameResult = 'ongoing' | 'red_win' | 'black_win' | 'draw';

export function checkGameOver(board: Board, currentPlayer: number): GameResult {
  if (!findKing(board, 1)) return 'black_win';
  if (!findKing(board, -1)) return 'red_win';
  const legal = getLegalMoves(board, currentPlayer);
  if (legal.length === 0) return currentPlayer === 1 ? 'black_win' : 'red_win';
  return 'ongoing';
}

// ========== 简单AI（用于无模型时的随机/贪心对手） ==========

const PIECE_VALUES: Record<number, number> = {
  1: 10000, 2: 20, 3: 20, 4: 40, 5: 90, 6: 45, 7: 10,
  [-1]: 10000, [-2]: 20, [-3]: 20, [-4]: 40, [-5]: 90, [-6]: 45, [-7]: 10
};

function evaluateBoard(board: Board, player: number): number {
  let score = 0;
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const piece = board[r][c];
      if (piece === 0) continue;
      const val = PIECE_VALUES[piece] || 0;
      if ((player === 1 && piece > 0) || (player === -1 && piece < 0)) {
        score += val;
      } else {
        score -= val;
      }
    }
  }
  return score;
}

export function getAIMove(board: Board, player: number, depth: number = 2): Move | null {
  const moves = getLegalMoves(board, player);
  if (moves.length === 0) return null;

  if (depth <= 0) {
    // 随机选择
    return moves[Math.floor(Math.random() * moves.length)];
  }

  let bestMove = moves[0];
  let bestScore = -Infinity;

  for (const move of moves) {
    const newBoard = makeMove(board, move);
    const score = -minimax(newBoard, -player, depth - 1, -Infinity, Infinity);
    if (score > bestScore) {
      bestScore = score;
      bestMove = move;
    }
  }

  return bestMove;
}

function minimax(board: Board, player: number, depth: number, alpha: number, beta: number): number {
  const result = checkGameOver(board, player);
  if (result === 'red_win') return player === 1 ? 100000 : -100000;
  if (result === 'black_win') return player === -1 ? 100000 : -100000;
  if (result === 'draw') return 0;
  if (depth <= 0) return evaluateBoard(board, player);

  const moves = getLegalMoves(board, player);
  let best = -Infinity;

  for (const move of moves) {
    const newBoard = makeMove(board, move);
    const score = -minimax(newBoard, -player, depth - 1, -beta, -alpha);
    best = Math.max(best, score);
    alpha = Math.max(alpha, score);
    if (alpha >= beta) break;
  }

  return best;
}

export function getMovesForPiece(board: Board, row: number, col: number, player: number): [number, number][] {
  const allMoves = getLegalMoves(board, player);
  return allMoves
    .filter(([fr, fc]) => fr === row && fc === col)
    .map(([, , tr, tc]) => [tr, tc] as [number, number]);
}
