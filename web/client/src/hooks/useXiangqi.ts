import { useState, useCallback, useRef, useEffect } from 'react';
import {
  Board, Move, GameResult,
  createInitialBoard, makeMove, getLegalMoves,
  checkGameOver, getAIMove, getMovesForPiece
} from '@/lib/xiangqi-engine';

export type PlayerSide = 'red' | 'black';
export type AILevel = 0 | 1 | 2 | 3; // 0=随机, 1=简单, 2=中等, 3=困难

interface MoveRecord {
  move: Move;
  captured: number;
  board: Board;
}

export function useXiangqi() {
  const [board, setBoard] = useState<Board>(createInitialBoard);
  const [currentPlayer, setCurrentPlayer] = useState<number>(1); // 1=红, -1=黑
  const [selectedPiece, setSelectedPiece] = useState<[number, number] | null>(null);
  const [validMoves, setValidMoves] = useState<[number, number][]>([]);
  const [gameResult, setGameResult] = useState<GameResult>('ongoing');
  const [humanSide, setHumanSide] = useState<PlayerSide>('red');
  const [aiLevel, setAiLevel] = useState<AILevel>(2);
  const [moveHistory, setMoveHistory] = useState<MoveRecord[]>([]);
  const [isAIThinking, setIsAIThinking] = useState(false);
  const [lastMove, setLastMove] = useState<Move | null>(null);
  const [inCheck, setInCheck] = useState(false);
  const aiTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const humanPlayer = humanSide === 'red' ? 1 : -1;
  const isHumanTurn = currentPlayer === humanPlayer;

  // AI走棋
  const doAIMove = useCallback((currentBoard: Board, player: number, level: AILevel) => {
    setIsAIThinking(true);

    // 使用setTimeout模拟思考延迟
    aiTimeoutRef.current = setTimeout(() => {
      const depth = level === 0 ? 0 : level;
      const aiMove = getAIMove(currentBoard, player, depth);

      if (aiMove) {
        const captured = currentBoard[aiMove[2]][aiMove[3]];
        const newBoard = makeMove(currentBoard, aiMove);

        setMoveHistory(prev => [...prev, { move: aiMove, captured, board: currentBoard }]);
        setBoard(newBoard);
        setLastMove(aiMove);
        setCurrentPlayer(-player);

        const result = checkGameOver(newBoard, -player);
        setGameResult(result);

        // 检查是否将军
        if (result === 'ongoing') {
          const legalMoves = getLegalMoves(newBoard, -player);
          // 简单检查：如果对方将/帅被攻击
          setInCheck(false); // 简化处理
        }
      }

      setIsAIThinking(false);
    }, 300 + Math.random() * 400);
  }, []);

  // 当轮到AI时自动走棋
  useEffect(() => {
    if (gameResult !== 'ongoing') return;
    if (!isHumanTurn && !isAIThinking) {
      doAIMove(board, currentPlayer, aiLevel);
    }
    return () => {
      if (aiTimeoutRef.current) clearTimeout(aiTimeoutRef.current);
    };
  }, [currentPlayer, isHumanTurn, gameResult, board, aiLevel, isAIThinking, doAIMove]);

  // 点击棋盘格子
  const handleCellClick = useCallback((row: number, col: number) => {
    if (gameResult !== 'ongoing' || !isHumanTurn || isAIThinking) return;

    const piece = board[row][col];

    // 如果点击了自己的棋子
    if ((humanPlayer === 1 && piece > 0) || (humanPlayer === -1 && piece < 0)) {
      setSelectedPiece([row, col]);
      const moves = getMovesForPiece(board, row, col, humanPlayer);
      setValidMoves(moves);
      return;
    }

    // 如果已选中棋子且点击了合法目标
    if (selectedPiece) {
      const isValid = validMoves.some(([r, c]) => r === row && c === col);
      if (isValid) {
        const move: Move = [selectedPiece[0], selectedPiece[1], row, col];
        const captured = board[row][col];
        const newBoard = makeMove(board, move);

        setMoveHistory(prev => [...prev, { move, captured, board }]);
        setBoard(newBoard);
        setLastMove(move);
        setSelectedPiece(null);
        setValidMoves([]);
        setCurrentPlayer(-currentPlayer);

        const result = checkGameOver(newBoard, -currentPlayer);
        setGameResult(result);
        setInCheck(false);
      } else {
        // 点击空白处取消选择
        setSelectedPiece(null);
        setValidMoves([]);
      }
    }
  }, [board, currentPlayer, gameResult, humanPlayer, isHumanTurn, isAIThinking, selectedPiece, validMoves]);

  // 新游戏
  const newGame = useCallback((side?: PlayerSide, level?: AILevel) => {
    if (aiTimeoutRef.current) clearTimeout(aiTimeoutRef.current);
    const newBoard = createInitialBoard();
    setBoard(newBoard);
    setCurrentPlayer(1);
    setSelectedPiece(null);
    setValidMoves([]);
    setGameResult('ongoing');
    setMoveHistory([]);
    setIsAIThinking(false);
    setLastMove(null);
    setInCheck(false);
    if (side !== undefined) setHumanSide(side);
    if (level !== undefined) setAiLevel(level);
  }, []);

  // 悔棋
  const undoMove = useCallback(() => {
    if (moveHistory.length < 2 || isAIThinking) return;
    if (aiTimeoutRef.current) clearTimeout(aiTimeoutRef.current);

    // 撤销两步（AI一步 + 人一步）
    const newHistory = moveHistory.slice(0, -2);
    const prevBoard = newHistory.length > 0
      ? makeMove(newHistory[newHistory.length - 1].board, newHistory[newHistory.length - 1].move)
      : createInitialBoard();

    // 实际上应该恢复到倒数第二步之前的棋盘
    const restoreBoard = moveHistory.length >= 2
      ? moveHistory[moveHistory.length - 2].board
      : createInitialBoard();

    setBoard(restoreBoard);
    setMoveHistory(newHistory);
    setCurrentPlayer(humanPlayer);
    setSelectedPiece(null);
    setValidMoves([]);
    setGameResult('ongoing');
    setIsAIThinking(false);
    setLastMove(newHistory.length > 0 ? newHistory[newHistory.length - 1].move : null);
  }, [moveHistory, isAIThinking, humanPlayer]);

  return {
    board,
    currentPlayer,
    selectedPiece,
    validMoves,
    gameResult,
    humanSide,
    aiLevel,
    moveHistory,
    isAIThinking,
    lastMove,
    inCheck,
    isHumanTurn,
    handleCellClick,
    newGame,
    undoMove,
    setHumanSide,
    setAiLevel,
  };
}
