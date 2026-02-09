/**
 * 中国象棋棋盘组件
 * 设计风格：温润木质棋馆风 - 拟物化设计
 * 
 * 棋盘使用SVG绘制，棋子使用中文字符渲染
 * 红方在下，黑方在上（从红方视角）
 */

import React, { useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Board, ROWS, COLS, PIECE_CHARS, Move } from '@/lib/xiangqi-engine';

interface XiangqiBoardProps {
  board: Board;
  flipped?: boolean; // 是否翻转棋盘（黑方在下）
  selectedPiece: [number, number] | null;
  validMoves: [number, number][];
  lastMove: Move | null;
  isHumanTurn: boolean;
  onCellClick: (row: number, col: number) => void;
}

const CELL_SIZE = 56;
const PADDING = 36;
const BOARD_WIDTH = (COLS - 1) * CELL_SIZE + PADDING * 2;
const BOARD_HEIGHT = (ROWS - 1) * CELL_SIZE + PADDING * 2;
const PIECE_RADIUS = 23;

export default function XiangqiBoard({
  board, flipped = false, selectedPiece, validMoves, lastMove, isHumanTurn, onCellClick
}: XiangqiBoardProps) {

  // 坐标转换：逻辑坐标 -> 屏幕坐标
  const toScreen = useCallback((row: number, col: number): [number, number] => {
    const displayRow = flipped ? row : (ROWS - 1 - row);
    const displayCol = flipped ? (COLS - 1 - col) : col;
    return [PADDING + displayCol * CELL_SIZE, PADDING + displayRow * CELL_SIZE];
  }, [flipped]);

  // 绘制棋盘线条
  const boardLines = useMemo(() => {
    const lines: React.JSX.Element[] = [];
    const id = (i: number, j: number) => `line-${i}-${j}`;

    // 横线
    for (let r = 0; r < ROWS; r++) {
      const [x1, y] = toScreen(r, 0);
      const [x2] = toScreen(r, COLS - 1);
      lines.push(<line key={`h${r}`} x1={x1} y1={y} x2={x2} y2={y} stroke="#5C3D2E" strokeWidth="1.2" />);
    }

    // 竖线（注意楚河汉界处断开）
    for (let c = 0; c < COLS; c++) {
      if (c === 0 || c === COLS - 1) {
        // 边线贯穿
        const [x, y1] = toScreen(0, c);
        const [, y2] = toScreen(ROWS - 1, c);
        lines.push(<line key={`v${c}`} x1={x} y1={y1} x2={x} y2={y2} stroke="#5C3D2E" strokeWidth="1.2" />);
      } else {
        // 上半部分
        const [x, y1] = toScreen(0, c);
        const [, y2] = toScreen(4, c);
        lines.push(<line key={`vt${c}`} x1={x} y1={y1} x2={x} y2={y2} stroke="#5C3D2E" strokeWidth="1.2" />);
        // 下半部分
        const [, y3] = toScreen(5, c);
        const [, y4] = toScreen(9, c);
        lines.push(<line key={`vb${c}`} x1={x} y1={y3} x2={x} y2={y4} stroke="#5C3D2E" strokeWidth="1.2" />);
      }
    }

    // 九宫格斜线
    const drawPalaceDiagonals = (r1: number, c1: number, r2: number, c2: number, key: string) => {
      const [x1, y1] = toScreen(r1, c1);
      const [x2, y2] = toScreen(r2, c2);
      lines.push(<line key={key} x1={x1} y1={y1} x2={x2} y2={y2} stroke="#5C3D2E" strokeWidth="1" />);
    };

    // 红方九宫
    drawPalaceDiagonals(0, 3, 2, 5, 'pd1');
    drawPalaceDiagonals(0, 5, 2, 3, 'pd2');
    // 黑方九宫
    drawPalaceDiagonals(7, 3, 9, 5, 'pd3');
    drawPalaceDiagonals(7, 5, 9, 3, 'pd4');

    return lines;
  }, [toScreen]);

  // 兵/炮位标记
  const positionMarkers = useMemo(() => {
    const markers: React.JSX.Element[] = [];
    const markPositions = [
      // 兵位
      [3, 0], [3, 2], [3, 4], [3, 6], [3, 8],
      [6, 0], [6, 2], [6, 4], [6, 6], [6, 8],
      // 炮位
      [2, 1], [2, 7], [7, 1], [7, 7]
    ];

    const drawMark = (row: number, col: number, key: string) => {
      const [cx, cy] = toScreen(row, col);
      const d = 5;
      const gap = 2;
      const parts: React.JSX.Element[] = [];

      // 四个角的L形标记
      if (col > 0) {
        parts.push(
          <path key={`${key}-tl`} d={`M${cx - gap - d},${cy - gap} L${cx - gap},${cy - gap} L${cx - gap},${cy - gap - d}`}
            fill="none" stroke="#5C3D2E" strokeWidth="1" />,
          <path key={`${key}-bl`} d={`M${cx - gap - d},${cy + gap} L${cx - gap},${cy + gap} L${cx - gap},${cy + gap + d}`}
            fill="none" stroke="#5C3D2E" strokeWidth="1" />
        );
      }
      if (col < COLS - 1) {
        parts.push(
          <path key={`${key}-tr`} d={`M${cx + gap + d},${cy - gap} L${cx + gap},${cy - gap} L${cx + gap},${cy - gap - d}`}
            fill="none" stroke="#5C3D2E" strokeWidth="1" />,
          <path key={`${key}-br`} d={`M${cx + gap + d},${cy + gap} L${cx + gap},${cy + gap} L${cx + gap},${cy + gap + d}`}
            fill="none" stroke="#5C3D2E" strokeWidth="1" />
        );
      }
      return parts;
    };

    markPositions.forEach(([r, c], i) => {
      markers.push(...drawMark(r, c, `mark-${i}`));
    });

    return markers;
  }, [toScreen]);

  // 楚河汉界文字
  const riverText = useMemo(() => {
    const [, y1] = toScreen(4, 0);
    const [, y2] = toScreen(5, 0);
    const cy = (y1 + y2) / 2;
    const [lx] = toScreen(0, 2);
    const [rx] = toScreen(0, 6);

    return (
      <>
        <text x={lx} y={cy} textAnchor="middle" dominantBaseline="central"
          fill="#6B4423" fontSize="20" fontFamily="'Ma Shan Zheng', 'ZCOOL XiaoWei', serif"
          opacity="0.7" letterSpacing="8">
          {flipped ? '漢界' : '楚河'}
        </text>
        <text x={rx} y={cy} textAnchor="middle" dominantBaseline="central"
          fill="#6B4423" fontSize="20" fontFamily="'Ma Shan Zheng', 'ZCOOL XiaoWei', serif"
          opacity="0.7" letterSpacing="8">
          {flipped ? '楚河' : '漢界'}
        </text>
      </>
    );
  }, [toScreen, flipped]);

  // 渲染棋子
  const renderPiece = useCallback((piece: number, row: number, col: number) => {
    if (piece === 0) return null;

    const [cx, cy] = toScreen(row, col);
    const isRed = piece > 0;
    const isSelected = selectedPiece && selectedPiece[0] === row && selectedPiece[1] === col;
    const isLastMoveFrom = lastMove && lastMove[0] === row && lastMove[1] === col;
    const isLastMoveTo = lastMove && lastMove[2] === row && lastMove[3] === col;
    const char = PIECE_CHARS[piece] || '';

    return (
      <motion.g
        key={`piece-${row}-${col}`}
        initial={false}
        animate={{
          x: cx,
          y: cy,
          scale: isSelected ? 1.12 : 1,
        }}
        transition={{ type: 'spring', stiffness: 300, damping: 25 }}
        onClick={() => onCellClick(row, col)}
        style={{ cursor: isHumanTurn ? 'pointer' : 'default' }}
      >
        {/* 棋子阴影 */}
        <circle cx={2} cy={3} r={PIECE_RADIUS} fill="rgba(0,0,0,0.3)" />

        {/* 棋子外圈 */}
        <circle cx={0} cy={0} r={PIECE_RADIUS}
          fill={isRed ? '#F5E6D3' : '#F5E6D3'}
          stroke={isRed ? '#8B2500' : '#1A1A1A'}
          strokeWidth="2"
        />

        {/* 棋子内圈 */}
        <circle cx={0} cy={0} r={PIECE_RADIUS - 4}
          fill="none"
          stroke={isRed ? '#8B2500' : '#1A1A1A'}
          strokeWidth="0.8"
        />

        {/* 棋子文字 */}
        <text x={0} y={1} textAnchor="middle" dominantBaseline="central"
          fill={isRed ? '#B22222' : '#1A1A1A'}
          fontSize="22" fontWeight="700"
          fontFamily="'Ma Shan Zheng', 'ZCOOL XiaoWei', serif"
        >
          {char}
        </text>

        {/* 选中高亮 */}
        {isSelected && (
          <circle cx={0} cy={0} r={PIECE_RADIUS + 3}
            fill="none" stroke="#FFD700" strokeWidth="2.5"
            opacity="0.9"
          />
        )}

        {/* 上一步标记 */}
        {(isLastMoveFrom || isLastMoveTo) && !isSelected && (
          <circle cx={0} cy={0} r={PIECE_RADIUS + 2}
            fill="none" stroke="#DAA520" strokeWidth="1.5"
            opacity="0.6" strokeDasharray="4 3"
          />
        )}
      </motion.g>
    );
  }, [toScreen, selectedPiece, lastMove, isHumanTurn, onCellClick]);

  // 渲染合法走法标记
  const renderValidMoves = useMemo(() => {
    return validMoves.map(([r, c]) => {
      const [cx, cy] = toScreen(r, c);
      const hasPiece = board[r][c] !== 0;

      return (
        <motion.g
          key={`valid-${r}-${c}`}
          initial={{ opacity: 0, scale: 0 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0 }}
          onClick={() => onCellClick(r, c)}
          style={{ cursor: 'pointer' }}
        >
          {hasPiece ? (
            // 可吃子标记
            <circle cx={cx} cy={cy} r={PIECE_RADIUS + 2}
              fill="none" stroke="#FF4444" strokeWidth="2.5"
              opacity="0.7" strokeDasharray="6 3"
            />
          ) : (
            // 可移动标记
            <circle cx={cx} cy={cy} r={7}
              fill="#DAA520" opacity="0.6"
            />
          )}
        </motion.g>
      );
    });
  }, [validMoves, board, toScreen, onCellClick]);

  // 收集所有棋子
  const pieces = useMemo(() => {
    const result: React.JSX.Element[] = [];
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        if (board[r][c] !== 0) {
          const el = renderPiece(board[r][c], r, c);
          if (el) result.push(el);
        }
      }
    }
    return result;
  }, [board, renderPiece]);

  return (
    <div className="relative inline-block">
      <svg
        width={BOARD_WIDTH}
        height={BOARD_HEIGHT}
        viewBox={`0 0 ${BOARD_WIDTH} ${BOARD_HEIGHT}`}
        className="block"
        style={{
          background: 'linear-gradient(135deg, #D4A76A 0%, #C49A5C 30%, #B8894E 60%, #C49A5C 100%)',
          borderRadius: '4px',
          boxShadow: 'inset 0 0 30px rgba(0,0,0,0.15), 0 8px 32px rgba(0,0,0,0.4)',
        }}
      >
        {/* 棋盘外框 */}
        <rect x={PADDING - 8} y={PADDING - 8}
          width={(COLS - 1) * CELL_SIZE + 16}
          height={(ROWS - 1) * CELL_SIZE + 16}
          fill="none" stroke="#5C3D2E" strokeWidth="2.5" rx="2"
        />

        {/* 棋盘线条 */}
        {boardLines}

        {/* 位置标记 */}
        {positionMarkers}

        {/* 楚河汉界 */}
        {riverText}

        {/* 点击热区 */}
        {Array.from({ length: ROWS }, (_, r) =>
          Array.from({ length: COLS }, (_, c) => {
            const [cx, cy] = toScreen(r, c);
            return (
              <rect
                key={`click-${r}-${c}`}
                x={cx - CELL_SIZE / 2}
                y={cy - CELL_SIZE / 2}
                width={CELL_SIZE}
                height={CELL_SIZE}
                fill="transparent"
                onClick={() => onCellClick(r, c)}
                style={{ cursor: isHumanTurn ? 'pointer' : 'default' }}
              />
            );
          })
        )}

        {/* 合法走法标记 */}
        <AnimatePresence>{renderValidMoves}</AnimatePresence>

        {/* 棋子 */}
        {pieces}
      </svg>
    </div>
  );
}
