/**
 * 中国象棋AI对战 - 主页面
 * 设计风格：温润木质棋馆风
 * 
 * 布局：左侧棋盘 + 右侧控制面板
 * 深色木纹背景，温暖的琥珀色调
 */

import { useXiangqi } from '@/hooks/useXiangqi';
import XiangqiBoard from '@/components/XiangqiBoard';
import GamePanel from '@/components/GamePanel';
import MoveHistory from '@/components/MoveHistory';
import { motion } from 'framer-motion';

const HERO_BG = "https://private-us-east-1.manuscdn.com/sessionFile/0iQ5idVkYnM9XGEYojjGXZ/sandbox/7DzdJjol8DZmVBHwR410Wo-img-2_1770624204000_na1fn_ZGFyay13YWxudXQtYmc.jpg?x-oss-process=image/resize,w_1920,h_1920/format,webp/quality,q_80&Expires=1798761600&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvMGlRNWlkVmtZbk05WEdFWW9qakdYWi9zYW5kYm94LzdEemRKam9sOERabVZCSHdSNDEwV28taW1nLTJfMTc3MDYyNDIwNDAwMF9uYTFmbl9aR0Z5YXkxM1lXeHVkWFF0WW1jLmpwZz94LW9zcy1wcm9jZXNzPWltYWdlL3Jlc2l6ZSx3XzE5MjAsaF8xOTIwL2Zvcm1hdCx3ZWJwL3F1YWxpdHkscV84MCIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc5ODc2MTYwMH19fV19&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=RNTrUp6E5LcSBkxCjg5SO3BLkN9cqmfgsLgB9lXo~n4awXBnZkx92yxdxYeajclh0VPU341-hKe-qIUw61RB7KSUOjfoKduvFb3DEfEh07rihcYZ-WXrL64vXNBxPlLT~uFAfRBo7cdlahOK~SXs2-6uxdXcjLBOXEHuMAaeaqkOhF8dA34u2OWU9LZYGMmYGy~XXR-HsIhCttyppv31RCZwvcb4ORyf6tCZ6w56hs4TLV5fX8xaFsxC40g2RIyFp9Xl7OHtr6I8rwoDKPAF5eUTj3IiKt8KyAzueXi3y2x4O4eonNWT~iNn4ObeGt6TOJpX7L-NaIRGwLCTNjAJoQ__";

export default function Home() {
  const game = useXiangqi();

  return (
    <div
      className="min-h-screen relative"
      style={{
        backgroundImage: `url(${HERO_BG})`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        backgroundAttachment: 'fixed',
      }}
    >
      {/* 暗色覆盖层 */}
      <div className="absolute inset-0 bg-black/50 backdrop-blur-[2px]" />

      {/* 内容 */}
      <div className="relative z-10">
        {/* 顶部标题栏 */}
        <header className="py-4 border-b border-border/20">
          <div className="container flex items-center justify-between">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="flex items-center gap-3"
            >
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-red-800 to-red-950 
                flex items-center justify-center shadow-lg shadow-red-900/30">
                <span className="text-xl font-bold text-red-200" style={{ fontFamily: "var(--font-brush)" }}>
                  棋
                </span>
              </div>
              <div>
                <h1 className="text-lg font-bold text-foreground tracking-wide"
                  style={{ fontFamily: "var(--font-display)" }}>
                  中国象棋AI
                </h1>
                <p className="text-xs text-muted-foreground -mt-0.5">
                  AlphaZero 强化学习
                </p>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="text-xs text-muted-foreground hidden sm:block"
            >
              基于 PyTorch · MCTS · 自对弈训练
            </motion.div>
          </div>
        </header>

        {/* 主内容区 */}
        <main className="container py-6">
          <div className="flex flex-col lg:flex-row items-start justify-center gap-6">
            {/* 棋盘区域 */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="flex-shrink-0"
            >
              <div className="p-3 rounded-xl bg-card/30 backdrop-blur-sm border border-border/20
                shadow-2xl shadow-black/30">
                <XiangqiBoard
                  board={game.board}
                  flipped={game.humanSide === 'black'}
                  selectedPiece={game.selectedPiece}
                  validMoves={game.validMoves}
                  lastMove={game.lastMove}
                  isHumanTurn={game.isHumanTurn}
                  onCellClick={game.handleCellClick}
                />
              </div>
            </motion.div>

            {/* 右侧面板 */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="flex flex-col gap-4 w-full lg:w-auto"
            >
              <GamePanel
                currentPlayer={game.currentPlayer}
                gameResult={game.gameResult}
                humanSide={game.humanSide}
                aiLevel={game.aiLevel}
                isAIThinking={game.isAIThinking}
                isHumanTurn={game.isHumanTurn}
                moveCount={game.moveHistory.length}
                onNewGame={game.newGame}
                onUndo={game.undoMove}
              />
              <MoveHistory moves={game.moveHistory} />
            </motion.div>
          </div>
        </main>

        {/* 底部 */}
        <footer className="py-4 border-t border-border/10">
          <div className="container text-center">
            <p className="text-xs text-muted-foreground/60">
              中国象棋强化学习项目 · PyTorch + AlphaZero + MCTS
            </p>
          </div>
        </footer>
      </div>
    </div>
  );
}
