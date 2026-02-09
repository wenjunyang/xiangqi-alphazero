/**
 * 游戏控制面板
 * 设计风格：温润木质棋馆风 - 深色木纹卡片
 */

import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { 
  RotateCcw, Play, Undo2, Settings2, 
  Crown, Bot, User, Swords
} from 'lucide-react';
import { motion } from 'framer-motion';
import { GameResult, PIECE_NAMES, Move } from '@/lib/xiangqi-engine';
import type { PlayerSide, AILevel } from '@/hooks/useXiangqi';

interface GamePanelProps {
  currentPlayer: number;
  gameResult: GameResult;
  humanSide: PlayerSide;
  aiLevel: AILevel;
  isAIThinking: boolean;
  isHumanTurn: boolean;
  moveCount: number;
  onNewGame: (side?: PlayerSide, level?: AILevel) => void;
  onUndo: () => void;
}

const AI_LEVEL_NAMES: Record<AILevel, string> = {
  0: '随机',
  1: '入门',
  2: '业余',
  3: '进阶',
};

export default function GamePanel({
  currentPlayer, gameResult, humanSide, aiLevel,
  isAIThinking, isHumanTurn, moveCount,
  onNewGame, onUndo
}: GamePanelProps) {

  const resultText: Record<GameResult, string> = {
    ongoing: '',
    red_win: '红方胜！',
    black_win: '黑方胜！',
    draw: '和棋',
  };

  const isPlayerWin = (gameResult === 'red_win' && humanSide === 'red') ||
                      (gameResult === 'black_win' && humanSide === 'black');
  const isPlayerLose = (gameResult === 'red_win' && humanSide === 'black') ||
                       (gameResult === 'black_win' && humanSide === 'red');

  return (
    <div className="flex flex-col gap-4 w-full max-w-[320px]">
      {/* 对局信息 */}
      <Card className="bg-card/80 backdrop-blur border-border/50">
        <CardContent className="p-5">
          {/* 玩家信息 */}
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <div className={`w-8 h-8 rounded-full flex items-center justify-center
                ${humanSide === 'red' ? 'bg-red-900/40 text-red-400' : 'bg-zinc-700 text-zinc-300'}`}>
                <User size={16} />
              </div>
              <div>
                <div className="text-sm font-semibold text-foreground">你</div>
                <div className="text-xs text-muted-foreground">
                  {humanSide === 'red' ? '执红先行' : '执黑后手'}
                </div>
              </div>
            </div>
            <Swords size={20} className="text-muted-foreground" />
            <div className="flex items-center gap-2">
              <div>
                <div className="text-sm font-semibold text-foreground text-right">AI</div>
                <div className="text-xs text-muted-foreground text-right">
                  {AI_LEVEL_NAMES[aiLevel]}
                </div>
              </div>
              <div className={`w-8 h-8 rounded-full flex items-center justify-center
                ${humanSide === 'black' ? 'bg-red-900/40 text-red-400' : 'bg-zinc-700 text-zinc-300'}`}>
                <Bot size={16} />
              </div>
            </div>
          </div>

          {/* 当前状态 */}
          <div className="text-center py-3 rounded-lg bg-secondary/50 mb-4">
            {gameResult !== 'ongoing' ? (
              <motion.div
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                className="flex flex-col items-center gap-1"
              >
                {isPlayerWin && <Crown size={24} className="text-yellow-500" />}
                <span className={`text-lg font-bold font-[var(--font-display)]
                  ${isPlayerWin ? 'text-yellow-500' : isPlayerLose ? 'text-red-400' : 'text-muted-foreground'}`}>
                  {resultText[gameResult]}
                </span>
                <span className="text-xs text-muted-foreground">
                  {isPlayerWin ? '恭喜获胜！' : isPlayerLose ? '再接再厉' : '势均力敌'}
                </span>
              </motion.div>
            ) : (
              <div className="flex items-center justify-center gap-2">
                <div className={`w-3 h-3 rounded-full ${currentPlayer === 1 ? 'bg-red-500' : 'bg-zinc-400'}`} />
                <span className="text-sm text-foreground">
                  {isHumanTurn ? '轮到你走棋' : 'AI思考中...'}
                </span>
                {isAIThinking && (
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ repeat: Infinity, duration: 1, ease: 'linear' }}
                  >
                    <Settings2 size={14} className="text-muted-foreground" />
                  </motion.div>
                )}
              </div>
            )}
          </div>

          {/* 步数 */}
          <div className="text-center text-xs text-muted-foreground mb-4">
            第 {Math.ceil(moveCount / 2)} 回合 · 共 {moveCount} 步
          </div>

          {/* 操作按钮 */}
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              className="flex-1 bg-secondary/50 hover:bg-secondary"
              onClick={onUndo}
              disabled={moveCount < 2 || isAIThinking || gameResult !== 'ongoing'}
            >
              <Undo2 size={14} className="mr-1" />
              悔棋
            </Button>
            <Button
              variant="default"
              size="sm"
              className="flex-1"
              onClick={() => onNewGame()}
            >
              <RotateCcw size={14} className="mr-1" />
              新局
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* 设置面板 */}
      <Card className="bg-card/80 backdrop-blur border-border/50">
        <CardContent className="p-5">
          <h3 className="text-sm font-semibold text-foreground mb-3 flex items-center gap-2">
            <Settings2 size={14} />
            对局设置
          </h3>

          {/* 选择执子 */}
          <div className="mb-4">
            <label className="text-xs text-muted-foreground mb-2 block">执子方</label>
            <div className="flex gap-2">
              <button
                onClick={() => onNewGame('red', aiLevel)}
                className={`flex-1 py-2 px-3 rounded-md text-sm font-medium transition-all
                  ${humanSide === 'red'
                    ? 'bg-red-900/40 text-red-400 ring-1 ring-red-500/50'
                    : 'bg-secondary/50 text-muted-foreground hover:bg-secondary'}`}
              >
                执红
              </button>
              <button
                onClick={() => onNewGame('black', aiLevel)}
                className={`flex-1 py-2 px-3 rounded-md text-sm font-medium transition-all
                  ${humanSide === 'black'
                    ? 'bg-zinc-600/40 text-zinc-300 ring-1 ring-zinc-400/50'
                    : 'bg-secondary/50 text-muted-foreground hover:bg-secondary'}`}
              >
                执黑
              </button>
            </div>
          </div>

          {/* AI难度 */}
          <div>
            <label className="text-xs text-muted-foreground mb-2 block">AI难度</label>
            <div className="grid grid-cols-4 gap-1.5">
              {([0, 1, 2, 3] as AILevel[]).map(level => (
                <button
                  key={level}
                  onClick={() => onNewGame(humanSide, level)}
                  className={`py-1.5 px-2 rounded-md text-xs font-medium transition-all
                    ${aiLevel === level
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-secondary/50 text-muted-foreground hover:bg-secondary'}`}
                >
                  {AI_LEVEL_NAMES[level]}
                </button>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 项目说明 */}
      <Card className="bg-card/60 backdrop-blur border-border/30">
        <CardContent className="p-4">
          <p className="text-xs text-muted-foreground leading-relaxed">
            本项目是一个基于 <span className="text-foreground font-medium">AlphaZero</span> 算法的中国象棋强化学习AI。
            当前演示使用内置的 Minimax 搜索引擎。配置训练好的模型文件后，可切换为神经网络 + MCTS 驱动的AI。
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
