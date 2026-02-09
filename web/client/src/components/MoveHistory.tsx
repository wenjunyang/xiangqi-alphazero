/**
 * 走法历史记录组件
 */

import { useRef, useEffect } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { History } from 'lucide-react';
import { PIECE_NAMES, Move, Board } from '@/lib/xiangqi-engine';

interface MoveRecord {
  move: Move;
  captured: number;
  board: Board;
}

interface MoveHistoryProps {
  moves: MoveRecord[];
}

function formatMove(record: MoveRecord, index: number): string {
  const [fr, fc, tr, tc] = record.move;
  const piece = record.board[fr][fc];
  const name = PIECE_NAMES[piece] || '?';
  const captured = record.captured !== 0 ? `吃${PIECE_NAMES[record.captured]}` : '';
  return `${name}(${fc},${fr})→(${tc},${tr})${captured}`;
}

export default function MoveHistory({ moves }: MoveHistoryProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [moves.length]);

  return (
    <Card className="bg-card/80 backdrop-blur border-border/50">
      <CardContent className="p-4">
        <h3 className="text-sm font-semibold text-foreground mb-3 flex items-center gap-2">
          <History size={14} />
          走法记录
        </h3>
        <div
          ref={scrollRef}
          className="max-h-[200px] overflow-y-auto pr-1 scrollbar-thin"
          style={{ scrollbarWidth: 'thin' }}
        >
          {moves.length === 0 ? (
            <p className="text-xs text-muted-foreground text-center py-4">
              暂无走法记录
            </p>
          ) : (
            <div className="space-y-0.5">
              {moves.map((record, i) => {
                const isRed = i % 2 === 0;
                const roundNum = Math.floor(i / 2) + 1;
                return (
                  <div key={i} className="flex items-center gap-2 text-xs py-1 px-2 rounded
                    hover:bg-secondary/30 transition-colors">
                    {isRed && (
                      <span className="text-muted-foreground w-6 text-right font-mono">
                        {roundNum}.
                      </span>
                    )}
                    {!isRed && <span className="w-6" />}
                    <span className={`w-2 h-2 rounded-full flex-shrink-0
                      ${isRed ? 'bg-red-500' : 'bg-zinc-400'}`} />
                    <span className={`font-medium
                      ${isRed ? 'text-red-400' : 'text-zinc-300'}`}>
                      {formatMove(record, i)}
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
