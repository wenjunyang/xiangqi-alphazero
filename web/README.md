# 中国象棋AI - Web 对战界面

基于 React + TypeScript + Tailwind CSS 构建的人机对战演示界面。

## 技术栈

- **React 19** + **TypeScript**
- **Tailwind CSS 4** + **shadcn/ui**
- **Framer Motion** — 动画效果
- **Vite** — 构建工具

## 功能特性

- 精美的木质棋馆风格界面（SVG 绘制棋盘，中文字符渲染棋子）
- 完整的中国象棋规则实现（含将军检测、飞将判定）
- 支持选择执红/执黑
- 4 级 AI 难度（随机 / 入门 / 业余 / 进阶）
- 走法高亮和合法走法提示
- 悔棋功能
- 走法历史记录

## 核心文件说明

```
client/src/
├── lib/
│   └── xiangqi-engine.ts    # 象棋引擎（规则、走法生成、AI搜索）
├── hooks/
│   └── useXiangqi.ts        # 游戏状态管理Hook
├── components/
│   ├── XiangqiBoard.tsx     # 棋盘SVG渲染组件
│   ├── GamePanel.tsx        # 游戏控制面板
│   └── MoveHistory.tsx      # 走法历史记录
└── pages/
    └── Home.tsx             # 主页面布局
```

## 本地运行

```bash
cd web
pnpm install
pnpm dev
```

浏览器访问 `http://localhost:3000` 即可开始对战。

## AI 引擎

界面内置了基于 **Minimax + Alpha-Beta 剪枝** 的搜索引擎作为默认 AI。如需使用训练好的神经网络模型，可将模型导出为 ONNX 格式后，通过 [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) 在浏览器端加载推理。
