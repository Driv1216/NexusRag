import React from 'react';
import { Layers, Activity, BarChart3, Trash2, PanelRight, Sparkles } from 'lucide-react';

interface NavbarProps {
  activeModel: string;
  onOpenEval: () => void;
  onToggleDrawer: () => void;
  onClearAll: () => void;
  isDrawerOpen: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeModel,
  onOpenEval,
  onToggleDrawer,
  onClearAll,
  isDrawerOpen
}) => {
  return (
    <header className="h-14 border-b border-slate-800/80 bg-[#0d1322]/80 backdrop-blur px-5 flex items-center justify-between z-10 shrink-0">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-sky-500 flex items-center justify-center text-white shadow-md shadow-indigo-500/20">
          <Layers className="w-4 h-4" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <span className="font-semibold text-sm tracking-tight text-white">NexusRAG</span>
            <span className="text-[10px] uppercase font-bold tracking-wider px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
              Multimodal Agentic
            </span>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-3">
        {/* Model Indicator */}
        <div className="hidden sm:flex items-center gap-2 px-3 py-1 rounded-full bg-slate-900/90 border border-slate-800 text-xs text-slate-300">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          <span className="font-medium text-slate-200">{activeModel || 'Gemini 3.6 Flash'}</span>
        </div>

        {/* Benchmark Evaluation Modal Button */}
        <button
          onClick={onOpenEval}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-indigo-500/10 hover:bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 text-xs font-medium transition-all"
          title="Open RAG Benchmark Evaluation Dashboard"
        >
          <BarChart3 className="w-3.5 h-3.5 text-indigo-400" />
          <span>Evaluation Metrics</span>
        </button>

        {/* Clear Workspace */}
        <button
          onClick={onClearAll}
          className="p-1.5 rounded-lg text-slate-400 hover:text-rose-400 hover:bg-rose-500/10 transition-colors"
          title="Clear all workspace data"
        >
          <Trash2 className="w-4 h-4" />
        </button>

        {/* Marginalia Drawer Toggle */}
        <button
          onClick={onToggleDrawer}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border transition-all ${
            isDrawerOpen
              ? 'bg-indigo-600 text-white border-indigo-500'
              : 'bg-slate-900/90 text-slate-300 border-slate-800 hover:bg-slate-800'
          }`}
        >
          <PanelRight className="w-3.5 h-3.5" />
          <span className="hidden md:inline">Marginalia & Figures</span>
        </button>
      </div>
    </header>
  );
};
