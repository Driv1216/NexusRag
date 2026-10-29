import React from 'react';
import {
  Bookmark,
  X,
  FileText,
  Image as ImageIcon,
  Table,
  Music,
  ExternalLink,
  Maximize2
} from 'lucide-react';
import { Citation } from '../types';

interface MarginaliaDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  selectedCitation: Citation | null;
  citations: Citation[];
  onOpenLightbox: (url: string, title: string) => void;
}

export const MarginaliaDrawer: React.FC<MarginaliaDrawerProps> = ({
  isOpen,
  onClose,
  selectedCitation,
  citations,
  onOpenLightbox
}) => {
  if (!isOpen) return null;

  const displayList = citations.length > 0 ? citations : (selectedCitation ? [selectedCitation] : []);

  return (
    <aside className="w-88 w-[350px] bg-[#0d131f] border-l border-slate-800/80 flex flex-col shrink-0 h-full overflow-hidden select-none animate-in slide-in-from-right duration-200">
      {/* Header */}
      <div className="p-3.5 border-b border-slate-800/80 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Bookmark className="w-4 h-4 text-indigo-400" />
          <span className="text-xs font-semibold text-slate-200">
            Marginalia & Figures ({displayList.length})
          </span>
        </div>
        <button
          onClick={onClose}
          className="p-1 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition-colors"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Citations List */}
      <div className="flex-1 overflow-y-auto p-3.5 space-y-3">
        {displayList.length === 0 ? (
          <div className="text-center py-12 text-slate-500 text-xs px-4">
            Click on any footnote citation tag (e.g. <code>[Fig. 1 (Pg 2)]</code>) in an AI answer to inspect the exact excerpt and diagram.
          </div>
        ) : (
          displayList.map((c) => {
            const isSelected = selectedCitation?.id === c.id;
            return (
              <div
                key={c.id}
                id={`citation-card-${c.id}`}
                className={`p-3 rounded-xl border transition-all space-y-2.5 ${
                  isSelected
                    ? 'bg-indigo-950/30 border-indigo-500/80 ring-1 ring-indigo-500/50'
                    : 'bg-slate-900/60 border-slate-800/80 hover:border-slate-700'
                }`}
              >
                {/* Header Tag */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-200">
                    {c.type === 'figure' && <ImageIcon className="w-3.5 h-3.5 text-emerald-400" />}
                    {c.type === 'table' && <Table className="w-3.5 h-3.5 text-amber-400" />}
                    {c.type === 'audio' && <Music className="w-3.5 h-3.5 text-sky-400" />}
                    {c.type === 'text' && <FileText className="w-3.5 h-3.5 text-indigo-400" />}
                    <span>
                      {c.type === 'figure' ? `Figure ${c.id}` : c.type === 'table' ? `Table ${c.id}` : `Document ${c.id}`}
                    </span>
                  </div>

                  {c.page && (
                    <span className="text-[10px] bg-slate-800 text-slate-300 px-2 py-0.5 rounded-full font-medium">
                      Page {c.page}
                    </span>
                  )}
                </div>

                {/* Source File */}
                <div className="text-[11px] text-slate-400 truncate flex items-center gap-1">
                  <FileText className="w-3 h-3 text-slate-500 shrink-0" />
                  <span className="truncate">{c.file}</span>
                </div>

                {/* Image Preview if figure */}
                {c.image_url && (
                  <div
                    onClick={() => onOpenLightbox(c.image_url!, `${c.file} (Page ${c.page || 1})`)}
                    className="relative rounded-lg overflow-hidden border border-slate-800 bg-slate-950 aspect-video cursor-pointer group"
                  >
                    <img
                      src={c.image_url}
                      alt={c.file}
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                    />
                    <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center text-white text-xs gap-1">
                      <Maximize2 className="w-3.5 h-3.5" />
                      <span>Zoom</span>
                    </div>
                  </div>
                )}

                {/* Snippet Excerpt */}
                <div className="text-xs text-slate-300 bg-slate-950/60 p-2.5 rounded-lg border border-slate-800/60 leading-relaxed font-serif italic">
                  "{c.snippet}"
                </div>
              </div>
            );
          })
        )}
      </div>
    </aside>
  );
};
