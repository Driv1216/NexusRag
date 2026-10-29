import React, { useState, useRef } from 'react';
import {
  FileText,
  Image as ImageIcon,
  Table,
  Music,
  Video,
  UploadCloud,
  Layers,
  Sparkles,
  FilePlus,
  Trash2,
  ExternalLink,
  ChevronRight
} from 'lucide-react';
import { DocumentInfo, GalleryItem } from '../types';

interface SidebarProps {
  documents: DocumentInfo[];
  gallery: GalleryItem[];
  onUploadFile: (file: File) => Promise<void>;
  onOpenLightbox: (url: string, title: string) => void;
  isUploading: boolean;
  uploadStatusText: string;
}

export const Sidebar: React.FC<SidebarProps> = ({
  documents,
  gallery,
  onUploadFile,
  onOpenLightbox,
  isUploading,
  uploadStatusText
}) => {
  const [activeTab, setActiveTab] = useState<'docs' | 'gallery'>('docs');
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onUploadFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onUploadFile(e.target.files[0]);
    }
  };

  const getDocIcon = (doc: DocumentInfo) => {
    if (doc.is_pdf) return <FileText className="w-4 h-4 text-rose-400" />;
    if (doc.is_image) return <ImageIcon className="w-4 h-4 text-emerald-400" />;
    if (doc.is_sheet) return <Table className="w-4 h-4 text-amber-400" />;
    if (doc.is_audio) return <Music className="w-4 h-4 text-sky-400" />;
    if (doc.is_video) return <Video className="w-4 h-4 text-purple-400" />;
    return <FileText className="w-4 h-4 text-indigo-400" />;
  };

  const getDocBadge = (doc: DocumentInfo) => {
    if (doc.is_pdf) return <span className="text-[10px] bg-rose-500/10 text-rose-300 px-1.5 py-0.5 rounded border border-rose-500/20">PDF</span>;
    if (doc.is_image) return <span className="text-[10px] bg-emerald-500/10 text-emerald-300 px-1.5 py-0.5 rounded border border-emerald-500/20">IMG</span>;
    if (doc.is_sheet) return <span className="text-[10px] bg-amber-500/10 text-amber-300 px-1.5 py-0.5 rounded border border-amber-500/20">TABLE</span>;
    if (doc.is_audio) return <span className="text-[10px] bg-sky-500/10 text-sky-300 px-1.5 py-0.5 rounded border border-sky-500/20">AUDIO</span>;
    if (doc.is_video) return <span className="text-[10px] bg-purple-500/10 text-purple-300 px-1.5 py-0.5 rounded border border-purple-500/20">VIDEO</span>;
    return <span className="text-[10px] bg-indigo-500/10 text-indigo-300 px-1.5 py-0.5 rounded border border-indigo-500/20">DOC</span>;
  };

  return (
    <aside className="w-80 bg-[#0d131f] border-r border-slate-800/80 flex flex-col shrink-0 h-full overflow-hidden select-none">
      {/* Segmented Tab Controls */}
      <div className="p-3 border-b border-slate-800/80">
        <div className="grid grid-cols-2 gap-1 p-1 bg-slate-900/90 rounded-lg border border-slate-800">
          <button
            onClick={() => setActiveTab('docs')}
            className={`flex items-center justify-center gap-1.5 py-1.5 text-xs font-medium rounded-md transition-all ${
              activeTab === 'docs'
                ? 'bg-indigo-600 text-white shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <FileText className="w-3.5 h-3.5" />
            <span>Sources ({documents.length})</span>
          </button>

          <button
            onClick={() => setActiveTab('gallery')}
            className={`flex items-center justify-center gap-1.5 py-1.5 text-xs font-medium rounded-md transition-all ${
              activeTab === 'gallery'
                ? 'bg-indigo-600 text-white shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <ImageIcon className="w-3.5 h-3.5" />
            <span>Gallery ({gallery.length})</span>
          </button>
        </div>
      </div>

      {/* Main Tab Panes */}
      <div className="flex-1 overflow-y-auto p-3 space-y-4">
        {activeTab === 'docs' ? (
          <>
            {/* Drag & Drop Upload Zone */}
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`p-4 rounded-xl border border-dashed transition-all cursor-pointer text-center ${
                isDragOver
                  ? 'border-indigo-400 bg-indigo-500/10'
                  : 'border-slate-800 bg-slate-900/40 hover:border-slate-700 hover:bg-slate-900/70'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.png,.jpg,.jpeg,.webp,.docx,.pptx,.csv,.xlsx,.mp3,.wav,.m4a,.mp4"
                onChange={handleFileChange}
                className="hidden"
              />
              <div className="w-10 h-10 mx-auto mb-2 rounded-full bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400">
                <UploadCloud className="w-5 h-5" />
              </div>
              <p className="text-xs font-medium text-slate-200">
                Drop PDF, DOCX, CSV, Image, Audio, Video
              </p>
              <p className="text-[11px] text-slate-400 mt-1">
                Extracts text, charts, tables & figures
              </p>
            </div>

            {/* Upload Status Feedback */}
            {isUploading && (
              <div className="p-3 rounded-lg bg-indigo-500/10 border border-indigo-500/30 flex items-center gap-2.5">
                <div className="w-2.5 h-2.5 rounded-full bg-indigo-400 animate-ping shrink-0" />
                <span className="text-xs text-indigo-200 font-medium truncate">
                  {uploadStatusText || 'Ingesting & extracting visual diagrams...'}
                </span>
              </div>
            )}

            {/* Indexed Document List */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between px-1 mb-1">
                <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                  Indexed Sources
                </span>
                <span className="text-[11px] text-slate-500">{documents.length} Total</span>
              </div>

              {documents.length === 0 ? (
                <div className="text-center py-8 text-slate-500 text-xs">
                  No documents indexed yet. Upload above to begin.
                </div>
              ) : (
                documents.map((doc, idx) => (
                  <div
                    key={idx}
                    className="p-2.5 rounded-lg bg-slate-900/60 border border-slate-800/80 hover:border-slate-700 transition-all flex items-start gap-2.5 group"
                  >
                    <div className="p-1.5 rounded-md bg-slate-800/80 shrink-0 mt-0.5">
                      {getDocIcon(doc)}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-xs font-medium text-slate-200 truncate group-hover:text-indigo-300 transition-colors">
                        {doc.filename}
                      </p>
                      <div className="flex items-center gap-2 mt-1">
                        {getDocBadge(doc)}
                        <span className="text-[10px] text-slate-400">{doc.size_mb} MB</span>
                        {doc.figures_count > 0 && (
                          <span className="text-[10px] text-emerald-400">
                            • {doc.figures_count} fig{doc.figures_count > 1 ? 's' : ''}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </>
        ) : (
          /* Visual Gallery Tab */
          <div className="space-y-2">
            <div className="flex items-center justify-between px-1 mb-1">
              <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                Visual Figures & Charts
              </span>
              <span className="text-[11px] text-slate-500">{gallery.length} Extracted</span>
            </div>

            {gallery.length === 0 ? (
              <div className="text-center py-8 text-slate-500 text-xs">
                No figures extracted yet. Upload documents containing diagrams.
              </div>
            ) : (
              <div className="grid grid-cols-2 gap-2">
                {gallery.map((fig, idx) => (
                  <div
                    key={idx}
                    onClick={() => onOpenLightbox(fig.url, fig.filename)}
                    className="group relative rounded-lg border border-slate-800 overflow-hidden bg-slate-950 aspect-video cursor-pointer hover:border-indigo-500/50 transition-all"
                  >
                    <img
                      src={fig.url}
                      alt={fig.filename}
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                      loading="lazy"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex items-end p-1.5">
                      <span className="text-[10px] text-slate-200 truncate">
                        {fig.size_kb} KB
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Footer Info */}
      <div className="p-3 border-t border-slate-800/80 bg-slate-950/40 text-[11px] text-slate-400 flex items-center justify-between">
        <span className="flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
          <span>ChromaDB Vector Store</span>
        </span>
        <span className="text-slate-500 font-mono text-[10px]">v3.0.0</span>
      </div>
    </aside>
  );
};
