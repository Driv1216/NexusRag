import React, { useState, useRef, useEffect } from 'react';
import {
  Sparkles,
  ArrowUp,
  ImagePlus,
  X,
  Square,
  Lightbulb,
  FileSearch,
  Bot
} from 'lucide-react';
import { ChatMessage, Citation } from '../types';
import { MessageBubble } from './MessageBubble';

interface ChatFeedProps {
  messages: ChatMessage[];
  suggestedQuestions: string[];
  onSendMessage: (text: string, imageB64?: string | null) => Promise<void>;
  onInspectCitation: (citation: Citation) => void;
  onOpenLightbox: (url: string, title: string) => void;
  isStreaming: boolean;
  onStopStreaming: () => void;
}

export const ChatFeed: React.FC<ChatFeedProps> = ({
  messages,
  suggestedQuestions,
  onSendMessage,
  onInspectCitation,
  onOpenLightbox,
  isStreaming,
  onStopStreaming
}) => {
  const [inputText, setInputText] = useState('');
  const [attachedImageB64, setAttachedImageB64] = useState<string | null>(null);
  const [attachedImageName, setAttachedImageName] = useState<string>('');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isStreaming]);

  // Handle textarea resize
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 140)}px`;
    }
  }, [inputText]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const query = inputText.trim();
    if (!query && !attachedImageB64) return;

    onSendMessage(query, attachedImageB64);
    setInputText('');
    setAttachedImageB64(null);
    setAttachedImageName('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      handleSubmit(e);
    }
  };

  const handleImageAttach = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      setAttachedImageName(file.name);
      const reader = new FileReader();
      reader.onload = (event) => {
        setAttachedImageB64(event.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handlePaste = (e: React.ClipboardEvent) => {
    const items = e.clipboardData.items;
    for (let i = 0; i < items.length; i++) {
      if (items[i].type.indexOf('image') !== -1) {
        const file = items[i].getAsFile();
        if (file) {
          setAttachedImageName(file.name || 'Pasted Image');
          const reader = new FileReader();
          reader.onload = (event) => {
            setAttachedImageB64(event.target?.result as string);
          };
          reader.readAsDataURL(file);
          break;
        }
      }
    }
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-[#0a0d16] relative overflow-hidden">
      {/* Scrollable Conversation Feed */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-8">
        <div className="max-w-4xl mx-auto space-y-6">
          {/* Welcome Screen when no messages */}
          {messages.length === 0 ? (
            <div className="py-12 flex flex-col items-center text-center space-y-6 max-w-2xl mx-auto">
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-indigo-500/20 via-sky-500/20 to-purple-500/20 border border-indigo-500/30 flex items-center justify-center text-indigo-300 shadow-xl shadow-indigo-500/10">
                <Sparkles className="w-8 h-8 text-indigo-400 animate-pulse" />
              </div>

              <div>
                <h1 className="text-2xl font-bold tracking-tight text-white sm:text-3xl font-serif">
                  NexusRAG Multimodal Intelligence
                </h1>
                <p className="mt-2 text-sm text-slate-400 leading-relaxed max-w-lg mx-auto">
                  Ask deep cross-modal research questions across your indexed PDFs, technical charts, spreadsheets, audio recordings, and office documents.
                </p>
              </div>

              {/* Suggested Questions */}
              {suggestedQuestions.length > 0 && (
                <div className="w-full pt-4 space-y-2 text-left">
                  <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-400 uppercase tracking-wider px-1">
                    <Lightbulb className="w-3.5 h-3.5 text-amber-400" />
                    <span>Suggested Inquiries From Sources</span>
                  </div>
                  <div className="grid sm:grid-cols-2 gap-2">
                    {suggestedQuestions.map((q, idx) => (
                      <button
                        key={idx}
                        onClick={() => onSendMessage(q)}
                        className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 hover:border-indigo-500/40 hover:bg-slate-900 transition-all text-xs text-slate-300 text-left leading-relaxed flex items-start justify-between group"
                      >
                        <span className="group-hover:text-indigo-200 transition-colors">{q}</span>
                        <ArrowUp className="w-3.5 h-3.5 text-slate-500 group-hover:text-indigo-400 rotate-45 shrink-0 mt-0.5" />
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            messages.map((msg) => (
              <MessageBubble
                key={msg.id}
                message={msg}
                onInspectCitation={onInspectCitation}
                onOpenLightbox={onOpenLightbox}
              />
            ))
          )}
        </div>
      </div>

      {/* Input Composer Panel */}
      <div className="p-4 border-t border-slate-800/80 bg-[#0d131f]/90 backdrop-blur shrink-0">
        <div className="max-w-4xl mx-auto space-y-2">
          {/* Attached Image Preview Card */}
          {attachedImageB64 && (
            <div className="flex items-center gap-3 p-2 bg-slate-900/90 rounded-xl border border-indigo-500/30 w-fit">
              <img
                src={attachedImageB64}
                alt="Attachment preview"
                className="w-12 h-12 object-cover rounded-lg border border-slate-800"
              />
              <div className="pr-2">
                <p className="text-xs font-medium text-slate-200 truncate max-w-[200px]">
                  {attachedImageName}
                </p>
                <span className="text-[10px] text-indigo-400 font-semibold uppercase">
                  Attached to Query
                </span>
              </div>
              <button
                type="button"
                onClick={() => {
                  setAttachedImageB64(null);
                  setAttachedImageName('');
                }}
                className="p-1 rounded-full hover:bg-slate-800 text-slate-400 hover:text-white"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          )}

          {/* Form Composer */}
          <form
            onSubmit={handleSubmit}
            className="flex items-end gap-2 p-2 bg-slate-900/90 rounded-2xl border border-slate-800 focus-within:border-indigo-500/60 focus-within:ring-1 focus-within:ring-indigo-500/40 transition-all shadow-lg"
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".png,.jpg,.jpeg,.webp"
              onChange={handleImageAttach}
              className="hidden"
            />

            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="p-2.5 text-slate-400 hover:text-indigo-400 hover:bg-slate-800/80 rounded-xl transition-all"
              title="Attach diagram or chart to inquiry"
            >
              <ImagePlus className="w-5 h-5" />
            </button>

            <textarea
              ref={textareaRef}
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyDown}
              onPaste={handlePaste}
              placeholder="Inquire of your documents or attach a visual diagram... (⌘ + Enter to send)"
              rows={1}
              className="flex-1 bg-transparent border-0 resize-none text-sm text-slate-100 placeholder-slate-500 focus:ring-0 focus:outline-none max-h-36 py-2 px-1 leading-relaxed"
            />

            {isStreaming ? (
              <button
                type="button"
                onClick={onStopStreaming}
                className="p-2.5 rounded-xl bg-rose-500/20 hover:bg-rose-500/30 text-rose-300 border border-rose-500/40 transition-all flex items-center justify-center shrink-0"
                title="Stop generation"
              >
                <Square className="w-4 h-4 fill-current" />
              </button>
            ) : (
              <button
                type="submit"
                disabled={!inputText.trim() && !attachedImageB64}
                className="p-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:hover:bg-indigo-600 text-white transition-all shadow-md shadow-indigo-500/20 shrink-0"
                title="Send inquiry"
              >
                <ArrowUp className="w-4 h-4" />
              </button>
            )}
          </form>

          <div className="flex items-center justify-between text-[11px] text-slate-500 px-2">
            <span>Grounded in dense vectors + BM25 keyword fusion & cross-modal citations</span>
            <span className="hidden sm:inline">⌘ + Enter to send</span>
          </div>
        </div>
      </div>
    </div>
  );
};
