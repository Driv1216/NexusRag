import React, { useMemo } from 'react';
import { marked } from 'marked';
import { Sparkles, User, FileText, Image as ImageIcon, Table, Music, CheckCircle2 } from 'lucide-react';
import { ChatMessage, Citation } from '../types';

interface MessageBubbleProps {
  message: ChatMessage;
  onInspectCitation: (citation: Citation) => void;
  onOpenLightbox: (url: string, title: string) => void;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({
  message,
  onInspectCitation,
  onOpenLightbox
}) => {
  const isUser = message.role === 'user';

  // Process markdown and transform citation patterns into interactive buttons
  const renderedContent = useMemo(() => {
    if (isUser) return message.content;

    let raw = message.content || '';
    let parsedHtml = marked.parse(raw) as string;

    // Pattern matches: [Doc X (Pg Y)], [Fig. X (Pg Y)], [Figure X (Pg Y)], [Table X (Pg Y)], [Audio X]
    parsedHtml = parsedHtml.replace(
      /\[(?:(Fig\.?|Figure|Doc|Citation|Table|Audio)\s*(\d+)(?:,\s*Pg?\s*(\d+)|\s*\(Pg\.?\s*(\d+)\))?|Page\s*(\d+))\]/gi,
      (match, prefix, idStr, pg1, pg2, pg3) => {
        const id = parseInt(idStr || '1', 10);
        const page = pg1 || pg2 || pg3 || '';
        const pref = (prefix || '').toLowerCase();

        let iconType = 'doc';
        let pillClass = 'citation-pill';
        let label = `Doc ${id}`;

        if (pref.includes('fig')) {
          iconType = 'fig';
          pillClass = 'citation-pill figure-pill';
          label = page ? `Fig. ${id} (Pg ${page})` : `Figure ${id}`;
        } else if (pref.includes('table')) {
          iconType = 'table';
          pillClass = 'citation-pill table-pill';
          label = page ? `Table ${id} (Pg ${page})` : `Table ${id}`;
        } else if (pref.includes('audio')) {
          iconType = 'audio';
          pillClass = 'citation-pill audio-pill';
          label = `Audio ${id}`;
        } else {
          label = page ? `Doc ${id} (Pg ${page})` : `Citation ${id}`;
        }

        return `<button class="${pillClass}" data-citation-id="${id}"><span>${label}</span></button>`;
      }
    );

    return parsedHtml;
  }, [message.content, isUser]);

  const handleContainerClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const target = (e.target as HTMLElement).closest('button[data-citation-id]');
    if (target) {
      const citeId = parseInt(target.getAttribute('data-citation-id') || '1', 10);
      const matched = message.citations?.find(c => c.id === citeId);
      if (matched) {
        onInspectCitation(matched);
      } else if (message.citations && message.citations.length > 0) {
        onInspectCitation(message.citations[0]);
      }
    }
  };

  return (
    <div className={`flex gap-3.5 max-w-4xl w-full ${isUser ? 'ml-auto flex-row-reverse' : ''}`}>
      {/* Avatar */}
      <div
        className={`w-8 h-8 rounded-xl flex items-center justify-center shrink-0 shadow-sm ${
          isUser
            ? 'bg-gradient-to-br from-indigo-500 to-indigo-700 text-white'
            : 'bg-gradient-to-br from-indigo-500/20 to-sky-500/20 border border-indigo-500/30 text-indigo-300'
        }`}
      >
        {isUser ? <User className="w-4 h-4" /> : <Sparkles className="w-4 h-4 text-indigo-400" />}
      </div>

      {/* Message Content Container */}
      <div className={`flex-1 min-w-0 ${isUser ? 'text-right' : ''}`}>
        {/* Route Badge */}
        {!isUser && message.route && (
          <div className="flex items-center gap-2 mb-1.5">
            <span className="text-[10px] font-semibold tracking-wider uppercase px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-300 border border-indigo-500/20">
              Route: {message.route}
            </span>
          </div>
        )}

        {/* User image attachment */}
        {isUser && message.imageB64 && (
          <div className="inline-block mb-2 rounded-xl overflow-hidden border border-indigo-500/30 bg-slate-900 aspect-video max-h-48 cursor-pointer">
            <img
              src={message.imageB64}
              alt="User attachment"
              className="w-full h-full object-cover"
              onClick={() => onOpenLightbox(message.imageB64!, 'Query Attachment')}
            />
          </div>
        )}

        {/* Bubble */}
        <div
          onClick={handleContainerClick}
          className={`p-4 rounded-2xl text-left ${
            isUser
              ? 'bg-indigo-600 text-white inline-block shadow-md rounded-tr-sm'
              : 'bg-slate-900/80 border border-slate-800/90 text-slate-200 shadow-sm rounded-tl-sm prose-rag'
          }`}
        >
          {isUser ? (
            <p className="text-[15px] leading-relaxed whitespace-pre-wrap">{message.content}</p>
          ) : (
            <div dangerouslySetInnerHTML={{ __html: renderedContent }} />
          )}

          {/* Streaming cursor */}
          {message.isStreaming && (
            <span className="inline-block w-2 h-4 bg-indigo-400 animate-pulse ml-1 align-middle" />
          )}
        </div>

        {/* Citations Footer Tray (for AI answers) */}
        {!isUser && message.citations && message.citations.length > 0 && !message.isStreaming && (
          <div className="mt-2.5 flex flex-wrap items-center gap-1.5">
            <span className="text-[11px] text-slate-400 mr-1 font-medium">Grounded Citations:</span>
            {message.citations.map((c) => (
              <button
                key={c.id}
                onClick={() => onInspectCitation(c)}
                className={`citation-pill ${
                  c.type === 'figure'
                    ? 'figure-pill'
                    : c.type === 'table'
                    ? 'table-pill'
                    : c.type === 'audio'
                    ? 'audio-pill'
                    : ''
                }`}
              >
                {c.type === 'figure' && <ImageIcon className="w-3 h-3" />}
                {c.type === 'table' && <Table className="w-3 h-3" />}
                {c.type === 'audio' && <Music className="w-3 h-3" />}
                {c.type === 'text' && <FileText className="w-3 h-3" />}
                <span>
                  {c.type === 'figure' ? `Fig. ${c.id}` : c.type === 'table' ? `Table ${c.id}` : `Doc ${c.id}`}
                  {c.page ? ` (Pg ${c.page})` : ''}
                </span>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
