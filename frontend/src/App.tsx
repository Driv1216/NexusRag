import React, { useState, useEffect, useRef } from 'react';
import { Navbar } from './components/Navbar';
import { Sidebar } from './components/Sidebar';
import { ChatFeed } from './components/ChatFeed';
import { MarginaliaDrawer } from './components/MarginaliaDrawer';
import { EvaluationModal } from './components/EvaluationModal';
import { LightboxModal } from './components/LightboxModal';
import {
  DocumentInfo,
  GalleryItem,
  ChatMessage,
  Citation,
  EvalSummary
} from './types';

export const App: React.FC = () => {
  // State
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [gallery, setGallery] = useState<GalleryItem[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [suggestedQuestions, setSuggestedQuestions] = useState<string[]>([
    "Summarize the primary conclusions across the indexed sources.",
    "What visual diagrams, charts, or algorithms are described?",
    "Compare the quantitative benchmark metrics reported."
  ]);
  const [activeModel, setActiveModel] = useState<string>('Gemini 3.6 Flash');

  // UI state
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const [selectedCitation, setSelectedCitation] = useState<Citation | null>(null);
  const [activeAnswerCitations, setActiveAnswerCitations] = useState<Citation[]>([]);

  // Modals
  const [isEvalOpen, setIsEvalOpen] = useState(false);
  const [evalSummary, setEvalSummary] = useState<EvalSummary | null>(null);
  const [isRunningEval, setIsRunningEval] = useState(false);

  const [lightboxUrl, setLightboxUrl] = useState<string>('');
  const [lightboxTitle, setLightboxTitle] = useState<string>('');
  const [isLightboxOpen, setIsLightboxOpen] = useState(false);

  // Upload state
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatusText, setUploadStatusText] = useState('');

  // Streaming state
  const [isStreaming, setIsStreaming] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Fetch initial data
  useEffect(() => {
    fetchHealth();
    fetchDocuments();
    fetchGallery();
    fetchEvalResults();
  }, []);

  const fetchHealth = async () => {
    try {
      const res = await fetch('/api/health');
      if (res.ok) {
        const data = await res.json();
        setActiveModel(data.active_model || 'Gemini 3.6 Flash');
      }
    } catch (err) {
      console.error('Health check error:', err);
    }
  };

  const fetchDocuments = async () => {
    try {
      const res = await fetch('/api/documents');
      if (res.ok) {
        const data = await res.json();
        setDocuments(data.documents || []);
      }
    } catch (err) {
      console.error('Fetch documents error:', err);
    }
  };

  const fetchGallery = async () => {
    try {
      const res = await fetch('/api/gallery');
      if (res.ok) {
        const data = await res.json();
        setGallery(data.gallery || []);
      }
    } catch (err) {
      console.error('Fetch gallery error:', err);
    }
  };

  const fetchEvalResults = async () => {
    try {
      const res = await fetch('/api/evaluate/results');
      if (res.ok) {
        const data = await res.json();
        setEvalSummary(data);
      }
    } catch (err) {
      console.error('Fetch eval error:', err);
    }
  };

  // Upload handler
  const handleUploadFile = async (file: File) => {
    setIsUploading(true);
    setUploadStatusText(`Ingesting & extracting multimodal content from "${file.name}"...`);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });

      const data = await res.json();
      if (res.ok) {
        setUploadStatusText('Ingestion complete!');
        setTimeout(() => {
          setIsUploading(false);
          setUploadStatusText('');
        }, 1200);

        fetchDocuments();
        fetchGallery();

        if (data.document && data.document.suggested_questions) {
          setSuggestedQuestions(data.document.suggested_questions);
        }
      } else {
        alert(`Upload error: ${data.detail || 'Failed to process file'}`);
        setIsUploading(false);
      }
    } catch (err) {
      console.error('Upload network error:', err);
      alert('Network error while uploading file.');
      setIsUploading(false);
    }
  };

  // SSE Streaming Send Message Handler
  const handleSendMessage = async (text: string, imageB64?: string | null) => {
    if (!text.trim() && !imageB64) return;

    const userMsgId = `user_${Date.now()}`;
    const aiMsgId = `ai_${Date.now()}`;

    // Append User Message
    const userMsg: ChatMessage = {
      id: userMsgId,
      role: 'user',
      content: text,
      imageB64: imageB64,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    // AI Placeholder
    const initialAiMsg: ChatMessage = {
      id: aiMsgId,
      role: 'assistant',
      content: '',
      isStreaming: true,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages((prev) => [...prev, userMsg, initialAiMsg]);
    setIsStreaming(true);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const response = await fetch('/api/query/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: text,
          session_id: 'default_session',
          top_k: 4,
          query_image_b64: imageB64,
          enable_agent: true
        }),
        signal: controller.signal
      });

      if (!response.ok || !response.body) {
        throw new Error('Failed to start streaming query');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data:')) {
            const jsonStr = line.replace('data:', '').trim();
            if (!jsonStr) continue;

            try {
              const eventData = JSON.parse(jsonStr);

              if (eventData.type === 'route') {
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === aiMsgId ? { ...m, route: eventData.route } : m
                  )
                );
              } else if (eventData.type === 'token') {
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === aiMsgId
                      ? { ...m, content: m.content + eventData.content }
                      : m
                  )
                );
              } else if (eventData.type === 'done') {
                const citations = eventData.citations || [];
                setActiveAnswerCitations(citations);
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === aiMsgId
                      ? { ...m, isStreaming: false, citations: citations }
                      : m
                  )
                );
              } else if (eventData.type === 'error') {
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === aiMsgId
                      ? {
                          ...m,
                          isStreaming: false,
                          content: m.content + `\n\n*Error: ${eventData.content}*`
                        }
                      : m
                  )
                );
              }
            } catch (pErr) {
              console.error('SSE parsing error:', pErr);
            }
          }
        }
      }
    } catch (err: any) {
      if (err.name === 'AbortError') {
        console.log('Stream aborted by user');
      } else {
        console.error('Stream error:', err);
        setMessages((prev) =>
          prev.map((m) =>
            m.id === aiMsgId
              ? {
                  ...m,
                  isStreaming: false,
                  content: 'Error: Unable to complete synthesis stream.'
                }
              : m
          )
        );
      }
    } finally {
      setIsStreaming(false);
      abortControllerRef.current = null;
      setMessages((prev) =>
        prev.map((m) => (m.id === aiMsgId ? { ...m, isStreaming: false } : m))
      );
    }
  };

  const handleStopStreaming = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    setIsStreaming(false);
  };

  // Inspect Citation
  const handleInspectCitation = (citation: Citation) => {
    setSelectedCitation(citation);
    setIsDrawerOpen(true);
    const cardEl = document.getElementById(`citation-card-${citation.id}`);
    if (cardEl) {
      cardEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  };

  // Lightbox
  const handleOpenLightbox = (url: string, title: string) => {
    setLightboxUrl(url);
    setLightboxTitle(title);
    setIsLightboxOpen(true);
  };

  // Evaluation
  const handleRunEvaluation = async () => {
    setIsRunningEval(true);
    try {
      const res = await fetch('/api/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ top_k: 4 })
      });
      if (res.ok) {
        const data = await res.json();
        setEvalSummary(data);
      }
    } catch (err) {
      console.error('Run eval error:', err);
    } finally {
      setIsRunningEval(false);
    }
  };

  // Clear workspace
  const handleClearAll = async () => {
    if (confirm('Clear all indexed documents, vector indices, and extracted figures?')) {
      try {
        await fetch('/api/clear', { method: 'DELETE' });
        setDocuments([]);
        setGallery([]);
        setMessages([]);
        setActiveAnswerCitations([]);
        setSelectedCitation(null);
      } catch (err) {
        console.error('Clear error:', err);
      }
    }
  };

  return (
    <div className="flex flex-col h-screen w-screen bg-[#0a0d16] text-slate-100 overflow-hidden font-sans">
      {/* Top Navigation */}
      <Navbar
        activeModel={activeModel}
        onOpenEval={() => setIsEvalOpen(true)}
        onToggleDrawer={() => setIsDrawerOpen(!isDrawerOpen)}
        onClearAll={handleClearAll}
        isDrawerOpen={isDrawerOpen}
      />

      {/* Main 3-Panel Layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Sidebar */}
        <Sidebar
          documents={documents}
          gallery={gallery}
          onUploadFile={handleUploadFile}
          onOpenLightbox={handleOpenLightbox}
          isUploading={isUploading}
          uploadStatusText={uploadStatusText}
        />

        {/* Center Conversation Canvas */}
        <ChatFeed
          messages={messages}
          suggestedQuestions={suggestedQuestions}
          onSendMessage={handleSendMessage}
          onInspectCitation={handleInspectCitation}
          onOpenLightbox={handleOpenLightbox}
          isStreaming={isStreaming}
          onStopStreaming={handleStopStreaming}
        />

        {/* Right Marginalia & Figure Inspector */}
        <MarginaliaDrawer
          isOpen={isDrawerOpen}
          onClose={() => setIsDrawerOpen(false)}
          selectedCitation={selectedCitation}
          citations={activeAnswerCitations}
          onOpenLightbox={handleOpenLightbox}
        />
      </div>

      {/* Evaluation Dashboard Modal */}
      <EvaluationModal
        isOpen={isEvalOpen}
        onClose={() => setIsEvalOpen(false)}
        evalSummary={evalSummary}
        onRunEvaluation={handleRunEvaluation}
        isRunningEval={isRunningEval}
      />

      {/* Fullscreen Lightbox Modal */}
      <LightboxModal
        isOpen={isLightboxOpen}
        onClose={() => setIsLightboxOpen(false)}
        imageUrl={lightboxUrl}
        title={lightboxTitle}
      />
    </div>
  );
};

export default App;
