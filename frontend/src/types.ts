export interface Citation {
  id: number;
  chunk_id: string;
  type: 'text' | 'figure' | 'table' | 'audio' | 'video';
  file: string;
  page?: number;
  timestamp?: string;
  snippet: string;
  image_url?: string | null;
  score?: number;
}

export interface DocumentInfo {
  filename: string;
  size_mb: number;
  path: string;
  is_pdf: boolean;
  is_image: boolean;
  is_sheet: boolean;
  is_audio: boolean;
  is_video: boolean;
  figures_count: number;
}

export interface GalleryItem {
  filename: string;
  url: string;
  size_kb: number;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  imageB64?: string | null;
  citations?: Citation[];
  route?: string;
  isStreaming?: boolean;
  timestamp: string;
}

export interface EvalResultItem {
  question: string;
  ground_truth?: string;
  retrieved_contexts: string[];
  answer: string;
  faithfulness_score: number;
  relevancy_score: number;
  context_precision: number;
  context_recall: number;
  critique: string;
}

export interface EvalSummary {
  total_evaluated: number;
  average_faithfulness: number;
  average_relevancy: number;
  average_context_precision: number;
  average_context_recall: number;
  overall_rag_score: number;
  results: EvalResultItem[];
}

export interface SessionInfo {
  session_id: string;
  title: string;
  created_at: string;
  message_count: number;
}
