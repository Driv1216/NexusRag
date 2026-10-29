import React, { useState } from 'react';
import {
  X,
  BarChart3,
  CheckCircle2,
  ShieldCheck,
  Target,
  Search,
  BookOpen,
  Play,
  RotateCw,
  Award
} from 'lucide-react';
import { EvalSummary } from '../types';

interface EvaluationModalProps {
  isOpen: boolean;
  onClose: () => void;
  evalSummary: EvalSummary | null;
  onRunEvaluation: () => Promise<void>;
  isRunningEval: boolean;
}

export const EvaluationModal: React.FC<EvaluationModalProps> = ({
  isOpen,
  onClose,
  evalSummary,
  onRunEvaluation,
  isRunningEval
}) => {
  if (!isOpen) return null;

  const getScoreColor = (score: number) => {
    if (score >= 0.9) return 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10';
    if (score >= 0.75) return 'text-amber-400 border-amber-500/30 bg-amber-500/10';
    return 'text-rose-400 border-rose-500/30 bg-rose-500/10';
  };

  const getBarColor = (score: number) => {
    if (score >= 0.9) return 'bg-emerald-400';
    if (score >= 0.75) return 'bg-amber-400';
    return 'bg-rose-400';
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm animate-in fade-in duration-150">
      <div className="bg-[#0f1524] border border-slate-800 w-full max-w-3xl rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[85vh]">
        {/* Header */}
        <div className="p-5 border-b border-slate-800 flex items-center justify-between bg-slate-900/60">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-indigo-500/10 border border-indigo-500/30 flex items-center justify-center text-indigo-400">
              <Award className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-semibold text-white">RAG Evaluation Benchmark</h2>
              <p className="text-xs text-slate-400">
                Impartial LLM Judge academic scoring (Faithfulness, Relevancy, Precision, Recall)
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={onRunEvaluation}
              disabled={isRunningEval}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white text-xs font-medium transition-all shadow-md shadow-indigo-500/20"
            >
              {isRunningEval ? (
                <>
                  <RotateCw className="w-3.5 h-3.5 animate-spin" />
                  <span>Evaluating...</span>
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5 fill-current" />
                  <span>Run Benchmark</span>
                </>
              )}
            </button>

            <button
              onClick={onClose}
              className="p-1.5 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-white transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Modal Body */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {evalSummary ? (
            <>
              {/* Top Overall Score Card */}
              <div className="grid grid-cols-1 sm:grid-cols-5 gap-3 p-4 rounded-xl bg-slate-900/80 border border-slate-800">
                <div className="sm:col-span-2 flex flex-col justify-center items-center text-center p-3 border-b sm:border-b-0 sm:border-r border-slate-800">
                  <span className="text-xs text-slate-400 font-medium">Overall Composite Score</span>
                  <div className="text-3xl font-bold text-indigo-300 my-1 font-mono">
                    {(evalSummary.overall_rag_score * 100).toFixed(1)}%
                  </div>
                  <span className="text-[11px] text-emerald-400 font-semibold flex items-center gap-1">
                    <CheckCircle2 className="w-3 h-3" />
                    Production Ready
                  </span>
                </div>

                <div className="sm:col-span-3 grid grid-cols-2 gap-3 pl-0 sm:pl-2">
                  {/* Faithfulness */}
                  <div className="space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-slate-400 flex items-center gap-1">
                        <ShieldCheck className="w-3 h-3 text-emerald-400" /> Faithfulness
                      </span>
                      <span className="font-mono text-slate-200 font-semibold">
                        {(evalSummary.average_faithfulness * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
                      <div
                        className={`h-full ${getBarColor(evalSummary.average_faithfulness)} transition-all duration-500`}
                        style={{ width: `${evalSummary.average_faithfulness * 100}%` }}
                      />
                    </div>
                  </div>

                  {/* Relevancy */}
                  <div className="space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-slate-400 flex items-center gap-1">
                        <Target className="w-3 h-3 text-sky-400" /> Relevancy
                      </span>
                      <span className="font-mono text-slate-200 font-semibold">
                        {(evalSummary.average_relevancy * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
                      <div
                        className={`h-full ${getBarColor(evalSummary.average_relevancy)} transition-all duration-500`}
                        style={{ width: `${evalSummary.average_relevancy * 100}%` }}
                      />
                    </div>
                  </div>

                  {/* Context Precision */}
                  <div className="space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-slate-400 flex items-center gap-1">
                        <Search className="w-3 h-3 text-amber-400" /> Precision
                      </span>
                      <span className="font-mono text-slate-200 font-semibold">
                        {(evalSummary.average_context_precision * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
                      <div
                        className={`h-full ${getBarColor(evalSummary.average_context_precision)} transition-all duration-500`}
                        style={{ width: `${evalSummary.average_context_precision * 100}%` }}
                      />
                    </div>
                  </div>

                  {/* Context Recall */}
                  <div className="space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-slate-400 flex items-center gap-1">
                        <BookOpen className="w-3 h-3 text-purple-400" /> Recall
                      </span>
                      <span className="font-mono text-slate-200 font-semibold">
                        {(evalSummary.average_context_recall * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
                      <div
                        className={`h-full ${getBarColor(evalSummary.average_context_recall)} transition-all duration-500`}
                        style={{ width: `${evalSummary.average_context_recall * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Per Question Details */}
              <div className="space-y-3">
                <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                  Evaluated Inquiries & Judge Critique
                </h3>

                {evalSummary.results.map((r, idx) => (
                  <div
                    key={idx}
                    className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2.5"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <p className="text-xs font-medium text-slate-200">{r.question}</p>
                      <span className={`text-[10px] px-2 py-0.5 rounded-full border font-mono font-semibold ${getScoreColor(r.faithfulness_score)}`}>
                        {(r.faithfulness_score * 100).toFixed(0)}% Faith
                      </span>
                    </div>

                    <div className="text-xs text-slate-400 bg-slate-950 p-2.5 rounded-lg border border-slate-800/80 leading-relaxed font-serif">
                      "{r.critique}"
                    </div>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <div className="text-center py-12 text-slate-500 text-sm">
              Click "Run Benchmark" above to compute RAG quality scores on your indexed collection.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
