import React from 'react';
import { Sparkles, FileText, Layers, RotateCcw } from 'lucide-react';
import PDFUploader from './PDFUploader.jsx';

/**
 * Sidebar — branding, PDF upload, document info, and session controls.
 *
 * Props:
 *   session:        object|null   — { session_id, filename, num_chunks, num_pages }
 *   uploading:      bool
 *   uploadProgress: number
 *   uploadError:    string|null
 *   onUpload(file): void
 *   onNewChat():    void
 */
export default function Sidebar({
  session,
  uploading,
  uploadProgress,
  uploadError,
  onUpload,
  onNewChat,
}) {
  return (
    <aside className="sidebar" aria-label="Sidebar">
      {/* Logo */}
      <div className="sidebar-header">
        <div className="sidebar-logo">
          <div className="logo-icon" aria-hidden="true">
            <Sparkles size={16} color="#fff" />
          </div>
          <h1>RAG Chatbot</h1>
        </div>
        <p className="sidebar-subtitle">AI Document Assistant</p>
      </div>

      {/* Content */}
      <div className="sidebar-content">

        {/* Section label */}
        <p style={{ fontSize: '0.7rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
          Document
        </p>

        {/* Upload zone — always shown */}
        <PDFUploader
          onFile={onUpload}
          uploading={uploading}
          uploadProgress={uploadProgress}
          error={uploadError}
        />

        {/* Document info card — shown once a PDF is loaded */}
        {session && (
          <div className="doc-info-card" aria-label="Document information">
            <div className="doc-info-header">
              <div className="doc-info-icon" aria-hidden="true">
                <FileText size={18} />
              </div>
              <div>
                <p className="doc-info-name">{session.filename}</p>
                <div className="doc-info-meta">
                  <span className="doc-info-badge success">Indexed</span>
                  <span className="doc-info-badge">{session.num_pages} page{session.num_pages !== 1 ? 's' : ''}</span>
                </div>
              </div>
            </div>

            <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
              <div className="doc-info-badge" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <Layers size={10} />
                {session.num_chunks} chunks
              </div>
              <div className="doc-info-badge" style={{ fontFamily: 'monospace', fontSize: '0.65rem' }}>
                {session.session_id.slice(0, 8)}…
              </div>
            </div>
          </div>
        )}

        {/* New chat / clear session */}
        {session && (
          <button
            id="new-chat-btn"
            className="btn-new-chat"
            onClick={onNewChat}
            aria-label="Start a new chat session"
          >
            <RotateCcw size={14} />
            New chat
          </button>
        )}

        {/* How it works */}
        <div style={{ marginTop: 'auto' }}>
          <p style={{ fontSize: '0.7rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '10px' }}>
            How it works
          </p>
          {[
            ['1', 'Upload a PDF document'],
            ['2', 'Chunks are embedded into a vector store'],
            ['3', 'Ask questions in natural language'],
            ['4', 'Answers cite exact page sources'],
          ].map(([n, text]) => (
            <div key={n} style={{ display: 'flex', gap: '8px', marginBottom: '8px', alignItems: 'flex-start' }}>
              <span style={{
                width: '18px', height: '18px', borderRadius: '50%',
                background: 'var(--bg-elevated)', border: '1px solid var(--border)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: '0.62rem', fontWeight: 700, color: 'var(--text-accent)',
                flexShrink: 0, marginTop: '1px',
              }}>{n}</span>
              <span style={{ fontSize: '0.76rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>{text}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="sidebar-footer">
        Built with LangGraph + FastAPI + React
      </div>
    </aside>
  );
}
