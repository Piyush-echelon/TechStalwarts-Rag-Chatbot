import React, { useState } from 'react';
import { BookOpen, ChevronDown, ChevronUp, FileText } from 'lucide-react';

/**
 * SourcePanel — collapsible list of source citations beneath an AI message.
 *
 * Props:
 *   sources: Array<{ page_number, filename, content_preview }>
 */
export default function SourcePanel({ sources }) {
  const [open, setOpen] = useState(false);

  if (!sources || sources.length === 0) return null;

  return (
    <div className="source-panel">
      <button
        className="source-toggle"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-controls="source-list"
        id="source-toggle-btn"
      >
        <BookOpen size={13} />
        {sources.length} source{sources.length > 1 ? 's' : ''} cited
        {open ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
      </button>

      {open && (
        <div className="source-list" id="source-list" role="list">
          {sources.map((src, i) => (
            <div key={i} className="source-item" role="listitem">
              <div className="source-item-header">
                <FileText size={12} />
                <span>{src.filename}</span>
                <span style={{ marginLeft: 'auto', color: 'var(--text-muted)', fontWeight: 400 }}>
                  Page {src.page_number}
                </span>
              </div>
              {src.content_preview && (
                <p className="source-item-preview">{src.content_preview}</p>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
