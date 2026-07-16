import React from 'react';
import { MessageSquare } from 'lucide-react';

const SUGGESTIONS = [
  'Summarise this document',
  'What are the key findings?',
  'List the main topics covered',
  'What conclusions are drawn?',
];

/**
 * EmptyState — shown in the chat area before the user sends a message.
 *
 * Props:
 *   sessionReady: bool  — true if a PDF has been uploaded
 *   onSuggestion(text): called when a suggestion chip is clicked
 */
export default function EmptyState({ sessionReady, onSuggestion }) {
  return (
    <div className="empty-state" role="main" aria-label="Chat welcome screen">
      <div className="empty-state-icon" aria-hidden="true">
        <MessageSquare size={32} />
      </div>

      {sessionReady ? (
        <>
          <h2>Ready to answer your questions</h2>
          <p>
            Your document has been indexed. Ask anything about its content and
            I'll retrieve the most relevant context and cite my sources.
          </p>
          <div className="empty-suggestions" role="list" aria-label="Suggested questions">
            {SUGGESTIONS.map((s) => (
              <button
                key={s}
                className="suggestion-chip"
                onClick={() => onSuggestion(s)}
                role="listitem"
                aria-label={`Suggest: ${s}`}
              >
                {s}
              </button>
            ))}
          </div>
        </>
      ) : (
        <>
          <h2>Upload a PDF to get started</h2>
          <p>
            Use the sidebar to upload any PDF document. I'll extract the text,
            build a semantic index, and answer your questions with precise source
            citations.
          </p>
        </>
      )}
    </div>
  );
}
