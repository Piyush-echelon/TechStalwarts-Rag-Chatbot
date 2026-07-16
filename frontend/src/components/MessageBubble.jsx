import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Sparkles, User } from 'lucide-react';
import TypingIndicator from './TypingIndicator.jsx';
import SourcePanel from './SourcePanel.jsx';

/**
 * MessageBubble — renders a single chat message.
 *
 * Props:
 *   message: { id, role, content, sources, streaming, error }
 */
export default function MessageBubble({ message }) {
  const isAI   = message.role === 'assistant';
  const isUser = message.role === 'user';

  return (
    <div className={`message-wrapper ${isUser ? 'user' : ''}`}>
      {/* Avatar */}
      <div className={`message-avatar ${isAI ? 'ai' : 'user'}`} aria-hidden="true">
        {isAI  ? <Sparkles size={15} color="#fff" /> : <User size={15} />}
      </div>

      {/* Bubble */}
      <div
        className={`message-bubble ${isAI ? 'ai' : 'user'}`}
        role="article"
        aria-label={`${isAI ? 'Assistant' : 'You'}: ${message.content.slice(0, 60)}`}
      >
        {/* Show typing dots when streaming and no content yet */}
        {isAI && message.streaming && !message.content ? (
          <TypingIndicator />
        ) : isAI ? (
          <div className="markdown-body">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {message.content}
            </ReactMarkdown>
          </div>
        ) : (
          <span>{message.content}</span>
        )}

        {/* Sources — shown after streaming completes */}
        {isAI && !message.streaming && (
          <SourcePanel sources={message.sources} />
        )}

        {/* Streaming cursor */}
        {isAI && message.streaming && message.content && (
          <span
            style={{
              display: 'inline-block',
              width: '2px',
              height: '1em',
              background: 'var(--text-accent)',
              marginLeft: '2px',
              verticalAlign: 'middle',
              animation: 'pulse-dot 0.8s ease-in-out infinite',
            }}
            aria-hidden="true"
          />
        )}
      </div>
    </div>
  );
}
