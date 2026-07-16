import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Send } from 'lucide-react';
import MessageBubble from './MessageBubble.jsx';
import EmptyState from './EmptyState.jsx';

/**
 * ChatWindow — message list + input bar.
 *
 * Props:
 *   messages:     Message[]
 *   streaming:    bool
 *   sessionReady: bool
 *   onSend(query: string): void
 */
export default function ChatWindow({ messages, streaming, sessionReady, onSend }) {
  const [input, setInput] = useState('');
  const bottomRef  = useRef(null);
  const textareaRef = useRef(null);

  // Auto-scroll to bottom on new messages / tokens
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Auto-resize textarea
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = 'auto';
    ta.style.height = `${Math.min(ta.scrollHeight, 120)}px`;
  }, [input]);

  const handleSend = useCallback(() => {
    const trimmed = input.trim();
    if (!trimmed || streaming || !sessionReady) return;
    onSend(trimmed);
    setInput('');
  }, [input, streaming, sessionReady, onSend]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const canSend = sessionReady && !streaming && input.trim().length > 0;

  return (
    <>
      {/* Messages */}
      <div className="messages-container" role="log" aria-live="polite" aria-label="Conversation">
        {messages.length === 0 ? (
          <EmptyState sessionReady={sessionReady} onSuggestion={(s) => { setInput(s); textareaRef.current?.focus(); }} />
        ) : (
          messages.map((msg) => <MessageBubble key={msg.id} message={msg} />)
        )}
        <div ref={bottomRef} aria-hidden="true" />
      </div>

      {/* Input */}
      <div className="chat-input-area">
        <div className="chat-input-wrapper">
          <textarea
            ref={textareaRef}
            id="chat-input"
            className="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              sessionReady
                ? 'Ask a question about your document…'
                : 'Upload a PDF to start chatting'
            }
            disabled={!sessionReady || streaming}
            rows={1}
            aria-label="Chat message input"
            aria-multiline="true"
          />
          <button
            id="chat-send-btn"
            className="chat-send-btn"
            onClick={handleSend}
            disabled={!canSend}
            aria-label="Send message"
          >
            <Send size={16} />
          </button>
        </div>
        <p className="chat-input-hint">
          {sessionReady
            ? 'Press Enter to send · Shift+Enter for a new line'
            : 'Upload a PDF in the sidebar first'}
        </p>
      </div>
    </>
  );
}
