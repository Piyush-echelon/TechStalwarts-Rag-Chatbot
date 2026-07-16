import React from 'react';

/** Animated three-dot typing indicator. */
export default function TypingIndicator() {
  return (
    <div className="typing-indicator" aria-label="AI is typing" role="status">
      <span className="typing-dot" />
      <span className="typing-dot" />
      <span className="typing-dot" />
    </div>
  );
}
