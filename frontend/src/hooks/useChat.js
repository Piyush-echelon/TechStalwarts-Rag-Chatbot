import { useState, useCallback, useRef } from 'react';
import { streamChat } from '../api/client.js';

/**
 * Message shape:
 * { id, role: 'user'|'assistant', content, sources, streaming, error }
 */
let msgId = 0;
const nextId = () => `msg-${++msgId}`;

/**
 * useChat — manages the conversation state and SSE streaming.
 *
 * @param {string|null} sessionId  — the active RAG session ID
 */
export function useChat(sessionId) {
  const [messages, setMessages] = useState([]);
  const [streaming, setStreaming] = useState(false);
  const abortRef = useRef(false);

  const appendToken = useCallback((id, token) => {
    setMessages((prev) =>
      prev.map((m) =>
        m.id === id ? { ...m, content: m.content + token } : m
      )
    );
  }, []);

  const setSources = useCallback((id, sources) => {
    setMessages((prev) =>
      prev.map((m) => (m.id === id ? { ...m, sources } : m))
    );
  }, []);

  const markDone = useCallback((id) => {
    setMessages((prev) =>
      prev.map((m) => (m.id === id ? { ...m, streaming: false } : m))
    );
  }, []);

  const send = useCallback(
    async (query) => {
      if (!sessionId || !query.trim() || streaming) return;

      // Add user message
      const userMsgId = nextId();
      const aiMsgId   = nextId();

      setMessages((prev) => [
        ...prev,
        { id: userMsgId, role: 'user', content: query, sources: [], streaming: false },
        { id: aiMsgId,   role: 'assistant', content: '', sources: [], streaming: true, error: null },
      ]);

      setStreaming(true);
      abortRef.current = false;

      await streamChat(sessionId, query, {
        onToken: (token) => {
          if (!abortRef.current) appendToken(aiMsgId, token);
        },
        onSources: (sources) => {
          if (!abortRef.current) setSources(aiMsgId, sources);
        },
        onDone: () => {
          markDone(aiMsgId);
          setStreaming(false);
        },
        onError: (err) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === aiMsgId
                ? { ...m, content: `Error: ${err.message}`, streaming: false, error: true }
                : m
            )
          );
          setStreaming(false);
        },
      });
    },
    [sessionId, streaming, appendToken, setSources, markDone]
  );

  const clearMessages = useCallback(() => {
    setMessages([]);
    setStreaming(false);
    abortRef.current = true;
  }, []);

  return { messages, streaming, send, clearMessages };
}
