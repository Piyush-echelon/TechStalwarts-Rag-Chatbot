import React, { useCallback, useState } from 'react';
import Sidebar from './components/Sidebar.jsx';
import ChatWindow from './components/ChatWindow.jsx';
import { useIngest } from './hooks/useIngest.js';
import { useChat } from './hooks/useChat.js';

export default function App() {
  const { session, uploading, uploadProgress, error: uploadError, upload, clearSession } = useIngest();
  const { messages, streaming, send, clearMessages } = useChat(session?.session_id ?? null);

  // Toast state for error notifications
  const [toast, setToast] = useState(null);

  const showToast = useCallback((msg, type = 'error') => {
    setToast({ msg, type });
    setTimeout(() => setToast(null), 4000);
  }, []);

  const handleUpload = useCallback(async (file) => {
    try {
      await upload(file);
    } catch (err) {
      showToast(err.message || 'Upload failed');
    }
  }, [upload, showToast]);

  const handleNewChat = useCallback(async () => {
    clearMessages();
    await clearSession();
  }, [clearMessages, clearSession]);

  const handleSend = useCallback((query) => {
    send(query);
  }, [send]);

  return (
    <div className="app-layout">
      {/* ── Sidebar ────────────────────────────────────── */}
      <Sidebar
        session={session}
        uploading={uploading}
        uploadProgress={uploadProgress}
        uploadError={uploadError}
        onUpload={handleUpload}
        onNewChat={handleNewChat}
      />

      {/* ── Chat area ──────────────────────────────────── */}
      <main className="chat-area" aria-label="Chat area">
        {/* Header */}
        <header className="chat-header" role="banner">
          <div className="chat-header-status">
            <span className={`status-dot ${session ? 'active' : ''}`} aria-hidden="true" />
            <span>
              {session
                ? `Chatting about "${session.filename}"`
                : streaming
                ? 'Generating response…'
                : 'No document loaded'}
            </span>
          </div>
        </header>

        {/* Messages + Input */}
        <ChatWindow
          messages={messages}
          streaming={streaming}
          sessionReady={!!session}
          onSend={handleSend}
        />
      </main>

      {/* ── Toast notification ─────────────────────────── */}
      {toast && (
        <div
          className={`toast ${toast.type}`}
          role="alert"
          aria-live="assertive"
        >
          <span>{toast.msg}</span>
        </div>
      )}
    </div>
  );
}
