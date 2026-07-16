/**
 * API client — wraps all backend calls.
 * Uses VITE_API_URL in production; the Vite dev proxy handles /api in development.
 */

const API_BASE = import.meta.env.VITE_API_URL
  ? `${import.meta.env.VITE_API_URL}/api`
  : '/api';

// ── Helpers ───────────────────────────────────────────────────────────────────

async function handleResponse(res) {
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

// ── Endpoints ─────────────────────────────────────────────────────────────────

/**
 * Upload a PDF file and start a new RAG session.
 * @param {File} file
 * @param {(progress: number) => void} onProgress
 * @returns {Promise<{ session_id, filename, num_chunks, num_pages }>}
 */
export async function ingestPDF(file, onProgress) {
  const formData = new FormData();
  formData.append('file', file);

  // Use XMLHttpRequest for upload progress tracking
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API_BASE}/ingest`);

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    };

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText));
      } else {
        const err = JSON.parse(xhr.responseText || '{}');
        reject(new Error(err.detail || `Upload failed: HTTP ${xhr.status}`));
      }
    };

    xhr.onerror = () => reject(new Error('Network error during upload'));
    xhr.send(formData);
  });
}

/**
 * Stream a chat response using the SSE endpoint.
 *
 * Calls onToken for each streamed token, onSources when citations arrive,
 * and onDone when the stream finishes.
 *
 * @param {string} sessionId
 * @param {string} query
 * @param {{ onToken, onSources, onDone, onError }} callbacks
 */
export async function streamChat(sessionId, query, { onToken, onSources, onDone, onError }) {
  try {
    const res = await fetch(`${API_BASE}/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, query }),
    });

    if (!res.ok) {
      const body = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(body.detail || `HTTP ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n\n');
      buffer = lines.pop(); // Keep incomplete last segment

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith('data: ')) continue;

        try {
          const payload = JSON.parse(trimmed.slice(6));

          if (payload.type === 'token' && onToken) {
            onToken(payload.content);
          } else if (payload.type === 'sources' && onSources) {
            onSources(payload.sources);
          } else if (payload.type === 'done' && onDone) {
            onDone();
          } else if (payload.type === 'error' && onError) {
            onError(new Error(payload.message));
          }
        } catch {
          // Ignore malformed SSE lines
        }
      }
    }
  } catch (err) {
    if (onError) onError(err);
  }
}

/**
 * Delete a session and its vector collection.
 * @param {string} sessionId
 */
export async function deleteSession(sessionId) {
  const res = await fetch(`${API_BASE}/sessions/${sessionId}`, { method: 'DELETE' });
  return handleResponse(res);
}

/**
 * Fetch session metadata and history.
 * @param {string} sessionId
 */
export async function getSession(sessionId) {
  const res = await fetch(`${API_BASE}/sessions/${sessionId}`);
  return handleResponse(res);
}

/** Liveness check */
export async function healthCheck() {
  const res = await fetch(`${API_BASE}/health`);
  return handleResponse(res);
}
