import { useState, useCallback } from 'react';
import { ingestPDF, deleteSession } from '../api/client.js';

/**
 * useIngest — manages PDF upload and session lifecycle.
 */
export function useIngest() {
  const [session, setSession] = useState(null);  // { session_id, filename, num_chunks, num_pages }
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState(null);

  const upload = useCallback(async (file) => {
    setUploading(true);
    setUploadProgress(0);
    setError(null);

    try {
      const data = await ingestPDF(file, setUploadProgress);
      setSession(data);
      setUploadProgress(100);
    } catch (err) {
      setError(err.message || 'Upload failed');
      throw err;
    } finally {
      setUploading(false);
    }
  }, []);

  const clearSession = useCallback(async () => {
    if (session?.session_id) {
      try {
        await deleteSession(session.session_id);
      } catch {
        // Best-effort delete
      }
    }
    setSession(null);
    setError(null);
    setUploadProgress(0);
  }, [session]);

  return { session, uploading, uploadProgress, error, upload, clearSession };
}
