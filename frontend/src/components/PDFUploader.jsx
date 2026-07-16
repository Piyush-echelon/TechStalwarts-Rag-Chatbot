import React, { useRef, useState } from 'react';
import { Upload, FileText } from 'lucide-react';

/**
 * PDFUploader — drag-and-drop or click-to-upload PDF zone.
 *
 * Props:
 *   onFile(file: File) — called when a valid PDF is selected
 *   uploading: bool
 *   uploadProgress: number (0–100)
 *   error: string|null
 */
export default function PDFUploader({ onFile, uploading, uploadProgress, error }) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef(null);

  function handleDrop(e) {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type === 'application/pdf') onFile(file);
  }

  function handleChange(e) {
    const file = e.target.files[0];
    if (file) onFile(file);
    // Reset input so same file can be re-uploaded
    e.target.value = '';
  }

  return (
    <div>
      <div
        className={`uploader-zone${dragOver ? ' drag-over' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => !uploading && inputRef.current?.click()}
        role="button"
        aria-label="Upload PDF document"
        tabIndex={0}
        onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
        style={{ cursor: uploading ? 'not-allowed' : 'pointer' }}
      >
        <div className="upload-icon">
          <Upload size={28} />
        </div>
        <p className="upload-title">
          {uploading ? 'Processing PDF…' : 'Upload your PDF'}
        </p>
        <p className="upload-subtitle">
          {uploading
            ? 'Extracting text and building vector index'
            : 'Drag & drop or click to browse'}
        </p>

        <input
          ref={inputRef}
          type="file"
          accept=".pdf,application/pdf"
          onChange={handleChange}
          disabled={uploading}
          aria-hidden="true"
          tabIndex={-1}
        />
      </div>

      {uploading && (
        <div className="upload-progress" aria-live="polite" aria-label={`Upload progress ${uploadProgress}%`}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.74rem', color: 'var(--text-secondary)' }}>
            <span>Indexing document…</span>
            <span>{uploadProgress}%</span>
          </div>
          <div className="upload-progress-bar-track">
            <div className="upload-progress-bar-fill" style={{ width: `${uploadProgress}%` }} />
          </div>
        </div>
      )}

      {error && (
        <p style={{ fontSize: '0.75rem', color: 'var(--error)', marginTop: '8px', lineHeight: 1.5 }}>
          {error}
        </p>
      )}
    </div>
  );
}
