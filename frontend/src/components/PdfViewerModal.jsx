import React from 'react';
import { X, ExternalLink, FileText } from 'lucide-react';

export const PdfViewerModal = ({ pdfUrl, title, paperId, onClose }) => {
  if (!pdfUrl) return null;

  return (
    <div className="modal-overlay" onClick={onClose} style={{ zIndex: 11000 }}>
      <div
        className="modal-content card"
        style={{
          maxWidth: '1000px',
          width: '92vw',
          height: '88vh',
          display: 'flex',
          flexDirection: 'column',
          padding: 0,
          overflow: 'hidden',
          borderRadius: '16px',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div
          style={{
            padding: '1rem 1.5rem',
            background: 'linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%)',
            color: '#ffffff',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            borderBottom: '1px solid #334155',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', overflow: 'hidden' }}>
            <FileText size={22} style={{ color: '#a78bfa', flexShrink: 0 }} />
            <div style={{ overflow: 'hidden' }}>
              <h4 style={{ margin: 0, fontSize: '1.05rem', fontWeight: 700, color: '#f8fafc', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {title || `Paper ${paperId || ''}`}
              </h4>
              <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
                {paperId ? `Paper ID: ${paperId} • ` : ''}Document PDF Reader
              </span>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <a
              href={pdfUrl}
              target="_blank"
              rel="noopener noreferrer"
              style={{
                fontSize: '0.8rem',
                padding: '0.45rem 0.85rem',
                textDecoration: 'none',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.35rem',
                background: '#6d28d9',
                color: '#ffffff',
                borderRadius: '8px',
                fontWeight: 700,
              }}
            >
              <ExternalLink size={14} />
              <span>Open in New Tab</span>
            </a>
            <button
              type="button"
              onClick={onClose}
              style={{
                background: 'rgba(255,255,255,0.1)',
                border: 'none',
                color: '#ffffff',
                borderRadius: '8px',
                padding: '0.4rem',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <X size={20} />
            </button>
          </div>
        </div>

        {/* Embedded PDF Viewer Frame */}
        <div style={{ flex: 1, position: 'relative', background: '#525659' }}>
          <iframe
            src={pdfUrl}
            title={title || 'Paper PDF Document'}
            style={{ width: '100%', height: '100%', border: 'none' }}
          />
        </div>
      </div>
    </div>
  );
};
