import React, { useState, useEffect } from 'react';
import { ChevronDown, ChevronUp, FileText, Globe, Eye } from 'lucide-react';
import { PdfViewerModal } from './PdfViewerModal';
import { api } from '../services/api';

export const EvidenceAccordion = ({ evidence = [], activeCitationId = null }) => {
  const [openIndex, setOpenIndex] = useState(null);
  const [activePdfViewer, setActivePdfViewer] = useState(null); // { pdfUrl, title, paperId }

  useEffect(() => {
    if (activeCitationId && evidence.length > 0) {
      const idx = evidence.findIndex(
        (item) => item.citation_id === activeCitationId || item.unit_id === activeCitationId || item.chunk_id === activeCitationId
      );
      if (idx !== -1) {
        setOpenIndex(idx);
      }
    }
  }, [activeCitationId, evidence]);

  if (!evidence || evidence.length === 0) return null;

  const handleOpenPdfViewer = (item) => {
    const pdfUrl = api.getOnlinePaperPdfUrl(item);
    if (pdfUrl && pdfUrl !== '#') {
      setActivePdfViewer({
        pdfUrl,
        title: item.title || item.paper_id,
        paperId: item.paper_id,
      });
    }
  };

  return (
    <div style={{ marginTop: '2rem' }}>
      <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#0f172a', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <FileText size={18} style={{ color: '#6d28d9' }} />
        Retrieved Evidence ({evidence.length} Passages)
      </h3>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
        {evidence.map((item, idx) => {
          const isOpen = openIndex === idx;
          const citeTag = item.citation_id || `E${idx + 1}`;
          const isSelected = activeCitationId && (activeCitationId === citeTag || activeCitationId === item.unit_id);
          const isOnline =
            item.source_type === 'online' ||
            citeTag.startsWith('O') ||
            (item.paper_id && item.paper_id.startsWith('arXiv:'));
          const pdfUrl = api.getOnlinePaperPdfUrl(item);

          return (
            <div
              key={idx}
              id={`evidence-${citeTag}`}
              className={isSelected ? 'active-evidence-card' : ''}
              style={{
                background: '#ffffff',
                border: isSelected
                  ? isOnline ? '2px solid #2563eb' : '2px solid #6d28d9'
                  : isOnline ? '1px solid #bfdbfe' : '1px solid #e2e8f0',
                borderRadius: '12px',
                overflow: 'hidden',
                transition: 'all 0.2s ease',
              }}
            >
              <button
                type="button"
                onClick={() => setOpenIndex(isOpen ? null : idx)}
                style={{
                  width: '100%',
                  padding: '0.85rem 1.25rem',
                  background: isOpen ? (isOnline ? '#eff6ff' : '#faf8ff') : '#ffffff',
                  border: 'none',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  cursor: 'pointer',
                  textAlign: 'left',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
                  <span
                    style={{
                      background: isSelected
                        ? isOnline ? '#2563eb' : '#6d28d9'
                        : isOnline ? '#dbeafe' : '#efeafd',
                      color: isSelected ? '#ffffff' : isOnline ? '#1d4ed8' : '#6d28d9',
                      fontWeight: 700,
                      fontSize: '0.825rem',
                      padding: '0.2rem 0.6rem',
                      borderRadius: '6px',
                    }}
                  >
                    [{citeTag}]
                  </span>
                  <span style={{ fontWeight: 600, fontSize: '0.9rem', color: '#0f172a' }}>
                    Paper {item.paper_id} — {item.title ? item.title : `${item.section_name} (pp. ${item.page_start || 1}–${item.page_end || 1})`}
                  </span>
                  {isOnline && (
                    <span
                      style={{
                        background: '#eff6ff',
                        color: '#1d4ed8',
                        fontSize: '0.7rem',
                        fontWeight: 700,
                        padding: '0.1rem 0.45rem',
                        borderRadius: '4px',
                        border: '1px solid #bfdbfe',
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '0.2rem',
                      }}
                    >
                      <Globe size={10} /> Online Academic Search
                    </span>
                  )}
                </div>
                {isOpen ? (
                  <ChevronUp size={16} color={isOnline ? '#2563eb' : '#6d28d9'} />
                ) : (
                  <ChevronDown size={16} color="#64748b" />
                )}
              </button>

              {isOpen && (
                <div style={{ padding: '1rem 1.25rem', borderTop: '1px solid #e2e8f0', background: '#ffffff' }}>
                  <div style={{ fontSize: '0.8rem', color: '#64748b', marginBottom: '0.75rem', display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'center' }}>
                    <span>Domain: <strong>{item.domain}</strong></span>
                    <span>Subtopic: <strong>{item.subtopic || 'General'}</strong></span>
                    <span>Chunk ID: <code>{item.chunk_id || item.unit_id}</code></span>

                    {/* View PDF Button */}
                    {pdfUrl && pdfUrl !== '#' && (
                      <button
                        type="button"
                        onClick={() => handleOpenPdfViewer(item)}
                        style={{
                          marginLeft: 'auto',
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: '0.35rem',
                          background: isOnline ? '#2563eb' : '#6d28d9',
                          color: '#ffffff',
                          border: 'none',
                          padding: '0.35rem 0.75rem',
                          borderRadius: '6px',
                          fontSize: '0.78rem',
                          fontWeight: 700,
                          cursor: 'pointer',
                        }}
                      >
                        <Eye size={13} />
                        <span>View PDF</span>
                      </button>
                    )}
                  </div>

                  {item.title && (
                    <div style={{ fontSize: '0.9rem', fontWeight: 700, color: '#1e293b', marginBottom: '0.5rem' }}>
                      {item.title}
                    </div>
                  )}

                  <p style={{ fontSize: '0.925rem', color: '#334155', lineHeight: 1.6, background: isOnline ? '#f0f9ff' : '#f8fafc', padding: '0.85rem', borderRadius: '8px', borderLeft: isOnline ? '3px solid #2563eb' : '3px solid #6d28d9' }}>
                    "{item.text}"
                  </p>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Embedded PDF Viewer Modal */}
      {activePdfViewer && (
        <PdfViewerModal
          pdfUrl={activePdfViewer.pdfUrl}
          title={activePdfViewer.title}
          paperId={activePdfViewer.paperId}
          onClose={() => setActivePdfViewer(null)}
        />
      )}
    </div>
  );
};
