import React, { useState } from 'react';
import { ExternalLink, X, FileText, Sparkles, Globe, Eye } from 'lucide-react';
import { PdfViewerModal } from './PdfViewerModal';
import { api } from '../services/api';

export const KeySourcesPanel = ({ citations = {} }) => {
  const [showAllModal, setShowAllModal] = useState(false);
  const [activePdfViewer, setActivePdfViewer] = useState(null); // { pdfUrl, title, paperId }
  const citationsList = Object.values(citations);

  if (citationsList.length === 0) {
    return (
      <div className="key-sources-panel">
        <div className="key-sources-header-row">
          <h3 className="key-sources-header">Key Sources</h3>
        </div>
        <p style={{ fontSize: '0.85rem', color: '#64748b' }}>No key sources retrieved for this query.</p>
      </div>
    );
  }

  // Deduplicate sources by paper_id & section
  const uniqueSources = [];
  const seen = new Set();
  for (const c of citationsList) {
    const key = `${c.paper_id}-${c.section_name}`;
    if (!seen.has(key)) {
      seen.add(key);
      uniqueSources.push(c);
    }
  }

  const handleOpenPdfViewer = (e, source) => {
    e.stopPropagation();
    const pdfUrl = api.getOnlinePaperPdfUrl(source);
    if (pdfUrl && pdfUrl !== '#') {
      setActivePdfViewer({
        pdfUrl,
        title: source.title || source.paper_id,
        paperId: source.paper_id,
      });
    }
  };

  return (
    <>
      <div className="key-sources-panel">
        <div className="key-sources-header-row">
          <h3 className="key-sources-header">Key Sources</h3>
          <button type="button" className="view-all-link" onClick={() => setShowAllModal(true)}>
            View All ({uniqueSources.length})
          </button>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
          {uniqueSources.map((source, idx) => {
            const isOnline =
              source.source_type === 'online' ||
              (source.citation_id && source.citation_id.startsWith('O')) ||
              (source.paper_id && source.paper_id.startsWith('arXiv:'));

            return (
              <div key={idx} className="source-item-card" style={{ position: 'relative' }}>
                <div
                  className="pdf-icon-badge"
                  onClick={(e) => handleOpenPdfViewer(e, source)}
                  title="Click to view PDF document"
                  style={{
                    cursor: 'pointer',
                    background: isOnline ? '#dbeafe' : '#f3e8ff',
                    color: isOnline ? '#1e40af' : '#6d28d9',
                    border: isOnline ? '1px solid #93c5fd' : '1px solid #d8b4fe',
                  }}
                >
                  {isOnline ? <Globe size={11} style={{ marginRight: 2 }} /> : null}
                  <span>PDF</span>
                </div>
                <div className="source-item-info">
                  <h5 style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                    Paper {source.paper_id}
                    {isOnline && (
                      <span
                        style={{
                          background: '#eff6ff',
                          color: '#1d4ed8',
                          fontSize: '0.68rem',
                          fontWeight: 700,
                          padding: '0.1rem 0.35rem',
                          borderRadius: '4px',
                          border: '1px solid #bfdbfe',
                        }}
                      >
                        ONLINE
                      </span>
                    )}
                  </h5>
                  <p className="source-meta">{source.title || `${source.section_name} • pp. ${source.pages}`}</p>
                  
                  {/* Action button to open PDF viewer modal */}
                  <div style={{ marginTop: '0.35rem' }}>
                    <button
                      type="button"
                      onClick={(e) => handleOpenPdfViewer(e, source)}
                      style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '0.25rem',
                        fontSize: '0.75rem',
                        fontWeight: 700,
                        color: isOnline ? '#1d4ed8' : '#6d28d9',
                        background: isOnline ? '#eff6ff' : '#faf5ff',
                        padding: '0.25rem 0.55rem',
                        borderRadius: '6px',
                        border: isOnline ? '1px solid #bfdbfe' : '1px solid #e9d5ff',
                        cursor: 'pointer',
                      }}
                    >
                      <Eye size={12} />
                      <span>View PDF</span>
                    </button>
                  </div>
                </div>

                <button
                  type="button"
                  className="external-link-btn"
                  title="View PDF Document"
                  onClick={(e) => handleOpenPdfViewer(e, source)}
                  style={{
                    color: isOnline ? '#1d4ed8' : '#6d28d9',
                    background: isOnline ? '#eff6ff' : '#faf5ff',
                    border: isOnline ? '1px solid #bfdbfe' : '1px solid #e9d5ff',
                    borderRadius: '8px',
                    padding: '0.4rem',
                    cursor: 'pointer',
                  }}
                >
                  <Eye size={16} />
                </button>
              </div>
            );
          })}
        </div>
      </div>

      {/* View All Key Sources Modal */}
      {showAllModal && (
        <div className="modal-backdrop" onClick={() => setShowAllModal(false)}>
          <div className="modal-card" style={{ maxWidth: '680px' }} onClick={(e) => e.stopPropagation()}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#6d28d9', fontWeight: 800 }}>
                <FileText size={20} />
                <h3 style={{ fontSize: '1.2rem', color: '#0f172a' }}>Retrieved Key Sources ({uniqueSources.length})</h3>
              </div>
              <button type="button" onClick={() => setShowAllModal(false)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#64748b' }}>
                <X size={20} />
              </button>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem', maxHeight: '450px', overflowY: 'auto', paddingRight: '0.25rem' }}>
              {uniqueSources.map((source, idx) => {
                const isOnline =
                  source.source_type === 'online' ||
                  (source.citation_id && source.citation_id.startsWith('O')) ||
                  (source.paper_id && source.paper_id.startsWith('arXiv:'));

                return (
                  <div key={idx} className="source-item-card" style={{ background: isOnline ? '#f0f9ff' : '#f8fafc', border: isOnline ? '1px solid #bae6fd' : '1px solid #e2e8f0' }}>
                    <div
                      className="pdf-icon-badge"
                      onClick={(e) => handleOpenPdfViewer(e, source)}
                      style={{
                        cursor: 'pointer',
                        background: isOnline ? '#dbeafe' : '#f3e8ff',
                        color: isOnline ? '#1e40af' : '#6d28d9',
                      }}
                    >
                      <span>PDF</span>
                    </div>
                    <div className="source-item-info" style={{ flex: 1 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
                        <span
                          style={{
                            background: isOnline ? '#dbeafe' : '#efeafd',
                            color: isOnline ? '#1d4ed8' : '#6d28d9',
                            fontSize: '0.75rem',
                            fontWeight: 700,
                            padding: '0.1rem 0.5rem',
                            borderRadius: '4px',
                          }}
                        >
                          [{source.citation_id || `E${idx + 1}`}]
                        </span>
                        <h5 style={{ fontSize: '0.95rem' }}>Paper {source.paper_id}</h5>
                        {isOnline && (
                          <span style={{ background: '#2563eb', color: '#ffffff', fontSize: '0.68rem', fontWeight: 700, padding: '0.1rem 0.4rem', borderRadius: '4px' }}>
                            Online Retrieval
                          </span>
                        )}
                      </div>
                      {source.title && (
                        <p style={{ fontWeight: 600, fontSize: '0.875rem', color: '#1e293b', marginTop: '0.25rem' }}>
                          {source.title}
                        </p>
                      )}
                      <p className="source-meta" style={{ marginTop: '0.25rem' }}>
                        Section: <strong>"{source.section_name}"</strong> • Page range: {source.pages}
                      </p>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', alignSelf: 'center', marginLeft: '0.5rem' }}>
                      <button
                        type="button"
                        onClick={(e) => handleOpenPdfViewer(e, source)}
                        className="btn-primary"
                        style={{
                          fontSize: '0.8rem',
                          padding: '0.4rem 0.85rem',
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: '0.35rem',
                          background: isOnline ? '#2563eb' : '#6d28d9',
                          cursor: 'pointer',
                        }}
                      >
                        <Eye size={14} />
                        <span>View PDF</span>
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>

            <div style={{ textAlign: 'right', marginTop: '1.25rem' }}>
              <button type="button" className="btn-primary" onClick={() => setShowAllModal(false)}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Embedded PDF Viewer Modal */}
      {activePdfViewer && (
        <PdfViewerModal
          pdfUrl={activePdfViewer.pdfUrl}
          title={activePdfViewer.title}
          paperId={activePdfViewer.paperId}
          onClose={() => setActivePdfViewer(null)}
        />
      )}
    </>
  );
};
