import React, { useState } from 'react';
import { ExternalLink, X, FileText, Sparkles } from 'lucide-react';

export const KeySourcesPanel = ({ citations = {} }) => {
  const [showAllModal, setShowAllModal] = useState(false);
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
          {uniqueSources.map((source, idx) => (
            <div key={idx} className="source-item-card">
              <div className="pdf-icon-badge">
                <span>PDF</span>
              </div>
              <div className="source-item-info">
                <h5>Paper {source.paper_id}</h5>
                <p className="source-meta">{source.section_name} • pp. {source.pages}</p>
                <span className="relevance-pill">
                  {idx === 0 ? 'Highly Relevant' : 'Relevant'}
                </span>
              </div>
              <button
                type="button"
                className="external-link-btn"
                title="View Source Details"
                onClick={() => setShowAllModal(true)}
              >
                <ExternalLink size={16} />
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* View All Key Sources Modal */}
      {showAllModal && (
        <div className="modal-backdrop" onClick={() => setShowAllModal(false)}>
          <div className="modal-card" style={{ maxWidth: '640px' }} onClick={(e) => e.stopPropagation()}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#6d28d9', fontWeight: 800 }}>
                <FileText size={20} />
                <h3 style={{ fontSize: '1.2rem', color: '#0f172a' }}>Retrieved Key Sources ({uniqueSources.length})</h3>
              </div>
              <button type="button" onClick={() => setShowAllModal(false)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#64748b' }}>
                <X size={20} />
              </button>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem', maxHeight: '420px', overflowY: 'auto', paddingRight: '0.25rem' }}>
              {uniqueSources.map((source, idx) => (
                <div key={idx} className="source-item-card" style={{ background: '#f8fafc' }}>
                  <div className="pdf-icon-badge">
                    <span>PDF</span>
                  </div>
                  <div className="source-item-info">
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <span style={{ background: '#efeafd', color: '#6d28d9', fontSize: '0.75rem', fontWeight: 700, padding: '0.1rem 0.5rem', borderRadius: '4px' }}>
                        [{source.citation_id || `E${idx + 1}`}]
                      </span>
                      <h5 style={{ fontSize: '0.95rem' }}>Paper {source.paper_id}</h5>
                    </div>
                    <p className="source-meta" style={{ marginTop: '0.25rem' }}>
                      Section: <strong>"{source.section_name}"</strong> • Page range: {source.pages}
                    </p>
                    <p className="source-meta" style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
                      Chunk ID: <code>{source.chunk_id || source.unit_id}</code>
                    </p>
                  </div>
                </div>
              ))}
            </div>

            <div style={{ textAlign: 'right', marginTop: '1.25rem' }}>
              <button type="button" className="btn-primary" onClick={() => setShowAllModal(false)}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};
