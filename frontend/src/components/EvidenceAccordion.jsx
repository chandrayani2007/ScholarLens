import React, { useState, useEffect } from 'react';
import { ChevronDown, ChevronUp, FileText } from 'lucide-react';

export const EvidenceAccordion = ({ evidence = [], activeCitationId = null }) => {
  const [openIndex, setOpenIndex] = useState(null);

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

          return (
            <div
              key={idx}
              id={`evidence-${citeTag}`}
              className={isSelected ? 'active-evidence-card' : ''}
              style={{
                background: '#ffffff',
                border: isSelected ? '2px solid #6d28d9' : '1px solid #e2e8f0',
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
                  background: isOpen ? '#faf8ff' : '#ffffff',
                  border: 'none',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  cursor: 'pointer',
                  textAlign: 'left',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <span
                    style={{
                      background: isSelected ? '#6d28d9' : '#efeafd',
                      color: isSelected ? '#ffffff' : '#6d28d9',
                      fontWeight: 700,
                      fontSize: '0.825rem',
                      padding: '0.2rem 0.6rem',
                      borderRadius: '6px',
                    }}
                  >
                    [{citeTag}]
                  </span>
                  <span style={{ fontWeight: 600, fontSize: '0.9rem', color: '#0f172a' }}>
                    Paper {item.paper_id} — {item.section_name} (pp. {item.page_start}–{item.page_end})
                  </span>
                </div>
                {isOpen ? <ChevronUp size={16} color="#6d28d9" /> : <ChevronDown size={16} color="#64748b" />}
              </button>

              {isOpen && (
                <div style={{ padding: '1rem 1.25rem', borderTop: '1px solid #e2e8f0', background: '#ffffff' }}>
                  <div style={{ fontSize: '0.8rem', color: '#64748b', marginBottom: '0.5rem', display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                    <span>Domain: <strong>{item.domain}</strong></span>
                    <span>Subtopic: <strong>{item.subtopic}</strong></span>
                    <span>Chunk ID: <code>{item.chunk_id || item.unit_id}</code></span>
                  </div>
                  <p style={{ fontSize: '0.925rem', color: '#334155', lineHeight: 1.6, background: '#f8fafc', padding: '0.85rem', borderRadius: '8px', borderLeft: '3px solid #6d28d9' }}>
                    "{item.text}"
                  </p>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
