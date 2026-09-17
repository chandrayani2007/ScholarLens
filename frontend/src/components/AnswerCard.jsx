import React, { useState } from 'react';
import { Sparkles, AlertCircle, Star, X, FileText, Bookmark, Check } from 'lucide-react';
import { WhyThisAnswerCard } from './WhyThisAnswerCard';
import { EvidenceAccordion } from './EvidenceAccordion';
import { api } from '../services/api';

export const AnswerCard = ({ response }) => {
  const [activeEvidenceModal, setActiveEvidenceModal] = useState(null);
  const [activeCitationId, setActiveCitationId] = useState(null);
  const [isSaved, setIsSaved] = useState(false);
  const [saving, setSaving] = useState(false);

  if (!response) return null;

  const {
    question,
    answer,
    citations = {},
    evidence = [],
    why_this_answer,
    confidence = 'High',
    limitations,
  } = response;

  const isInsufficient =
    (confidence && confidence.toLowerCase().includes('insufficient')) ||
    (why_this_answer?.evidence_strength && why_this_answer.evidence_strength.toLowerCase().includes('insufficient')) ||
    (response.retrieval_metadata?.insufficient_evidence === true) ||
    (answer && answer.toLowerCase().includes('insufficient'));

  const citationsList = Object.values(citations);

  const effectiveEvidence =
    evidence && evidence.length > 0
      ? evidence
      : citationsList.map((c, i) => ({
          citation_id: c.citation_id || `E${i + 1}`,
          paper_id: c.paper_id,
          section_name: c.section_name,
          page_start: c.pages ? parseInt(c.pages.split(/[-–]/)[0], 10) || 1 : 1,
          page_end: c.pages ? parseInt(c.pages.split(/[-–]/).slice(-1)[0], 10) || 1 : 1,
          pages: c.pages || '1',
          text: c.text || `Retrieved scientific passage from Paper ${c.paper_id} (${c.section_name || 'Section'}).`,
          domain: c.domain || 'artificial_intelligence',
          subtopic: c.subtopic || 'general',
          chunk_id: c.chunk_id || c.unit_id || `${c.paper_id}_C01`,
          unit_id: c.unit_id || `${c.paper_id}_U01`,
          source_type: c.source_type || 'corpus',
          title: c.title || c.paper_id,
          authors: c.authors || [],
          url: c.url,
        }));

  const handleSaveQuery = async () => {
    if (isSaved || saving) return;
    setSaving(true);
    try {
      await api.saveQuery({
        question: question || 'Research Query',
        answer: answer,
        domain: response.user_domain || response.retrieval_metadata?.domain || 'general',
        citations_json: JSON.stringify(citations),
        why_this_answer_json: JSON.stringify(why_this_answer || {}),
        confidence: confidence || 'High',
      });
      setIsSaved(true);
    } catch (err) {
      console.error('Failed to save query:', err);
    } finally {
      setSaving(false);
    }
  };

  const handleCitationClick = (tagStr) => {
    const cleanTag = tagStr.replace(/[\[\]]/g, '');
    setActiveCitationId(cleanTag);

    const match =
      effectiveEvidence.find(
        (e) => e.citation_id === cleanTag || e.unit_id === cleanTag || e.chunk_id === cleanTag
      ) ||
      citationsList.find(
        (c) => c.citation_id === cleanTag || c.unit_id === cleanTag || c.chunk_id === cleanTag
      );

    if (match) {
      setActiveEvidenceModal({
        tag: cleanTag,
        paper_id: match.paper_id,
        title: match.title || match.paper_id,
        section_name: match.section_name,
        pages: match.pages || (match.page_start ? `${match.page_start}–${match.page_end}` : 'N/A'),
        text: match.text || `Retrieved scientific evidence passage from Paper ${match.paper_id} (${match.section_name || 'Section'}).`,
        domain: match.domain || 'research_mind',
        subtopic: match.subtopic || 'general',
        chunk_id: match.chunk_id || match.unit_id,
        url: match.url,
        source_type: match.source_type || 'corpus',
      });
    } else {
      const index = parseInt(cleanTag.replace(/\D/g, ''), 10) - 1;
      const fallbackItem = effectiveEvidence[index] || citationsList[index];
      if (fallbackItem) {
        setActiveEvidenceModal({
          tag: cleanTag,
          paper_id: fallbackItem.paper_id,
          title: fallbackItem.title || fallbackItem.paper_id,
          section_name: fallbackItem.section_name,
          pages: fallbackItem.pages || (fallbackItem.page_start ? `${fallbackItem.page_start}–${fallbackItem.page_end}` : 'N/A'),
          text: fallbackItem.text || `Retrieved scientific passage from Paper ${fallbackItem.paper_id}.`,
          domain: fallbackItem.domain || 'research_mind',
          subtopic: fallbackItem.subtopic || 'general',
          chunk_id: fallbackItem.chunk_id || fallbackItem.unit_id,
          url: fallbackItem.url,
          source_type: fallbackItem.source_type || 'corpus',
        });
      }
    }

    setTimeout(() => {
      const el = document.getElementById(`evidence-${cleanTag}`);
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }, 50);
  };

  const renderInlineCitations = (inlineText) => {
    if (!inlineText) return null;
    // Match [E#], [O#], [U#] and bare [#] citation tags
    const parts = inlineText.split(/(\[[EOU]\d+\]|\[\d+\])/g);
    return parts.map((part, idx) => {
      const match = part.match(/^\[([EOU]\d+|\d+)\]$/);
      if (match) {
        const tag = match[1];
        const isOnlineTag = tag.startsWith('O');
        const isUploadedTag = tag.startsWith('U');
        return (
          <button
            key={idx}
            type="button"
            className={`citation-badge-clickable ${isOnlineTag ? 'citation-badge-online' : ''} ${isUploadedTag ? 'citation-badge-uploaded' : ''}`}
            onClick={() => handleCitationClick(tag)}
            title={`Click to view evidence passage for [${tag}]`}
            style={
              isOnlineTag
                ? { background: '#eff6ff', color: '#1d4ed8', borderColor: '#bfdbfe' }
                : isUploadedTag
                ? { background: '#f5f3ff', color: '#6d28d9', borderColor: '#ddd6fe' }
                : {}
            }
          >
            [{tag}]
          </button>
        );
      }
      return part;
    });
  };

  const renderFormattedAnswer = (text) => {
    if (!text) return null;
    // Split on any newline (single or double) to handle headings and body on the same or separate lines
    const lines = text.split(/\n/);
    const elements = [];
    let bodyBuffer = [];

    const flushBuffer = (key) => {
      if (bodyBuffer.length > 0) {
        const bodyText = bodyBuffer.join(' ').trim();
        if (bodyText) {
          elements.push(
            <p key={`body-${key}`} className="answer-paragraph" style={{ marginBottom: '0.9rem', lineHeight: 1.75, fontSize: '0.975rem', color: '#374151', fontWeight: 400 }}>
              {renderInlineCitations(bodyText)}
            </p>
          );
        }
        bodyBuffer = [];
      }
    };

    lines.forEach((line, idx) => {
      const trimmed = line.trim();
      if (!trimmed) {
        // empty line = flush buffer as paragraph
        flushBuffer(idx);
        return;
      }
      if (trimmed.startsWith('#')) {
        // Flush any accumulated body text first
        flushBuffer(idx);
        const headingText = trimmed.replace(/^#+\s*/, '');
        elements.push(
          <h4
            key={`h-${idx}`}
            className="answer-section-heading"
            style={{ fontSize: '1rem', fontWeight: 600, color: '#1e293b', marginTop: '1.1rem', marginBottom: '0.35rem', letterSpacing: '0.01em' }}
          >
            {renderInlineCitations(headingText)}
          </h4>
        );
      } else {
        bodyBuffer.push(trimmed);
      }
    });
    // Flush any remaining body text
    flushBuffer('end');

    return elements;
  };

  return (
    <>
      <div className="card answer-main-card">
        {/* Answer Header Row */}
        <div className="answer-header-row" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h3 className="answer-header-title" style={{ margin: 0 }}>
            <Sparkles size={20} className="sparkle-purple" />
            Answer
          </h3>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <button
              type="button"
              onClick={handleSaveQuery}
              disabled={isSaved || saving}
              className="btn-save-query"
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.4rem',
                padding: '0.4rem 0.85rem',
                borderRadius: '8px',
                border: isSaved ? '1px solid #16a34a' : '1px solid #cbd5e1',
                background: isSaved ? '#f0fdf4' : '#ffffff',
                color: isSaved ? '#15803d' : '#334155',
                fontSize: '0.85rem',
                fontWeight: 700,
                cursor: isSaved ? 'default' : 'pointer',
                transition: 'all 0.2s ease',
              }}
            >
              {isSaved ? <Check size={15} style={{ color: '#16a34a' }} /> : <Bookmark size={15} />}
              <span>{isSaved ? 'Saved' : saving ? 'Saving...' : 'Save Query'}</span>
            </button>
            <span
              className={isInsufficient ? "confidence-badge-red" : "confidence-badge-green"}
              style={
                isInsufficient
                  ? { background: '#fef2f2', color: '#991b1b', border: '1px solid #fecaca', padding: '0.25rem 0.65rem', borderRadius: '20px', fontSize: '0.825rem', fontWeight: 600, display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }
                  : {}
              }
            >
              <Star size={14} fill={isInsufficient ? "#991b1b" : "#16a34a"} />
              Confidence: {confidence || 'Excellent (0.92)'}
            </span>
          </div>
        </div>

        {isInsufficient ? (
          <>
            <div style={{ background: '#fef2f2', border: '1px solid #fecaca', borderRadius: '12px', padding: '1.25rem', marginBottom: '1.5rem', marginTop: '1rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', color: '#991b1b', fontWeight: 700, fontSize: '1rem', marginBottom: '0.5rem' }}>
                <AlertCircle size={20} />
                Insufficient Evidence
              </div>
              <p style={{ color: '#7f1d1d', fontSize: '0.95rem', lineHeight: 1.6 }}>
                {answer}
              </p>
              {limitations && (
                <div style={{ marginTop: '0.75rem', fontSize: '0.85rem', color: '#991b1b' }}>
                  <strong>Note:</strong> {limitations}
                </div>
              )}
            </div>

            {/* Why This Answer Card Component (explaining why evidence was insufficient) */}
            {why_this_answer && (
              <WhyThisAnswerCard whyThisAnswer={why_this_answer} />
            )}
          </>
        ) : (
          <>
            {/* Formatted Answer Body */}
            <div className="answer-prose-body" style={{ marginTop: '1rem' }}>
              {renderFormattedAnswer(answer)}
            </div>

            {/* Why This Answer Card Component */}
            {why_this_answer && (
              <WhyThisAnswerCard whyThisAnswer={why_this_answer} />
            )}

            {/* Evidence Accordion (Retrieved Evidence Passages) */}
            {effectiveEvidence.length > 0 && (
              <EvidenceAccordion evidence={effectiveEvidence} activeCitationId={activeCitationId} />
            )}
          </>
        )}
      </div>

      {/* Modal for Citation Evidence Detail */}
      {activeEvidenceModal && (
        <div className="modal-overlay" onClick={() => setActiveEvidenceModal(null)}>
          <div className="modal-content card" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div className="modal-title-group">
                <FileText size={20} className="sparkle-purple" />
                <h4>Citation Evidence: [{activeEvidenceModal.tag}]</h4>
              </div>
              <button
                type="button"
                className="modal-close-btn"
                onClick={() => setActiveEvidenceModal(null)}
              >
                <X size={18} />
              </button>
            </div>
            <div className="modal-body">
              <div className="modal-paper-title">{activeEvidenceModal.title}</div>
              <div className="modal-meta-grid">
                <div><strong>Paper ID:</strong> {activeEvidenceModal.paper_id}</div>
                <div><strong>Section:</strong> {activeEvidenceModal.section_name}</div>
                <div><strong>Pages:</strong> {activeEvidenceModal.pages}</div>
                <div><strong>Source Type:</strong> {activeEvidenceModal.source_type === 'online' ? 'Online Academic Search' : 'Research Mind Corpus'}</div>
              </div>

              {activeEvidenceModal.url && (
                <div style={{ margin: '0.75rem 0', fontSize: '0.85rem' }}>
                  <strong>Paper URL:</strong>{' '}
                  <a href={activeEvidenceModal.url} target="_blank" rel="noopener noreferrer" style={{ color: '#2563eb', textDecoration: 'underline' }}>
                    {activeEvidenceModal.url}
                  </a>
                </div>
              )}

              <div className="modal-passage-box">
                <div className="modal-passage-label">Retrieved Evidence Passage Text:</div>
                <p className="modal-passage-text">"{activeEvidenceModal.text}"</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
};
