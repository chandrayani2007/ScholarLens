import React, { useState } from 'react';
import { Lightbulb, ChevronDown, ChevronUp, ShieldCheck } from 'lucide-react';

export const WhyThisAnswerCard = ({ whyThisAnswer }) => {
  const [expanded, setExpanded] = useState(false);

  if (!whyThisAnswer) return null;

  const {
    contributing_papers = [],
    contributing_sections = [],
    evidence_passages = [],
    multi_paper_support = false,
    evidence_strength = 'Moderate',
    explanation_summary = 'Answer is grounded in retrieved scientific literature evidence.',
    bullet_points = [],
    source_type = 'Research Mind Corpus',
    domain_scope = 'All Domains',
  } = whyThisAnswer;

  const strengthColor =
    evidence_strength === 'Excellent' || evidence_strength === 'Strong'
      ? '#15803d'
      : evidence_strength === 'High'
      ? '#16a34a'
      : evidence_strength === 'Moderate'
      ? '#6d28d9'
      : evidence_strength === 'Low'
      ? '#b45309'
      : '#dc2626';

  return (
    <div className="why-answer-card">
      <div className="why-answer-content" style={{ flex: 1 }}>
        <h4 className="why-title">
          <ShieldCheck size={18} style={{ color: strengthColor }} />
          Why This Answer?
        </h4>

        <div style={{ fontSize: '0.875rem', marginTop: '0.4rem', color: '#334155', display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
          <span>Evidence Strength: </span>
          <strong style={{ color: strengthColor, fontWeight: 700 }}>{evidence_strength}</strong>
          <span style={{ background: source_type.includes('Online') ? '#eff6ff' : '#efeafd', color: source_type.includes('Online') ? '#1d4ed8' : '#6d28d9', fontSize: '0.75rem', fontWeight: 600, padding: '0.15rem 0.55rem', borderRadius: '4px' }}>
            Source: {source_type}
          </span>
          <span style={{ background: '#f8fafc', color: '#475569', fontSize: '0.75rem', fontWeight: 600, padding: '0.15rem 0.55rem', borderRadius: '4px', border: '1px solid #cbd5e1' }}>
            Scope: {domain_scope}
          </span>
          {multi_paper_support && (
            <span style={{ background: '#f0fdf4', color: '#16a34a', fontSize: '0.75rem', fontWeight: 600, padding: '0.15rem 0.55rem', borderRadius: '4px' }}>
              Multi-paper consensus
            </span>
          )}
        </div>

        {/* Pointwise Bullet List */}
        {bullet_points && bullet_points.length > 0 ? (
          <ul style={{ margin: '0.6rem 0 0 0', paddingLeft: '0.2rem', listStyleType: 'none', fontSize: '0.875rem', color: '#334155', lineHeight: 1.55 }}>
            {bullet_points.map((pt, idx) => (
              <li key={idx} style={{ marginBottom: '0.25rem' }}>
                {pt}
              </li>
            ))}
          </ul>
        ) : (
          <p style={{ fontSize: '0.875rem', color: '#475569', marginTop: '0.4rem', lineHeight: 1.5 }}>
            {explanation_summary}
          </p>
        )}

        {expanded && (
          <div className="why-expanded-details" style={{ marginTop: '0.75rem', paddingTop: '0.75rem', borderTop: '1px solid #e2e8f0', fontSize: '0.825rem', color: '#475569' }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
              <div><strong>Domain Scope:</strong> {domain_scope}</div>
              <div><strong>Supporting Papers:</strong> {contributing_papers.join(', ') || 'N/A'}</div>
              <div><strong>Supporting Sections:</strong> {contributing_sections.join(', ') || 'N/A'}</div>
              <div><strong>Cited Evidence:</strong> {evidence_passages.join(', ') || 'None'}</div>
              <div><strong>Claims Checked:</strong> {whyThisAnswer.claims_checked ?? 'N/A'}</div>
              <div><strong>Supported Claims:</strong> {whyThisAnswer.supported_claims ?? 'N/A'}</div>
              <div><strong>Partially Supported:</strong> {whyThisAnswer.partially_supported_claims ?? 0}</div>
              <div><strong>Unsupported Claims:</strong> {whyThisAnswer.unsupported_claims ?? 0}</div>
              <div><strong>Multi-Paper Support:</strong> {multi_paper_support ? 'Yes' : 'No'}</div>
              <div><strong>Source Platform:</strong> {source_type}</div>
            </div>
          </div>
        )}
      </div>

      <button
        type="button"
        className="explain-answer-btn"
        onClick={() => setExpanded(!expanded)}
      >
        <Lightbulb size={16} />
        <span>{expanded ? 'Less Details' : 'Explain Answer'}</span>
        {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </button>
    </div>
  );
};
