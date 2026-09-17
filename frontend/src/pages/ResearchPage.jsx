import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Sidebar } from '../components/Sidebar';
import { Header } from '../components/Header';
import { AnswerCard } from '../components/AnswerCard';
import { KeySourcesPanel } from '../components/KeySourcesPanel';
import { api } from '../services/api';
import { Sparkles, Send, Paperclip, Cpu, Shield, Sprout, Globe, Heart, Lightbulb, Check, FileText, X, ExternalLink, Loader2 } from 'lucide-react';

const DOMAINS = [
  { id: 'all', label: 'All Domains', icon: Globe },
  { id: 'artificial_intelligence', label: 'AI / Machine Learning', icon: Cpu, prefix: 'AI' },
  { id: 'cybersecurity', label: 'Cyber Security', icon: Shield, prefix: 'CY' },
  { id: 'agriculture', label: 'Agriculture', icon: Sprout, prefix: 'AG' },
  { id: 'climate', label: 'Climate', icon: Globe, prefix: 'CL' },
  { id: 'healthcare', label: 'Healthcare', icon: Heart, prefix: 'HC' },
];

const SAMPLE_PAPERS = [
  { id: 'AI001', label: 'AI001 (AI/ML)', domain: 'artificial_intelligence' },
  { id: 'CY001', label: 'CY001 (Cyber)', domain: 'cybersecurity' },
  { id: 'HC001', label: 'HC001 (Health)', domain: 'healthcare' },
  { id: 'CL001', label: 'CL001 (Climate)', domain: 'climate' },
  { id: 'AG001', label: 'AG001 (Agri)', domain: 'agriculture' },
];

const EXAMPLE_QUESTIONS = [
  { text: 'What methodology did the authors propose?', domain: 'all', icon: Cpu },
  { text: 'What dataset was used for evaluation?', domain: 'all', icon: Cpu },
  { text: 'What were the main quantitative results?', domain: 'all', icon: Globe },
  { text: 'How does the proposed method compare with the baseline?', domain: 'all', icon: Globe },
  { text: 'What limitations did the authors identify?', domain: 'all', icon: Globe },
  { text: 'What problem does this paper address?', domain: 'all', icon: Lightbulb },
];

const getDomainFromPaperId = (pid) => {
  if (!pid) return undefined;
  const upper = pid.toUpperCase().trim();
  if (upper.startsWith('AI')) return 'artificial_intelligence';
  if (upper.startsWith('CY')) return 'cybersecurity';
  if (upper.startsWith('AG')) return 'agriculture';
  if (upper.startsWith('HC')) return 'healthcare';
  if (upper.startsWith('CL')) return 'climate';
  return undefined;
};

export const ResearchPage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [question, setQuestion] = useState('');
  const [selectedDomain, setSelectedDomain] = useState('all');
  const [selectedPaper, setSelectedPaper] = useState(location.state?.selectedPaper || null);
  const [paperIdInput, setPaperIdInput] = useState(location.state?.selectedPaper?.paper_id || '');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [response, setResponse] = useState(null);
  const [attachedFileText, setAttachedFileText] = useState('');
  const [attachedFileName, setAttachedFileName] = useState('');
  const fileInputRef = React.useRef(null);

  // Sync selectedPaper when location state updates
  useEffect(() => {
    if (location.state?.selectedPaper) {
      setSelectedPaper(location.state.selectedPaper);
      setPaperIdInput(location.state.selectedPaper.paper_id || '');
      if (location.state.selectedPaper.domain) {
        setSelectedDomain(location.state.selectedPaper.domain);
      }
    }
  }, [location.state]);

  const handleExampleClick = (ex) => {
    setQuestion(ex.text);
    // If user has not chosen a specific paper or domain, use example's domain
    if (!paperIdInput.trim() && !selectedPaper && ex.domain && ex.domain !== 'all') {
      setSelectedDomain(ex.domain);
    }
    handleAsk(ex.text);
  };

  const handleSamplePaperClick = (sample) => {
    setSelectedPaper(null);
    setPaperIdInput(sample.id);
    setSelectedDomain(sample.domain);
  };

  const handleClearPaperScope = () => {
    setSelectedPaper(null);
    setPaperIdInput('');
    setAttachedFileName('');
    setAttachedFileText('');
  };

  const [uploadingFile, setUploadingFile] = useState(false);
  const [uploadInfo, setUploadInfo] = useState(null);

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setAttachedFileName(file.name);
    setUploadingFile(true);
    setError('');

    try {
      const res = await api.uploadPaper(file);
      setAttachedFileText(res.text || '');
      setAttachedFileName(res.filename || file.name);
      setUploadInfo({
        pages: res.pages_count,
        chars: res.char_count,
        chunks: res.chunks_count,
        headings: res.section_headings ? res.section_headings.length : 0,
      });
      console.log(`[FRONTEND PDF UPLOAD] Successfully extracted ${res.char_count} chars from ${res.filename} (${res.pages_count} pages)`);
    } catch (err) {
      console.warn('[FRONTEND UPLOAD FALLBACK] Backend PDF upload failed, trying local text reader:', err);
      if (file.name.endsWith('.txt') || file.name.endsWith('.md')) {
        const reader = new FileReader();
        reader.onload = (event) => {
          setAttachedFileText(event.target.result || '');
        };
        reader.readAsText(file);
      } else {
        setError(`Failed to extract text from ${file.name}: ${err.message}`);
      }
    } finally {
      setUploadingFile(false);
    }
  };

  const handleAsk = async (qText = question) => {
    const activeQuestion = qText || question;
    if (!activeQuestion.trim()) {
      setError('Please enter a research question.');
      return;
    }

    setError('');
    setLoading(true);
    setResponse(null);

    try {
      let activePaperId = selectedPaper
        ? selectedPaper.paper_id
        : (paperIdInput.trim() ? paperIdInput.trim().toUpperCase().replace(/[^A-Z0-9_-]/g, '') : undefined);

      if (!activePaperId && attachedFileName) {
        activePaperId = attachedFileName.replace(/\.[^/.]+$/, "");
      }

      // If a specific paper ID is set, ensure domain does not conflict with paper prefix
      let activeDomain = selectedDomain && selectedDomain !== 'all' ? selectedDomain : undefined;
      if (activePaperId) {
        const inferredDomain = getDomainFromPaperId(activePaperId);
        if (inferredDomain) {
          activeDomain = inferredDomain;
        }
      }

      const payload = {
        question: activeQuestion.trim(),
        domain: activeDomain,
        paper_id: activePaperId,
        top_k: 10,
        uploaded_paper_text: attachedFileText || undefined,
        uploaded_paper_name: attachedFileName || undefined,
      };

      console.log('[FRONTEND PAYLOAD LOG]', payload);
      const resData = await api.askResearchQuestion(payload);
      if (!resData || (!resData.answer && typeof resData.answer !== 'string')) {
        throw new Error('Received an empty or malformed response from ScholarLens server.');
      }
      setResponse(resData);
    } catch (err) {
      console.error('[FRONTEND QUERY ERROR]', err);
      setError(err.message || 'An error occurred while fetching the answer from Research Mind.');
    } finally {
      setLoading(false);
    }
  };


  const rawPaperId = (selectedPaper?.paper_id || paperIdInput || '').trim().toUpperCase();
  const isPaperModeActive = Boolean(rawPaperId || attachedFileName);
  const activePaperLabel = selectedPaper
    ? `Paper ${selectedPaper.paper_id}${selectedPaper.title ? ` — ${selectedPaper.title}` : ''}`
    : attachedFileName
    ? `Attached File: ${attachedFileName}`
    : rawPaperId
    ? `Paper ${rawPaperId}`
    : null;

  return (
    <div className="app-container">
      <Sidebar />
      <div className="main-content">
        <Header />
        <main className="page-body">
          {/* Hero Section */}
          <div className="hero-section">
            <h1 className="hero-title">
              Ask. <span className="highlight-purple">Discover.</span> Understand. ✨
            </h1>
            <p className="hero-subtitle">
              Get evidence-based answers from millions of research papers
            </p>
          </div>

          {/* Paper Scope Selection Card */}
          <div className="card" style={{ marginBottom: '1.25rem', padding: '1rem 1.25rem', background: isPaperModeActive ? '#f5f3ff' : '#ffffff', border: isPaperModeActive ? '1.5px solid #8b5cf6' : '1px solid #e2e8f0', borderRadius: '12px', boxShadow: '0 2px 8px rgba(0,0,0,0.04)' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.75rem' }}>
              
              {/* Left: Status & Input */}
              <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: '0.75rem', flex: 1 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: isPaperModeActive ? '#6d28d9' : '#64748b' }}>
                  <FileText size={18} />
                  <span style={{ fontWeight: 700, fontSize: '0.875rem' }}>
                    Paper Scope:
                  </span>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  <input
                    type="text"
                    id="paper-id-input"
                    placeholder="Enter Paper ID (e.g. AI001, CY001)"
                    value={paperIdInput}
                    onChange={(e) => {
                      const val = e.target.value.toUpperCase();
                      setPaperIdInput(val);
                      if (selectedPaper && val !== selectedPaper.paper_id) {
                        setSelectedPaper(null);
                      }
                      const dom = getDomainFromPaperId(val);
                      if (dom) setSelectedDomain(dom);
                    }}
                    style={{
                      padding: '0.35rem 0.65rem',
                      borderRadius: '8px',
                      border: isPaperModeActive ? '1.5px solid #8b5cf6' : '1px solid #cbd5e1',
                      fontSize: '0.85rem',
                      fontWeight: 600,
                      width: '210px',
                      background: '#ffffff',
                      color: '#1e1b4b',
                      outline: 'none',
                    }}
                  />

                  {isPaperModeActive && (
                    <button
                      type="button"
                      onClick={handleClearPaperScope}
                      style={{
                        background: '#ffffff',
                        color: '#dc2626',
                        border: '1px solid #fca5a5',
                        padding: '0.35rem 0.65rem',
                        borderRadius: '8px',
                        fontSize: '0.75rem',
                        fontWeight: 700,
                        cursor: 'pointer',
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '0.25rem',
                      }}
                      title="Clear paper scope to search across all corpus papers"
                    >
                      <X size={13} /> Clear Scope
                    </button>
                  )}
                </div>

                {/* Quick Sample Papers */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', flexWrap: 'wrap' }}>
                  <span style={{ fontSize: '0.75rem', color: '#64748b', fontWeight: 500 }}>Quick pick:</span>
                  {SAMPLE_PAPERS.map((sample) => (
                    <button
                      key={sample.id}
                      type="button"
                      onClick={() => handleSamplePaperClick(sample)}
                      style={{
                        background: rawPaperId === sample.id ? '#6d28d9' : '#f1f5f9',
                        color: rawPaperId === sample.id ? '#ffffff' : '#475569',
                        border: 'none',
                        padding: '0.2rem 0.5rem',
                        borderRadius: '6px',
                        fontSize: '0.72rem',
                        fontWeight: 600,
                        cursor: 'pointer',
                        transition: 'all 0.15s ease',
                      }}
                    >
                      {sample.id}
                    </button>
                  ))}
                </div>
              </div>

              {/* Right: Browse Library Button */}
              <button
                type="button"
                onClick={() => navigate('/corpus')}
                style={{
                  background: '#f8fafc',
                  color: '#4f46e5',
                  border: '1px solid #c7d2fe',
                  padding: '0.35rem 0.75rem',
                  borderRadius: '8px',
                  fontSize: '0.8rem',
                  fontWeight: 600,
                  cursor: 'pointer',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '0.35rem',
                }}
              >
                <ExternalLink size={13} /> Browse Paper Library (1,000 Papers)
              </button>
            </div>

            {/* Scope Active Subtext */}
            {isPaperModeActive && (
              <div style={{ marginTop: '0.5rem', paddingTop: '0.5rem', borderTop: '1px solid #ede9fe', fontSize: '0.8rem', color: '#5b21b6', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                <Sparkles size={13} style={{ color: '#7c3aed' }} />
                <span><strong>Active Scope:</strong> Questions will synthesize evidence specifically from <strong>{activePaperLabel}</strong>.</span>
              </div>
            )}
          </div>

          {/* Ask a Research Question Card */}
          <div className="card query-card">
            <div className="query-card-header">
              <span className="query-card-title">
                <Sparkles size={16} className="sparkle-purple" />
                Ask a Research Question &gt;
              </span>
              <span className="domain-label-hint">Choose domain (optional)</span>
            </div>

            {/* Domain Selector Pills */}
            <div className="domain-selector-group">
              {DOMAINS.map((d) => {
                const Icon = d.icon;
                const isActive = selectedDomain === d.id;
                return (
                  <button
                    key={d.id}
                    type="button"
                    className={`domain-pill ${isActive ? 'active' : ''}`}
                    onClick={() => setSelectedDomain(d.id)}
                  >
                    <Icon size={14} />
                    <span>{d.label}</span>
                    {isActive && (
                      <span className="pill-check">
                        <Check size={10} />
                      </span>
                    )}
                  </button>
                );
              })}
            </div>

            {/* Integrated Input Box & Toolbar */}
            <div className="input-box-wrapper">
              <textarea
                className="question-textarea"
                placeholder={isPaperModeActive ? `Ask any question about ${activePaperLabel} (e.g. What methodology did the authors propose?)` : "Ask any scientific or technical research question (e.g. How does RAG improve retrieval accuracy?)"}
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleAsk();
                  }
                }}
                rows={3}
              />
              <div className="input-bottom-toolbar">
                <div className="textarea-hint-prompt">
                  <Sparkles size={14} style={{ color: '#6d28d9' }} />
                  <span>
                    {isPaperModeActive
                      ? `Synthesizing answer from ${activePaperLabel}`
                      : 'Synthesizing answer grounded in 250 indexed scientific papers'}
                  </span>
                </div>

                <div className="input-actions-right">
                  <input
                    type="file"
                    ref={fileInputRef}
                    style={{ display: 'none' }}
                    accept=".pdf,.txt,.md"
                    onChange={handleFileUpload}
                  />
                  <button
                    type="button"
                    className={`btn-attach ${attachedFileName ? 'active' : ''}`}
                    title="Attach paper PDF or text file"
                    onClick={() => fileInputRef.current && fileInputRef.current.click()}
                  >
                    <Paperclip size={14} />
                    <span>{attachedFileName ? attachedFileName.slice(0, 15) + '...' : 'Attach Paper'}</span>
                  </button>
                  <button
                    type="button"
                    className="btn-ask-solid"
                    onClick={() => handleAsk()}
                    disabled={loading}
                  >
                    {loading ? (
                      <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
                        <span className="spinner-sm"></span>
                        <span>Synthesizing...</span>
                      </span>
                    ) : (
                      <>
                        <span>Ask ScholarLens</span>
                        <Send size={14} />
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>

            {error && <div className="error-alert" style={{ marginTop: '0.75rem' }}>{error}</div>}
          </div>

          {/* Loading Indicator with helpful text */}
          {loading && (
            <div className="card" style={{ padding: '2rem', textAlign: 'center', marginBottom: '1.5rem', background: '#faf5ff', border: '1px solid #e9d5ff', borderRadius: '12px' }}>
              <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.75rem', color: '#6d28d9', fontWeight: 700, fontSize: '1.05rem', marginBottom: '0.5rem' }}>
                <Sparkles size={20} className="sparkle-purple" />
                <span>Analyzing Evidence & Generating Answer...</span>
              </div>
              <p style={{ color: '#64748b', fontSize: '0.875rem', margin: 0 }}>
                {isPaperModeActive
                  ? `Filtering chunks from ${activePaperLabel} and cross-verifying citations.`
                  : 'Retrieving top evidence across corpus papers with Reciprocal Rank Fusion.'}
              </p>
            </div>
          )}

          {/* Example Questions Section (Shown if no response yet and not loading) */}
          {!response && !loading && (
            <div className="example-questions-container">
              <p className="example-questions-label">
                <Lightbulb size={16} style={{ color: '#eab308' }} />
                <span>Or try asking one of these example questions:</span>
              </p>
              <div className="example-questions-grid">
                {EXAMPLE_QUESTIONS.map((ex, idx) => {
                  const ExIcon = ex.icon;
                  return (
                    <button
                      key={idx}
                      type="button"
                      className="example-chip"
                      onClick={() => handleExampleClick(ex)}
                    >
                      <ExIcon size={14} style={{ color: '#6d28d9' }} />
                      <span>{ex.text}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {/* Response Container (Answer Card + Key Sources Side-by-Side) */}
          {response && !loading && (
            <div className="results-grid">
              <div className="results-main">
                <AnswerCard response={response} />
              </div>
              <div className="results-sidebar">
                <KeySourcesPanel citations={response.citations} evidence={response.evidence} />
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
};
