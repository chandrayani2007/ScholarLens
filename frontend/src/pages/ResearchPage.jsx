import React, { useState } from 'react';
import { Sidebar } from '../components/Sidebar';
import { Header } from '../components/Header';
import { AnswerCard } from '../components/AnswerCard';
import { KeySourcesPanel } from '../components/KeySourcesPanel';
import { api } from '../services/api';
import { Sparkles, Send, Paperclip, Cpu, Shield, Sprout, Globe, Heart, Lightbulb, Check } from 'lucide-react';

const DOMAINS = [
  { id: 'all', label: 'All Domains', icon: Globe },
  { id: 'artificial_intelligence', label: 'AI / Machine Learning', icon: Cpu },
  { id: 'cybersecurity', label: 'Cyber Security', icon: Shield },
  { id: 'agriculture', label: 'Agriculture', icon: Sprout },
  { id: 'climate', label: 'Climate', icon: Globe },
  { id: 'healthcare', label: 'Healthcare', icon: Heart },
];

const EXAMPLE_QUESTIONS = [
  { domain: 'artificial_intelligence', icon: Cpu, text: 'Latest advances in large language models?' },
  { domain: 'cybersecurity', icon: Shield, text: 'Cyber attack detection using deep learning?' },
  { domain: 'agriculture', icon: Sprout, text: 'Smart irrigation techniques for crop yield?' },
  { domain: 'climate', icon: Globe, text: 'Impact of climate change on biodiversity?' },
  { domain: 'healthcare', icon: Heart, text: 'AI in early disease diagnosis?' },
];

export const ResearchPage = () => {
  const [question, setQuestion] = useState('');
  const [selectedDomain, setSelectedDomain] = useState('all');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [response, setResponse] = useState(null);

  const handleAsk = async (qText = question, domainVal = selectedDomain) => {
    const activeQuestion = qText || question;
    if (!activeQuestion.trim()) {
      setError('Please enter a research question.');
      return;
    }

    setError('');
    setLoading(true);
    setResponse(null);

    try {
      const activeDomain = domainVal && domainVal !== 'all' ? domainVal : undefined;
      const payload = {
        question: activeQuestion.trim(),
        domain: activeDomain,
        top_k: 10,
      };
      const resData = await api.askResearchQuestion(payload);
      setResponse(resData);
    } catch (err) {
      setError(err.message || 'An error occurred while fetching the answer from Research Mind.');
    } finally {
      setLoading(false);
    }
  };

  const handleExampleClick = (ex) => {
    setQuestion(ex.text);
    setSelectedDomain(ex.domain);
    handleAsk(ex.text, ex.domain);
  };

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
                placeholder="Ask any scientific or technical research question (e.g. How does RAG improve retrieval accuracy?)"
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
                  <span>Synthesizing answer grounded in 250 indexed scientific papers</span>
                </div>
                <div className="input-actions-right">
                  <button type="button" className="btn-attach" title="Attach reference">
                    <Paperclip size={14} />
                    <span>Attach</span>
                  </button>
                  <button
                    type="button"
                    className="btn-ask-solid"
                    onClick={() => handleAsk()}
                    disabled={loading}
                  >
                    {loading ? (
                      <span className="spinner-sm"></span>
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

            {error && <div className="error-alert">{error}</div>}
          </div>

          {/* Example Questions Section (Shown if no response yet) */}
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
          {response && (
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
