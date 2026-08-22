import React, { useState, useEffect } from 'react';
import { Library, Search, Filter, FileText, ExternalLink, ChevronLeft, ChevronRight, X, Eye } from 'lucide-react';
import { Sidebar } from '../components/Sidebar';
import { Header } from '../components/Header';
import { api } from '../services/api';

const DOMAIN_TABS = [
  { id: 'all', label: 'All Domains (1,000)' },
  { id: 'artificial_intelligence', label: 'Artificial Intelligence (200)' },
  { id: 'cybersecurity', label: 'Cybersecurity (200)' },
  { id: 'agriculture', label: 'Agriculture (200)' },
  { id: 'climate', label: 'Climate (200)' },
  { id: 'healthcare', label: 'Healthcare (200)' },
];

export const CorpusPage = () => {
  const [activeTab, setActiveTab] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [papers, setPapers] = useState([]);
  const [totalCount, setTotalCount] = useState(0);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [loading, setLoading] = useState(true);
  const [selectedPaper, setSelectedPaper] = useState(null);

  const fetchPapers = async () => {
    setLoading(true);
    try {
      const res = await api.listCorpusPapers({
        domain: activeTab,
        search: searchQuery,
        page,
        page_size: 15,
      });
      setPapers(res.papers || []);
      setTotalCount(res.total || 0);
      setTotalPages(res.total_pages || 1);
    } catch (err) {
      console.error('Failed to fetch corpus papers:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPapers();
  }, [activeTab, page]);

  const handleSearchSubmit = (e) => {
    e.preventDefault();
    setPage(1);
    fetchPapers();
  };

  const getDomainBadge = (dom) => {
    const domainColors = {
      artificial_intelligence: { bg: '#efeafd', color: '#6d28d9', label: 'AI' },
      cybersecurity: { bg: '#fee2e2', color: '#b91c1c', label: 'Cybersecurity' },
      agriculture: { bg: '#fef3c7', color: '#b45309', label: 'Agriculture' },
      climate: { bg: '#e0f2fe', color: '#0369a1', label: 'Climate' },
      healthcare: { bg: '#dcfce7', color: '#15803d', label: 'Healthcare' },
    };
    const cfg = domainColors[dom] || { bg: '#f1f5f9', color: '#475569', label: dom };
    return (
      <span style={{ background: cfg.bg, color: cfg.color, padding: '0.2rem 0.6rem', borderRadius: '6px', fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase' }}>
        {cfg.label}
      </span>
    );
  };

  return (
    <div className="app-container">
      <Sidebar />
      <div className="main-content">
        <Header />
        <main className="page-body">
          <div style={{ maxWidth: '1100px', margin: '0 auto' }}>
      {/* Page Title & Intro */}
      <div style={{ marginBottom: '1.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', color: '#6d28d9', marginBottom: '0.25rem' }}>
          <Library size={24} />
          <h2 style={{ fontSize: '1.6rem', fontWeight: 800, color: '#0f172a', margin: 0 }}>
            ScholarLens Academic Paper Library
          </h2>
        </div>
        <p style={{ color: '#64748b', fontSize: '0.925rem' }}>
          Browse, search, and access the verified 1,000-paper genuine PDF research corpus.
        </p>
      </div>

      {/* Domain Tabs & Search Controls */}
      <div className="card" style={{ marginBottom: '1.5rem', padding: '1.25rem' }}>
        {/* Domain Filter Tabs */}
        <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '1.25rem', borderBottom: '1px solid #f1f5f9', pb: '0.75rem' }}>
          {DOMAIN_TABS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => {
                setActiveTab(tab.id);
                setPage(1);
              }}
              style={{
                padding: '0.45rem 0.9rem',
                borderRadius: '8px',
                border: activeTab === tab.id ? '1px solid #6d28d9' : '1px solid #e2e8f0',
                background: activeTab === tab.id ? '#efeafd' : '#ffffff',
                color: activeTab === tab.id ? '#6d28d9' : '#475569',
                fontSize: '0.85rem',
                fontWeight: activeTab === tab.id ? 700 : 500,
                cursor: 'pointer',
                transition: 'all 0.2s ease',
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Search Input Form */}
        <form onSubmit={handleSearchSubmit} style={{ display: 'flex', gap: '0.75rem' }}>
          <div style={{ position: 'relative', flex: 1 }}>
            <Search size={18} style={{ position: 'absolute', left: '12px', top: '12px', color: '#94a3b8' }} />
            <input
              type="text"
              className="form-input"
              style={{ paddingLeft: '2.5rem' }}
              placeholder="Search papers by Title, Author, or Paper ID (e.g. AI051, CY010, RAG)..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
          <button type="submit" className="btn-primary" style={{ padding: '0 1.25rem' }}>
            Search
          </button>
        </form>
      </div>

      {/* Papers Results Table */}
      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        {loading ? (
          <div style={{ padding: '3rem', textAlign: 'center', color: '#64748b' }}>
            Loading corpus paper records...
          </div>
        ) : papers.length === 0 ? (
          <div style={{ padding: '3rem', textAlign: 'center', color: '#64748b' }}>
            No research papers found matching query criteria.
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
              <thead>
                <tr style={{ background: '#f8fafc', borderBottom: '1px solid #e2e8f0', color: '#475569', fontSize: '0.8rem', fontWeight: 700, textTransform: 'uppercase' }}>
                  <th style={{ padding: '0.85rem 1.25rem' }}>ID</th>
                  <th style={{ padding: '0.85rem 1.25rem' }}>Domain</th>
                  <th style={{ padding: '0.85rem 1.25rem' }}>Paper Title</th>
                  <th style={{ padding: '0.85rem 1.25rem' }}>Authors</th>
                  <th style={{ padding: '0.85rem 1.25rem' }}>Year</th>
                  <th style={{ padding: '0.85rem 1.25rem', textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {papers.map((paper, idx) => (
                  <tr key={paper.paper_id} style={{ borderBottom: '1px solid #f1f5f9', fontSize: '0.9rem', background: idx % 2 === 0 ? '#ffffff' : '#f8fafc' }}>
                    <td style={{ padding: '0.85rem 1.25rem', fontWeight: 700, color: '#6d28d9', whiteSpace: 'nowrap' }}>
                      {paper.paper_id}
                    </td>
                    <td style={{ padding: '0.85rem 1.25rem', whiteSpace: 'nowrap' }}>
                      {getDomainBadge(paper.domain)}
                    </td>
                    <td style={{ padding: '0.85rem 1.25rem', fontWeight: 600, color: '#0f172a' }}>
                      {paper.title}
                    </td>
                    <td style={{ padding: '0.85rem 1.25rem', color: '#64748b', maxWidth: '180px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {Array.isArray(paper.authors) ? paper.authors.join(', ') : paper.authors}
                    </td>
                    <td style={{ padding: '0.85rem 1.25rem', color: '#64748b', whiteSpace: 'nowrap' }}>
                      {paper.published_date || '2024'}
                    </td>
                    <td style={{ padding: '0.85rem 1.25rem', textAlign: 'right', whiteSpace: 'nowrap' }}>
                      <button
                        type="button"
                        onClick={() => setSelectedPaper(paper)}
                        style={{ background: '#efeafd', color: '#6d28d9', border: 'none', padding: '0.4rem 0.75rem', borderRadius: '6px', fontSize: '0.8rem', fontWeight: 700, cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}
                      >
                        <Eye size={14} /> View
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Pagination Footer */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0.85rem 1.25rem', borderTop: '1px solid #e2e8f0', background: '#f8fafc' }}>
          <span style={{ fontSize: '0.85rem', color: '#64748b' }}>
            Showing {papers.length} of {totalCount} total research papers (Page {page} of {totalPages})
          </span>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <button
              type="button"
              disabled={page <= 1}
              onClick={() => setPage(page - 1)}
              style={{ background: '#ffffff', border: '1px solid #cbd5e1', padding: '0.4rem 0.75rem', borderRadius: '6px', cursor: page <= 1 ? 'not-allowed' : 'pointer', opacity: page <= 1 ? 0.5 : 1, display: 'flex', alignItems: 'center', gap: '0.25rem' }}
            >
              <ChevronLeft size={16} /> Prev
            </button>
            <button
              type="button"
              disabled={page >= totalPages}
              onClick={() => setPage(page + 1)}
              style={{ background: '#ffffff', border: '1px solid #cbd5e1', padding: '0.4rem 0.75rem', borderRadius: '6px', cursor: page >= totalPages ? 'not-allowed' : 'pointer', opacity: page >= totalPages ? 0.5 : 1, display: 'flex', alignItems: 'center', gap: '0.25rem' }}
            >
              Next <ChevronRight size={16} />
            </button>
          </div>
        </div>
      </div>

      {/* Paper Details & Genuine PDF Viewer Modal */}
      {selectedPaper && (
        <div className="modal-overlay" onClick={() => setSelectedPaper(null)}>
          <div className="modal-content card" style={{ maxWidth: '750px' }} onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div className="modal-title-group">
                <FileText size={20} className="sparkle-purple" />
                <h4>Paper Details: {selectedPaper.paper_id}</h4>
              </div>
              <button type="button" className="modal-close-btn" onClick={() => setSelectedPaper(null)}>
                <X size={18} />
              </button>
            </div>
            <div className="modal-body">
              <h3 style={{ fontSize: '1.2rem', fontWeight: 800, color: '#0f172a', marginBottom: '0.75rem' }}>
                {selectedPaper.title}
              </h3>
              <div className="modal-meta-grid" style={{ marginBottom: '1.25rem' }}>
                <div><strong>Paper ID:</strong> {selectedPaper.paper_id}</div>
                <div><strong>Domain:</strong> {selectedPaper.domain}</div>
                <div><strong>Published Date:</strong> {selectedPaper.published_date || '2024'}</div>
                <div><strong>Source Type:</strong> Genuine PDF Document</div>
              </div>

              <div style={{ marginBottom: '1rem' }}>
                <strong>Authors:</strong>{' '}
                <span style={{ color: '#475569' }}>
                  {Array.isArray(selectedPaper.authors) ? selectedPaper.authors.join(', ') : selectedPaper.authors}
                </span>
              </div>

              {selectedPaper.abstract && (
                <div className="modal-passage-box" style={{ marginBottom: '1.25rem' }}>
                  <div className="modal-passage-label">Abstract:</div>
                  <p className="modal-passage-text">{selectedPaper.abstract}</p>
                </div>
              )}

              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingTop: '1rem', borderTop: '1px solid #e2e8f0' }}>
                {selectedPaper.url ? (
                  <a href={selectedPaper.url} target="_blank" rel="noopener noreferrer" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem', color: '#2563eb', fontWeight: 700, fontSize: '0.9rem', textDecoration: 'none' }}>
                    <ExternalLink size={16} /> Open ArXiv Record ({selectedPaper.arxiv_id || selectedPaper.paper_id})
                  </a>
                ) : <span />}

                <a
                  href={api.getPaperPdfUrl(selectedPaper.paper_id)}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="btn-primary"
                  style={{ textDecoration: 'none' }}
                >
                  <FileText size={16} /> Open Genuine PDF
                </a>
              </div>
            </div>
          </div>
        </div>
      )}
          </div>
        </main>
      </div>
    </div>
  );
};
