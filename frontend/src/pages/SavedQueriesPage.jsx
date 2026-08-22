import React, { useState, useEffect } from 'react';
import { Bookmark, Search, Trash2, Eye, Calendar, Sparkles, X, FileText } from 'lucide-react';
import { Sidebar } from '../components/Sidebar';
import { Header } from '../components/Header';
import { api } from '../services/api';
import { WhyThisAnswerCard } from '../components/WhyThisAnswerCard';

export const SavedQueriesPage = () => {
  const [savedQueries, setSavedQueries] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(true);
  const [selectedSaved, setSelectedSaved] = useState(null);

  const fetchSavedQueries = async () => {
    setLoading(true);
    try {
      const data = await api.listSavedQueries();
      setSavedQueries(data || []);
    } catch (err) {
      console.error('Failed to fetch saved queries:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSavedQueries();
  }, []);

  const handleDelete = async (id, e) => {
    e.stopPropagation();
    if (!window.confirm('Are you sure you want to delete this saved research query?')) return;
    try {
      await api.deleteSavedQuery(id);
      setSavedQueries(savedQueries.filter((q) => q.id !== id));
      if (selectedSaved?.id === id) {
        setSelectedSaved(null);
      }
    } catch (err) {
      alert(err.message || 'Failed to delete saved query.');
    }
  };

  const filteredQueries = savedQueries.filter((q) => {
    if (!searchQuery.trim()) return true;
    const term = searchQuery.toLowerCase();
    return q.question.toLowerCase().includes(term) || q.answer.toLowerCase().includes(term) || (q.domain && q.domain.toLowerCase().includes(term));
  });

  return (
    <div className="app-container">
      <Sidebar />
      <div className="main-content">
        <Header />
        <main className="page-body">
          <div style={{ maxWidth: '1000px', margin: '0 auto' }}>
      {/* Header */}
      <div style={{ marginBottom: '1.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', color: '#6d28d9', marginBottom: '0.25rem' }}>
          <Bookmark size={24} />
          <h2 style={{ fontSize: '1.6rem', fontWeight: 800, color: '#0f172a', margin: 0 }}>
            Saved Research Queries
          </h2>
        </div>
        <p style={{ color: '#64748b', fontSize: '0.925rem' }}>
          Access your bookmarked research questions, grounded answers, citations, and evidence provenance.
        </p>
      </div>

      {/* Search Bar */}
      <div className="card" style={{ marginBottom: '1.5rem', padding: '1rem 1.25rem' }}>
        <div style={{ position: 'relative' }}>
          <Search size={18} style={{ position: 'absolute', left: '12px', top: '12px', color: '#94a3b8' }} />
          <input
            type="text"
            className="form-input"
            style={{ paddingLeft: '2.5rem' }}
            placeholder="Search saved queries by question text or topic..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
      </div>

      {/* Saved Queries List */}
      {loading ? (
        <div className="card" style={{ padding: '3rem', textAlign: 'center', color: '#64748b' }}>
          Loading saved research queries...
        </div>
      ) : filteredQueries.length === 0 ? (
        <div className="card" style={{ padding: '3rem', textAlign: 'center', color: '#64748b' }}>
          {searchQuery ? 'No saved queries match your search term.' : 'No saved research queries yet. Click "Save Query" on any answer page to bookmark it here.'}
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {filteredQueries.map((item) => (
            <div
              key={item.id}
              className="card"
              onClick={() => setSelectedSaved(item)}
              style={{ cursor: 'pointer', transition: 'all 0.2s ease', borderLeft: '4px solid #6d28d9' }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.5rem' }}>
                <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#0f172a', margin: 0 }}>
                  {item.question}
                </h3>
                <button
                  type="button"
                  onClick={(e) => handleDelete(item.id, e)}
                  style={{ background: 'none', border: 'none', color: '#ef4444', cursor: 'pointer', padding: '0.2rem' }}
                  title="Delete saved query"
                >
                  <Trash2 size={16} />
                </button>
              </div>

              <p style={{ color: '#475569', fontSize: '0.9rem', lineHeight: 1.5, marginBottom: '0.75rem', display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
                {item.answer}
              </p>

              <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', fontSize: '0.8rem', color: '#64748b' }}>
                <span style={{ background: '#efeafd', color: '#6d28d9', padding: '0.15rem 0.5rem', borderRadius: '6px', fontWeight: 700, textTransform: 'uppercase' }}>
                  {item.domain || 'General'}
                </span>
                <span style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                  <Calendar size={14} /> {new Date(item.created_at).toLocaleDateString()}
                </span>
                <span style={{ marginLeft: 'auto', color: '#2563eb', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                  <Eye size={14} /> Open Saved Answer
                </span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Saved Answer Full View Modal */}
      {selectedSaved && (
        <div className="modal-overlay" onClick={() => setSelectedSaved(null)}>
          <div className="modal-content card" style={{ maxWidth: '850px' }} onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div className="modal-title-group">
                <Bookmark size={20} className="sparkle-purple" />
                <h4>Saved Research Query Details</h4>
              </div>
              <button type="button" className="modal-close-btn" onClick={() => setSelectedSaved(null)}>
                <X size={18} />
              </button>
            </div>
            <div className="modal-body">
              <h3 style={{ fontSize: '1.25rem', fontWeight: 800, color: '#0f172a', marginBottom: '0.5rem' }}>
                {selectedSaved.question}
              </h3>
              <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center', marginBottom: '1.25rem', fontSize: '0.85rem' }}>
                <span style={{ background: '#efeafd', color: '#6d28d9', padding: '0.2rem 0.6rem', borderRadius: '6px', fontWeight: 700, textTransform: 'uppercase' }}>
                  {selectedSaved.domain || 'General'}
                </span>
                <span style={{ color: '#64748b' }}>
                  Saved on {new Date(selectedSaved.created_at).toLocaleString()}
                </span>
              </div>

              <div style={{ background: '#ffffff', padding: '1rem', borderRadius: '8px', border: '1px solid #e2e8f0', marginBottom: '1.25rem', lineHeight: 1.7, fontSize: '0.95rem', color: '#334155' }}>
                {selectedSaved.answer}
              </div>

              {selectedSaved.why_this_answer_json && (
                <div style={{ marginBottom: '1.25rem' }}>
                  <WhyThisAnswerCard whyThisAnswer={JSON.parse(selectedSaved.why_this_answer_json)} />
                </div>
              )}

              <div style={{ textAlign: 'right', paddingTop: '1rem', borderTop: '1px solid #e2e8f0' }}>
                <button type="button" className="btn-primary" onClick={() => setSelectedSaved(null)}>
                  Close
                </button>
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
