import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Sidebar } from '../components/Sidebar';
import { Header } from '../components/Header';
import { AnswerCard } from '../components/AnswerCard';
import { KeySourcesPanel } from '../components/KeySourcesPanel';
import { api } from '../services/api';
import { ArrowLeft, Clock } from 'lucide-react';

export const HistoryDetailPage = () => {
  const { id } = useParams();
  const [detail, setDetail] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    const fetchDetail = async () => {
      try {
        const data = await api.getHistoryItem(id);
        setDetail(data);
      } catch (err) {
        setError(err.message || 'Failed to load history record.');
      } finally {
        setLoading(false);
      }
    };
    fetchDetail();
  }, [id]);

  return (
    <div className="app-container">
      <Sidebar />
      <div className="main-content">
        <Header />
        <main className="page-body">
          <button
            onClick={() => navigate('/history')}
            style={{
              background: 'none',
              border: 'none',
              color: '#6d28d9',
              fontWeight: 600,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.4rem',
              marginBottom: '1.5rem',
            }}
          >
            <ArrowLeft size={18} /> Back to History
          </button>

          {error && <div className="alert-error">{error}</div>}

          {loading ? (
            <div style={{ textAlign: 'center', padding: '3rem', color: '#6d28d9', fontWeight: 600 }}>
              Loading history record...
            </div>
          ) : detail ? (
            <div>
              <div className="card" style={{ marginBottom: '1.75rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#64748b', fontSize: '0.85rem', marginBottom: '0.5rem' }}>
                  <Clock size={16} />
                  <span>{(() => {
                    const rawDate = detail.created_at || detail.timestamp;
                    const d = rawDate ? new Date(rawDate) : null;
                    return d && !isNaN(d.getTime()) ? d.toLocaleString() : 'Recent Query';
                  })()}</span>
                  {detail.domain && (
                    <span style={{ backgroundColor: '#efeafd', color: '#6d28d9', padding: '0.15rem 0.5rem', borderRadius: '9999px', fontWeight: 600 }}>
                      {detail.domain}
                    </span>
                  )}
                </div>
                <h2 style={{ fontSize: '1.5rem', fontWeight: 800, color: '#0f172a' }}>
                  {detail.question}
                </h2>
              </div>

              <div className="results-grid">
                <div className="main-results-col">
                  <AnswerCard response={detail} />
                </div>

                <div className="side-panel-col">
                  <KeySourcesPanel citations={detail.citations} />
                </div>
              </div>
            </div>
          ) : null}
        </main>
      </div>
    </div>
  );
};
