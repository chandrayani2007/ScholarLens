import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Sidebar } from '../components/Sidebar';
import { Header } from '../components/Header';
import { api } from '../services/api';
import { Search, Clock, ChevronRight, Sparkles, BookOpen } from 'lucide-react';

export const HistoryPage = () => {
  const [historyItems, setHistoryItems] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const data = await api.getHistory(50, 0);
        setHistoryItems(data);
      } catch (err) {
        setError(err.message || 'Failed to load research history.');
      } finally {
        setLoading(false);
      }
    };
    fetchHistory();
  }, []);

  const filteredItems = historyItems.filter((item) =>
    item.question.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="app-container">
      <Sidebar />
      <div className="main-content">
        <Header />
        <main className="page-body">
          <div className="history-header-row">
            <div>
              <h1 style={{ fontSize: '2rem', fontWeight: 800, color: '#0f172a' }}>History</h1>
              <p style={{ color: '#64748b', fontSize: '0.95rem' }}>View and revisit your past research queries</p>
            </div>

            <div style={{ position: 'relative', width: '320px' }}>
              <Search size={18} style={{ position: 'absolute', left: '14px', top: '12px', color: '#94a3b8' }} />
              <input
                type="text"
                className="form-input"
                style={{ paddingLeft: '2.5rem', borderRadius: '9999px', background: '#ffffff' }}
                placeholder="Search your history..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
          </div>

          {error && <div className="alert-error">{error}</div>}

          {loading ? (
            <div style={{ textAlign: 'center', padding: '3rem', color: '#6d28d9', fontWeight: 600 }}>
              Loading history...
            </div>
          ) : filteredItems.length === 0 ? (
            <div className="card" style={{ textAlign: 'center', padding: '4rem 2rem' }}>
              <Clock size={40} style={{ color: '#94a3b8', marginBottom: '1rem' }} />
              <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#0f172a' }}>No Research History Found</h3>
              <p style={{ color: '#64748b', fontSize: '0.9rem', marginTop: '0.35rem' }}>
                {searchTerm ? 'No history matching your search query.' : 'Ask your first research question to start building your query history.'}
              </p>
            </div>
          ) : (
            <div>
              {filteredItems.map((item) => {
                const rawDate = item.created_at || item.timestamp;
                const d = rawDate ? new Date(rawDate) : null;
                const dateStr = d && !isNaN(d.getTime()) ? d.toLocaleDateString('en-US', {
                  month: 'short',
                  day: 'numeric',
                  year: 'numeric',
                  hour: '2-digit',
                  minute: '2-digit',
                }) : 'Recent Query';

                return (
                  <div
                    key={item.id}
                    className="history-card"
                    onClick={() => navigate(`/history/${item.id}`)}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem', flex: 1 }}>
                      <div className="history-icon-pill">
                        <BookOpen size={20} />
                      </div>
                      <div className="history-card-info" style={{ flex: 1 }}>
                        <h4>{item.question}</h4>
                        <p>{dateStr}</p>
                        <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.5rem' }}>
                          {item.domain && (
                            <span
                              style={{
                                backgroundColor: '#efeafd',
                                color: '#6d28d9',
                                fontSize: '0.75rem',
                                fontWeight: 600,
                                padding: '0.2rem 0.6rem',
                                borderRadius: '9999px',
                              }}
                            >
                              {item.domain.replace('_', ' ')}
                            </span>
                          )}
                          <span
                            style={{
                              backgroundColor: '#f1f5f9',
                              color: '#475569',
                              fontSize: '0.75rem',
                              fontWeight: 600,
                              padding: '0.2rem 0.6rem',
                              borderRadius: '9999px',
                            }}
                          >
                            Confidence: {item.confidence}
                          </span>
                        </div>
                      </div>
                    </div>

                    <ChevronRight size={20} style={{ color: '#94a3b8' }} />
                  </div>
                );
              })}
            </div>
          )}
        </main>
      </div>
    </div>
  );
};
