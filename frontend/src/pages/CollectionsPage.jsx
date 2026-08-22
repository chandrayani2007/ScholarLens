import React from 'react';
import { Sidebar } from '../components/Sidebar';
import { Header } from '../components/Header';
import { Folder, Sparkles, Layers } from 'lucide-react';

export const CollectionsPage = () => {
  return (
    <div className="app-container">
      <Sidebar />
      <div className="main-content">
        <Header />
        <main className="page-body">
          <div className="history-header-row">
            <div>
              <h1 style={{ fontSize: '2rem', fontWeight: 800, color: '#0f172a' }}>Collections</h1>
              <p style={{ color: '#64748b', fontSize: '0.95rem' }}>Organize research topics across AI, Cybersecurity, Agriculture, Climate, and Healthcare</p>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '1.25rem' }}>
            <div className="card" style={{ marginBottom: 0 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.75rem' }}>
                <div style={{ background: '#efeafd', color: '#6d28d9', padding: '0.6rem', borderRadius: '12px' }}>
                  <Layers size={22} />
                </div>
                <div>
                  <h4 style={{ fontSize: '1rem', fontWeight: 700, color: '#0f172a' }}>Default Research Mind Collection</h4>
                  <p style={{ fontSize: '0.8rem', color: '#64748b' }}>250 Research Papers • 5 Canonical Domains</p>
                </div>
              </div>
              <p style={{ fontSize: '0.85rem', color: '#475569', lineHeight: 1.5 }}>
                Active scientific corpus spanning Artificial Intelligence, Cybersecurity, Smart Agriculture, Healthcare Diagnostics, and Climate Prediction.
              </p>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
};
