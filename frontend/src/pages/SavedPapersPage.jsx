import React from 'react';
import { Sidebar } from '../components/Sidebar';
import { Header } from '../components/Header';
import { Bookmark, Sparkles } from 'lucide-react';

export const SavedPapersPage = () => {
  return (
    <div className="app-container">
      <Sidebar />
      <div className="main-content">
        <Header />
        <main className="page-body">
          <div className="history-header-row">
            <div>
              <h1 style={{ fontSize: '2rem', fontWeight: 800, color: '#0f172a' }}>Saved Papers</h1>
              <p style={{ color: '#64748b', fontSize: '0.95rem' }}>Access your bookmarked research papers and key sources</p>
            </div>
          </div>

          <div className="card" style={{ textAlign: 'center', padding: '4rem 2rem' }}>
            <div className="logo-icon-bg" style={{ margin: '0 auto 1.25rem auto', width: '56px', height: '56px' }}>
              <Bookmark size={28} />
            </div>
            <h3 style={{ fontSize: '1.25rem', fontWeight: 800, color: '#0f172a', marginBottom: '0.5rem' }}>
              No Saved Papers Yet
            </h3>
            <p style={{ color: '#64748b', fontSize: '0.925rem', maxWidth: '480px', margin: '0 auto', lineHeight: 1.6 }}>
              When you ask research questions, click the external link icon or reference items in your Key Sources panel to save literature passages to your personal library.
            </p>
          </div>
        </main>
      </div>
    </div>
  );
};
