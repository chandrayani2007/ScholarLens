import React from 'react';
import { Sidebar } from '../components/Sidebar';
import { Header } from '../components/Header';
import { HelpCircle, Search, ShieldCheck, CheckCircle2, FileText, Sparkles } from 'lucide-react';

export const HelpPage = () => {
  return (
    <div className="app-container">
      <Sidebar />
      <div className="main-content">
        <Header />
        <main className="page-body">
          <div className="history-header-row" style={{ marginBottom: '1.5rem' }}>
            <div>
              <h1 style={{ fontSize: '2rem', fontWeight: 800, color: '#0f172a' }}>Platform Guide & Help Center</h1>
              <p style={{ color: '#64748b', fontSize: '0.95rem' }}>Learn how to search the 250-paper corpus with verifiable citations and provenance</p>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            <div className="card">
              <h3 style={{ fontSize: '1.15rem', fontWeight: 800, color: '#0f172a', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Search size={20} style={{ color: '#6d28d9' }} />
                How Search & Retrieval Works
              </h3>
              <p style={{ color: '#475569', fontSize: '0.925rem', lineHeight: 1.6 }}>
                ScholarLens runs a hybrid retrieval pipeline combining **ChromaDB dense vector search** (using BGE-small-en-v1.5 embeddings) and **BM25 lexical retrieval**. Candidate documents are fused using **Reciprocal Rank Fusion (RRF $k=60$)** and filtered to eliminate non-informational sections (Header, References, Acknowledgements).
              </p>
            </div>

            <div className="card">
              <h3 style={{ fontSize: '1.15rem', fontWeight: 800, color: '#0f172a', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <CheckCircle2 size={20} style={{ color: '#6d28d9' }} />
                Clickable Evidence Citations ([E1], [E2])
              </h3>
              <p style={{ color: '#475569', fontSize: '0.925rem', lineHeight: 1.6 }}>
                Every generated answer includes clickable citation badges like <span className="citation-badge-clickable">[E1]</span>. Clicking any citation tag opens an evidence modal showing the exact paper ID, section name, page range, and full quote from the source literature.
              </p>
            </div>

            <div className="card">
              <h3 style={{ fontSize: '1.15rem', fontWeight: 800, color: '#0f172a', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <ShieldCheck size={20} style={{ color: '#6d28d9' }} />
                Why This Answer & Grounding Safeguards
              </h3>
              <p style={{ color: '#475569', fontSize: '0.925rem', lineHeight: 1.6 }}>
                Our citation validator inspects every generated statement. If insufficient evidence exists in the 250-paper corpus for your query, ScholarLens safely warns: *"Insufficient evidence was found in the current Research Mind corpus to answer this question reliably."*
              </p>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
};
