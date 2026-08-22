import React, { useState } from 'react';
import { NavLink } from 'react-router-dom';
import { Plus, Clock, Bookmark, User, Crown, LogOut, BookOpen, Library, X, Info } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

export const Sidebar = () => {
  const { logout } = useAuth();
  const [showProModal, setShowProModal] = useState(false);

  return (
    <>
      <aside className="sidebar">
        {/* Brand Logo */}
        <NavLink to="/" className="sidebar-logo">
          <div className="logo-icon-bg">
            <BookOpen size={22} />
          </div>
          <div className="logo-text">
            <h1>ScholarLens</h1>
            <p>Your AI Research Companion</p>
          </div>
        </NavLink>

        {/* Primary Action Button (+ New Research) */}
        <NavLink to="/" end className="sidebar-action-btn">
          <Plus size={18} />
          <span>New Research</span>
        </NavLink>

        {/* Main Navigation Items */}
        <nav className="nav-group">
          <NavLink to="/corpus" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
            <Library size={18} />
            <span>Paper Library</span>
          </NavLink>

          <NavLink to="/saved" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
            <Bookmark size={18} />
            <span>Saved Queries</span>
          </NavLink>

          <NavLink to="/history" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
            <Clock size={18} />
            <span>My History</span>
          </NavLink>

          <NavLink to="/profile" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
            <User size={18} />
            <span>Profile</span>
          </NavLink>
        </nav>

        {/* Academic Edition Information Card */}
        <div className="pro-upgrade-card">
          <div className="pro-header">
            <Crown size={18} className="crown-icon" />
            <h4>Academic Edition</h4>
          </div>
          <p>Verified 1,000-paper PDF corpus with evidence grounding & citation tracking.</p>
          <button type="button" className="btn-upgrade" onClick={() => setShowProModal(true)}>
            Corpus Info
          </button>
          <div className="pro-graphic">
            <div className="pro-laptop-illustration">💻 📚</div>
          </div>
        </div>

        <div style={{ marginTop: '1rem', paddingTop: '0.75rem', borderTop: '1px solid #e2e8f0' }}>
          <button onClick={logout} className="nav-item" style={{ background: 'none', border: 'none', width: '100%', cursor: 'pointer' }}>
            <LogOut size={18} />
            <span>Sign Out</span>
          </button>
        </div>
      </aside>

      {/* Pro Features Notice Modal */}
      {showProModal && (
        <div className="modal-backdrop" onClick={() => setShowProModal(false)}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#6d28d9', fontWeight: 800 }}>
                <Crown size={22} />
                <h3 style={{ fontSize: '1.25rem', color: '#0f172a' }}>ScholarLens 1,000-Paper Corpus</h3>
              </div>
              <button type="button" onClick={() => setShowProModal(false)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#64748b' }}>
                <X size={20} />
              </button>
            </div>
            <p style={{ color: '#475569', fontSize: '0.925rem', lineHeight: 1.6, marginBottom: '1.25rem' }}>
              ScholarLens operates over a verified 1,000-paper genuine PDF corpus balanced across 5 academic domains: Artificial Intelligence, Cybersecurity, Agriculture, Climate, and Healthcare.
            </p>
            <p style={{ color: '#64748b', fontSize: '0.85rem', background: '#f8fafc', padding: '0.75rem 1rem', borderRadius: '8px', borderLeft: '3px solid #6d28d9' }}>
              <Info size={14} style={{ display: 'inline', marginRight: '0.35rem' }} />
              Full PDF text extraction, section-aware chunking, ChromaDB dense vector indexing, and BM25 lexical search are 100% active.
            </p>
            <div style={{ textAlign: 'right', marginTop: '1.5rem' }}>
              <button type="button" className="btn-primary" onClick={() => setShowProModal(false)}>
                Got It
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};
