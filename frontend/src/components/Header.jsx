import React, { useState } from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { Home, Clock, Bookmark, HelpCircle, Bell, ChevronDown, User, LogOut, X, CheckCircle, Library } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

export const Header = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [showDropdown, setShowDropdown] = useState(false);
  const [showNotifications, setShowNotifications] = useState(false);

  const displayName = user?.full_name || user?.username || user?.email?.split('@')[0] || 'Researcher';
  const email = user?.email || 'researcher@scholarlens.ai';
  const initial = displayName.charAt(0).toUpperCase();

  return (
    <header className="top-header">
      {/* Top Header Navigation Links */}
      <nav className="header-nav-links">
        <NavLink to="/" end className={({ isActive }) => `header-nav-item ${isActive ? 'active' : ''}`}>
          <Home size={18} />
          <span>Research</span>
        </NavLink>
        <NavLink to="/corpus" className={({ isActive }) => `header-nav-item ${isActive ? 'active' : ''}`}>
          <Library size={18} />
          <span>Paper Library</span>
        </NavLink>
        <NavLink to="/saved" className={({ isActive }) => `header-nav-item ${isActive ? 'active' : ''}`}>
          <Bookmark size={18} />
          <span>Saved</span>
        </NavLink>
        <NavLink to="/history" className={({ isActive }) => `header-nav-item ${isActive ? 'active' : ''}`}>
          <Clock size={18} />
          <span>History</span>
        </NavLink>
        <NavLink to="/help" className={({ isActive }) => `header-nav-item ${isActive ? 'active' : ''}`}>
          <HelpCircle size={18} />
          <span>Help</span>
        </NavLink>
      </nav>

      {/* Right User & Notification Controls */}
      <div className="header-right-controls" style={{ position: 'relative' }}>
        <button
          type="button"
          className="notification-btn"
          title="System Notifications"
          onClick={() => {
            setShowNotifications(!showNotifications);
            setShowDropdown(false);
          }}
        >
          <Bell size={18} />
          <span className="notification-badge">1</span>
        </button>

        {/* User Profile Badge & Dropdown */}
        <div
          className="user-profile-badge"
          onClick={() => {
            setShowDropdown(!showDropdown);
            setShowNotifications(false);
          }}
        >
          <div className="user-avatar">{initial}</div>
          <span className="user-name">{displayName}</span>
          <ChevronDown size={14} style={{ color: '#64748b' }} />
        </div>

        {/* Notifications Popover */}
        {showNotifications && (
          <div className="popover-menu" style={{ right: '140px', width: '300px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
              <span style={{ fontSize: '0.9rem', fontWeight: 800, color: '#0f172a' }}>System Notifications</span>
              <button type="button" onClick={() => setShowNotifications(false)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#64748b' }}>
                <X size={16} />
              </button>
            </div>
            <div style={{ display: 'flex', gap: '0.6rem', alignItems: 'flex-start', background: '#f8fafc', padding: '0.75rem', borderRadius: '8px' }}>
              <CheckCircle size={18} style={{ color: '#16a34a', flexShrink: 0, marginTop: '2px' }} />
              <div>
                <p style={{ fontSize: '0.825rem', fontWeight: 700, color: '#0f172a' }}>1,000 PDF Corpus Active</p>
                <p style={{ fontSize: '0.775rem', color: '#64748b', marginTop: '0.15rem' }}>
                  ChromaDB Dense & BM25 Lexical Indexes Active (1,000 Genuine Research PDFs | 95,881 vector units).
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Profile Dropdown Menu */}
        {showDropdown && (
          <div className="popover-menu" style={{ right: 0, width: '220px' }}>
            <div style={{ padding: '0.5rem 0.75rem', borderBottom: '1px solid #f1f5f9', marginBottom: '0.35rem' }}>
              <p style={{ fontSize: '0.85rem', fontWeight: 700, color: '#0f172a' }}>{displayName}</p>
              <p style={{ fontSize: '0.75rem', color: '#64748b', overflow: 'hidden', textOverflow: 'ellipsis' }}>{email}</p>
            </div>

            <button
              type="button"
              className="popover-item"
              onClick={() => {
                setShowDropdown(false);
                navigate('/profile');
              }}
            >
              <User size={16} />
              <span>Profile Settings</span>
            </button>

            <button
              type="button"
              className="popover-item"
              onClick={() => {
                setShowDropdown(false);
                logout();
              }}
              style={{ color: '#ef4444' }}
            >
              <LogOut size={16} />
              <span>Sign Out</span>
            </button>
          </div>
        )}
      </div>
    </header>
  );
};
