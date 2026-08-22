import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { BookOpen, Search, ShieldCheck, BarChart3, Mail, Lock, ArrowRight } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

export const LoginPage = () => {
  const [usernameOrEmail, setUsernameOrEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    const trimmedInput = usernameOrEmail.trim();
    if (!trimmedInput) {
      setError('Please enter your email address or username.');
      return;
    }

    if (!password) {
      setError('Please enter your password.');
      return;
    }

    setLoading(true);

    try {
      await login(trimmedInput, password);
      navigate('/');
    } catch (err) {
      setError(err.message || 'Failed to sign in. Please check your credentials.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-page-container">
      <div className="auth-split-wrapper">
        {/* Left Branding Area */}
        <div className="auth-left-branding">
          <div className="auth-brand-logo">
            <div className="logo-icon-bg">
              <BookOpen size={24} />
            </div>
            <div className="logo-text">
              <h1 style={{ fontSize: '1.4rem' }}>ScholarLens</h1>
              <p>AI-Powered Research Assistant</p>
            </div>
          </div>

          <h2 className="auth-hero-headline">
            Explore Research.<br />
            <span>Discover</span> Knowledge.
          </h2>
          <p style={{ color: '#64748b', fontSize: '0.95rem', marginBottom: '2.5rem', lineHeight: 1.6 }}>
            Get evidence-based answers from the 1,000-paper genuine PDF research corpus with verifiable citations and provenance.
          </p>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            <div style={{ display: 'flex', gap: '1rem', alignItems: 'flex-start' }}>
              <div style={{ background: '#efeafd', color: '#6d28d9', padding: '0.6rem', borderRadius: '12px' }}>
                <Search size={20} />
              </div>
              <div>
                <h4 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#0f172a' }}>Smart Search</h4>
                <p style={{ fontSize: '0.825rem', color: '#64748b' }}>Hybrid dense vector & lexical retrieval across 1,000 research PDFs (5 Domains).</p>
              </div>
            </div>

            <div style={{ display: 'flex', gap: '1rem', alignItems: 'flex-start' }}>
              <div style={{ background: '#efeafd', color: '#6d28d9', padding: '0.6rem', borderRadius: '12px' }}>
                <ShieldCheck size={20} />
              </div>
              <div>
                <h4 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#0f172a' }}>Evidence Grounded</h4>
                <p style={{ fontSize: '0.825rem', color: '#64748b' }}>Answers backed strictly by verifiable source literature passages.</p>
              </div>
            </div>

            <div style={{ display: 'flex', gap: '1rem', alignItems: 'flex-start' }}>
              <div style={{ background: '#efeafd', color: '#6d28d9', padding: '0.6rem', borderRadius: '12px' }}>
                <BarChart3 size={20} />
              </div>
              <div>
                <h4 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#0f172a' }}>Explainable Results</h4>
                <p style={{ fontSize: '0.825rem', color: '#64748b' }}>Transparent "Why This Answer?" breakdown and source provenance.</p>
              </div>
            </div>
          </div>
        </div>

        {/* Right Form Area */}
        <div className="auth-card-right">
          <h3 style={{ fontSize: '1.75rem', fontWeight: 800, color: '#0f172a', marginBottom: '0.35rem' }}>
            Welcome Back 👋
          </h3>
          <p style={{ color: '#64748b', fontSize: '0.9rem', marginBottom: '2rem' }}>
            Sign in to continue your scientific research journey
          </p>

          {error && <div className="alert-error">{error}</div>}

          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label className="form-label">Username or Email address</label>
              <div style={{ position: 'relative' }}>
                <Mail size={18} style={{ position: 'absolute', left: '12px', top: '12px', color: '#94a3b8' }} />
                <input
                  type="text"
                  className="form-input"
                  style={{ paddingLeft: '2.5rem' }}
                  placeholder="Enter your username or email"
                  value={usernameOrEmail}
                  onChange={(e) => setUsernameOrEmail(e.target.value)}
                  required
                />
              </div>
            </div>

            <div className="form-group">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.35rem' }}>
                <label className="form-label" style={{ marginBottom: 0 }}>Password</label>
                <Link to="/forgot-password" style={{ fontSize: '0.8rem', color: '#6d28d9', textDecoration: 'none', fontWeight: 600 }}>
                  Forgot Password?
                </Link>
              </div>
              <div style={{ position: 'relative' }}>
                <Lock size={18} style={{ position: 'absolute', left: '12px', top: '12px', color: '#94a3b8' }} />
                <input
                  type="password"
                  className="form-input"
                  style={{ paddingLeft: '2.5rem' }}
                  placeholder="Enter your password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                />
              </div>
            </div>

            <button type="submit" className="btn-primary" style={{ width: '100%', justifyContent: 'center', marginTop: '1.5rem', padding: '0.85rem' }} disabled={loading}>
              {loading ? 'Signing in...' : 'Sign In'} <ArrowRight size={18} />
            </button>
          </form>

          <p style={{ textAlign: 'center', marginTop: '2rem', fontSize: '0.9rem', color: '#64748b' }}>
            Don't have an account?{' '}
            <Link to="/register" style={{ color: '#6d28d9', fontWeight: 700, textDecoration: 'none' }}>
              Create Account
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
};
