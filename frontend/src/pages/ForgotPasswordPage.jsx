import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { BookOpen, Mail, Lock, Key, ArrowRight, CheckCircle2, ArrowLeft } from 'lucide-react';
import { api } from '../services/api';

export const ForgotPasswordPage = () => {
  const [step, setStep] = useState(1); // 1: Email, 2: Reset Token & New Password
  const [email, setEmail] = useState('');
  const [resetToken, setResetToken] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleRequestToken = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');

    const trimmedEmail = email.trim().toLowerCase();
    const emailRegex = /^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*\.[a-zA-Z]{2,}$/;
    if (!emailRegex.test(trimmedEmail) || trimmedEmail.includes('..') || trimmedEmail.split('@')[0].length < 2) {
      setError('Please enter a valid real email address with a domain (e.g. researcher@university.edu or name@domain.com).');
      return;
    }

    setLoading(true);

    try {
      const res = await api.forgotPassword(trimmedEmail);
      setResetToken(res.reset_token || '');
      setSuccess('Account verified. In this development architecture, your reset token has been generated below.');
      setStep(2);
    } catch (err) {
      setError(err.message || 'No account found with this email address.');
    } finally {
      setLoading(false);
    }
  };

  const handleResetPassword = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');

    if (newPassword !== confirmPassword) {
      setError('Passwords do not match.');
      return;
    }

    if (newPassword.length < 6) {
      setError('Password must be at least 6 characters long.');
      return;
    }

    setLoading(true);

    try {
      await api.resetPassword(email.trim(), resetToken.trim(), newPassword);
      setSuccess('Password updated successfully! Redirecting to login...');
      setTimeout(() => {
        navigate('/login');
      }, 1500);
    } catch (err) {
      setError(err.message || 'Failed to reset password. Please check your token.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-page-container">
      <div className="auth-split-wrapper" style={{ maxWidth: '900px' }}>
        {/* Left Branding Area */}
        <div className="auth-left-branding">
          <div className="auth-brand-logo">
            <div className="logo-icon-bg">
              <BookOpen size={24} />
            </div>
            <div className="logo-text">
              <h1 style={{ fontSize: '1.4rem' }}>ScholarLens</h1>
              <p>Password Recovery</p>
            </div>
          </div>

          <h2 className="auth-hero-headline">
            Reset Your<br />
            <span>Account Password</span>.
          </h2>
          <p style={{ color: '#64748b', fontSize: '0.95rem', marginBottom: '2rem', lineHeight: 1.6 }}>
            Follow the safe password recovery workflow to set a new password and regain access to your research queries.
          </p>

          <Link to="/login" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem', color: '#6d28d9', fontWeight: 700, textDecoration: 'none' }}>
            <ArrowLeft size={16} /> Back to Sign In
          </Link>
        </div>

        {/* Right Form Area */}
        <div className="auth-card-right">
          <h3 style={{ fontSize: '1.75rem', fontWeight: 800, color: '#0f172a', marginBottom: '0.35rem' }}>
            {step === 1 ? 'Forgot Password 🔑' : 'Set New Password 🔒'}
          </h3>
          <p style={{ color: '#64748b', fontSize: '0.9rem', marginBottom: '1.5rem' }}>
            {step === 1
              ? 'Enter your registered email address to verify your account'
              : 'Enter the reset token and your new password below'}
          </p>

          {error && <div className="alert-error">{error}</div>}
          {success && (
            <div className="alert-success" style={{ background: '#f0fdf4', border: '1px solid #bbf7d0', color: '#166534', padding: '0.75rem 1rem', borderRadius: '8px', marginBottom: '1rem', fontSize: '0.875rem' }}>
              <CheckCircle2 size={16} style={{ display: 'inline', marginRight: '0.4rem' }} />
              {success}
            </div>
          )}

          {step === 1 ? (
            <form onSubmit={handleRequestToken}>
              <div className="form-group">
                <label className="form-label">Registered Email Address</label>
                <div style={{ position: 'relative' }}>
                  <Mail size={18} style={{ position: 'absolute', left: '12px', top: '12px', color: '#94a3b8' }} />
                  <input
                    type="email"
                    className="form-input"
                    style={{ paddingLeft: '2.5rem' }}
                    placeholder="Enter your registered email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                  />
                </div>
              </div>

              <button type="submit" className="btn-primary" style={{ width: '100%', justifyContent: 'center', marginTop: '1.5rem', padding: '0.85rem' }} disabled={loading}>
                {loading ? 'Verifying Account...' : 'Verify Email & Continue'} <ArrowRight size={18} />
              </button>
            </form>
          ) : (
            <form onSubmit={handleResetPassword}>
              <div className="form-group">
                <label className="form-label">Reset Token</label>
                <div style={{ position: 'relative' }}>
                  <Key size={18} style={{ position: 'absolute', left: '12px', top: '12px', color: '#94a3b8' }} />
                  <input
                    type="text"
                    className="form-input"
                    style={{ paddingLeft: '2.5rem', fontFamily: 'monospace' }}
                    placeholder="Reset Token"
                    value={resetToken}
                    onChange={(e) => setResetToken(e.target.value)}
                    required
                  />
                </div>
              </div>

              <div className="form-group">
                <label className="form-label">New Password</label>
                <div style={{ position: 'relative' }}>
                  <Lock size={18} style={{ position: 'absolute', left: '12px', top: '12px', color: '#94a3b8' }} />
                  <input
                    type="password"
                    className="form-input"
                    style={{ paddingLeft: '2.5rem' }}
                    placeholder="Minimum 6 characters"
                    value={newPassword}
                    onChange={(e) => setNewPassword(e.target.value)}
                    required
                  />
                </div>
              </div>

              <div className="form-group">
                <label className="form-label">Confirm New Password</label>
                <div style={{ position: 'relative' }}>
                  <Lock size={18} style={{ position: 'absolute', left: '12px', top: '12px', color: '#94a3b8' }} />
                  <input
                    type="password"
                    className="form-input"
                    style={{ paddingLeft: '2.5rem' }}
                    placeholder="Re-enter new password"
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    required
                  />
                </div>
              </div>

              <button type="submit" className="btn-primary" style={{ width: '100%', justifyContent: 'center', marginTop: '1.5rem', padding: '0.85rem' }} disabled={loading}>
                {loading ? 'Updating Password...' : 'Reset Password & Sign In'} <ArrowRight size={18} />
              </button>
            </form>
          )}

          <p style={{ textAlign: 'center', marginTop: '1.5rem', fontSize: '0.9rem', color: '#64748b' }}>
            Remembered your password?{' '}
            <Link to="/login" style={{ color: '#6d28d9', fontWeight: 700, textDecoration: 'none' }}>
              Sign In
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
};
