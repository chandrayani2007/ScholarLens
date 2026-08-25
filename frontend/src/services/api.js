/**
 * Centralized API Service for ScholarLens / Research Mind Frontend
 */

const API_BASE_URL = ''; // Relative path handled by Vite proxy to localhost:8000

export const getToken = () => localStorage.getItem('token');
export const setToken = (token) => localStorage.setItem('token', token);
export const removeToken = () => localStorage.removeItem('token');

const formatErrorDetail = (detail, defaultMsg = 'An error occurred while communicating with the server.') => {
  if (!detail) return defaultMsg;
  if (typeof detail === 'string') {
    return detail.replace(/^Value error,\s*/i, '');
  }
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (typeof item === 'string') return item.replace(/^Value error,\s*/i, '');
        if (item && typeof item === 'object') {
          const msg = item.msg || item.message;
          if (msg) {
            return msg
              .replace(/^Value error,\s*/i, '')
              .replace(/^String should have at least \d+ characters/i, 'Field is too short.')
              .replace(/^value is not a valid email address: /i, 'Invalid email: ');
          }
          return JSON.stringify(item);
        }
        return String(item);
      })
      .join(', ');
  }
  if (typeof detail === 'object') {
    return detail.message || detail.msg || detail.detail || JSON.stringify(detail);
  }
  return String(detail);
};

const request = async (endpoint, options = {}) => {
  const token = getToken();
  const headers = {
    'Content-Type': 'application/json',
    ...(options.headers || {}),
  };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const config = {
    ...options,
    headers,
  };

  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, config);

    let data;
    const contentType = response.headers.get('content-type') || '';

    if (contentType.includes('application/json')) {
      try {
        data = await response.json();
      } catch (e) {
        data = null;
      }
    } else {
      const rawText = await response.text();
      if (!response.ok) {
        if (response.status === 502 || response.status === 503 || response.status === 504) {
          throw new Error('Unable to connect to the ScholarLens backend server. Please ensure the API server is running on port 8000.');
        }
        throw new Error(rawText || `Server error (${response.status}).`);
      }
      data = rawText;
    }

    if (response.status === 401) {
      removeToken();
      const isAuthPage =
        window.location.pathname.includes('/login') ||
        window.location.pathname.includes('/register') ||
        window.location.pathname.includes('/forgot-password');

      const errorMsg = formatErrorDetail(data?.detail, isAuthPage ? 'Incorrect credentials.' : 'Session expired. Please sign in again.');
      
      if (!isAuthPage) {
        window.location.href = '/login';
      }
      throw new Error(errorMsg);
    }

    if (!response.ok) {
      if (response.status === 502 || response.status === 503 || response.status === 504) {
        throw new Error('Unable to connect to the ScholarLens backend server. Please ensure the API server is running on port 8000.');
      }
      const errorMsg = formatErrorDetail(data?.detail, `Request failed with status ${response.status}`);
      throw new Error(errorMsg);
    }

    return data;
  } catch (err) {
    if (err.message && (err.message.includes('Failed to fetch') || err.message.includes('NetworkError') || err.name === 'TypeError')) {
      throw new Error('Unable to reach the ScholarLens backend server. Please check that the server is running on port 8000.');
    }
    throw err;
  }
};

export const api = {
  // Authentication
  register: (payload) =>
    request('/auth/register', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  login: async (usernameOrEmail, password) => {
    if (!usernameOrEmail || !usernameOrEmail.trim()) {
      throw new Error('Please enter your email address or username.');
    }
    if (!password) {
      throw new Error('Please enter your password.');
    }

    const formData = new URLSearchParams();
    formData.append('username', usernameOrEmail.trim());
    formData.append('password', password);

    try {
      const response = await fetch(`${API_BASE_URL}/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: formData.toString(),
      });

      let data;
      const contentType = response.headers.get('content-type') || '';

      if (contentType.includes('application/json')) {
        try {
          data = await response.json();
        } catch (e) {
          data = null;
        }
      } else {
        if (response.status === 502 || response.status === 503 || response.status === 504) {
          throw new Error('Unable to connect to the ScholarLens backend server. Please ensure the API server is running on port 8000.');
        }
        const rawText = await response.text();
        throw new Error(rawText || `Authentication failed with status ${response.status}.`);
      }

      if (!response.ok) {
        if (response.status === 502 || response.status === 503 || response.status === 504) {
          throw new Error('Unable to connect to the ScholarLens backend server. Please ensure the API server is running on port 8000.');
        }
        const errorMsg = formatErrorDetail(data?.detail, 'Invalid username/email or password.');
        throw new Error(errorMsg);
      }

      setToken(data.access_token);
      return data;
    } catch (err) {
      if (err.message && (err.message.includes('Failed to fetch') || err.message.includes('NetworkError') || err.name === 'TypeError')) {
        throw new Error('Unable to reach the ScholarLens backend server. Please check that the server is running on port 8000.');
      }
      throw err;
    }
  },

  getCurrentUser: () => request('/auth/me'),

  forgotPassword: (email) =>
    request('/auth/forgot-password', {
      method: 'POST',
      body: JSON.stringify({ email }),
    }),

  resetPassword: (email, resetToken, newPassword) =>
    request('/auth/reset-password', {
      method: 'POST',
      body: JSON.stringify({ email, reset_token: resetToken, new_password: newPassword }),
    }),

  // User Profile
  getProfile: () => request('/api/profile/me'),
  updateProfile: (profileData) =>
    request('/api/profile/me', {
      method: 'PUT',
      body: JSON.stringify(profileData),
    }),

  // Saved Queries
  listSavedQueries: () => request('/api/saved-queries'),
  saveQuery: (queryData) =>
    request('/api/saved-queries', {
      method: 'POST',
      body: JSON.stringify(queryData),
    }),
  getSavedQuery: (id) => request(`/api/saved-queries/${id}`),
  deleteSavedQuery: (id) =>
    request(`/api/saved-queries/${id}`, {
      method: 'DELETE',
    }),

  // Corpus & Academic Paper Library
  listCorpusPapers: (params = {}) => {
    const query = new URLSearchParams(params).toString();
    return request(`/api/corpus?${query}`);
  },
  getCorpusPaper: (paperId) => request(`/api/corpus/${paperId}`),
  getPaperPdfUrl: (paperId) => `${API_BASE_URL}/api/corpus/${paperId}/pdf`,

  // Research Query Pipeline
  uploadPaper: async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    const token = getToken();
    const headers = {};
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }
    const response = await fetch(`${API_BASE_URL}/research/upload-paper`, {
      method: 'POST',
      headers,
      body: formData,
    });
    if (!response.ok) {
      const errData = await response.json().catch(() => null);
      throw new Error(errData?.detail || `Upload failed with status ${response.status}`);
    }
    return await response.json();
  },

  askResearchQuestion: (payload) =>
    request('/research/query', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  // Query History
  getHistory: (limit = 50, offset = 0) => request(`/history?limit=${limit}&offset=${offset}`),
  getHistoryItem: (queryId) => request(`/history/${queryId}`),

  // Health
  getHealth: () => request('/health'),
  getRetrievalHealth: () => request('/health/retrieval'),
};
