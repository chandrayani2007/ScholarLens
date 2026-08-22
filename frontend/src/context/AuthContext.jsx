import React, { createContext, useContext, useState, useEffect } from 'react';
import { api, getToken, removeToken } from '../services/api';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchUser = async () => {
    const token = getToken();
    if (token) {
      try {
        const userData = await api.getCurrentUser();
        setUser(userData);
      } catch (err) {
        removeToken();
        setUser(null);
      }
    } else {
      setUser(null);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchUser();
  }, []);

  const login = async (usernameOrEmail, password) => {
    await api.login(usernameOrEmail, password);
    const userData = await api.getCurrentUser();
    setUser(userData);
    return userData;
  };

  const register = async (registerPayload) => {
    await api.register(registerPayload);
    return await login(registerPayload.email || registerPayload.username, registerPayload.password);
  };

  const updateProfile = async (profileData) => {
    const updated = await api.updateProfile(profileData);
    setUser(updated);
    return updated;
  };

  const logout = () => {
    removeToken();
    setUser(null);
    window.location.href = '/login';
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        loading,
        login,
        register,
        logout,
        updateProfile,
        reloadUser: fetchUser,
        isAuthenticated: !!user,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
