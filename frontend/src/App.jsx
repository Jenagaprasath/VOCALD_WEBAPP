import React, { useState, useEffect } from 'react';
import { Upload, Search, Clock, Users, ChevronRight, Edit2, Save, X, Loader, Database, User, Activity } from 'lucide-react';

// FIXED: Auto-detect API URL for any environment
const getApiBase = () => {
  const hostname = window.location.hostname;
  const protocol = window.location.protocol;
  
  // Local development
  if (hostname === 'localhost' || hostname === '127.0.0.1') {
    return 'http://localhost:5000/api';
  }
  
  // GitHub Codespaces - replace frontend port with backend port
  if (hostname.includes('app.github.dev')) {
    // Extract base URL and replace port number
    const backendHost = hostname.replace(/-\d+\.app\.github\.dev/, '-5000.app.github.dev');
    return `${protocol}//${backendHost}/api`;
  }
  
  // Production fallback
  return `${protocol}//${hostname}/api`;
};

const API_BASE = getApiBase();

export default function VocaldApp() {
  const [view, setView] = useState('logs');
  const [logs, setLogs] = useState([]);
  const [selectedLog, setSelectedLog] = useState(null);
  const [search, setSearch] = useState('');
  const [uploading, setUploading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [editingName, setEditingName] = useState(null);
  const [editValue, setEditValue] = useState('');
  const [voiceProfiles, setVoiceProfiles] = useState([]);
  const [connectionStatus, setConnectionStatus] = useState('checking');

  useEffect(() => {
    console.log('🔗 API Base URL:', API_BASE);
    checkConnection();
    fetchLogs();
  }, []);

  const checkConnection = async () => {
    try {
      const response = await fetch(`${API_BASE}/recordings`, { timeout: 3000 });
      if (response.ok) {
        setConnectionStatus('connected');
      } else {
        setConnectionStatus('error');
      }
    } catch (error) {
      setConnectionStatus('error');
      console.error('Connection check failed:', error);
    }
  };

  const fetchLogs = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/recordings`);
      if (!response.ok) throw new Error('Failed to fetch');
      const data = await response.json();
      setLogs(data.recordings || []);
      setConnectionStatus('connected');
    } catch (error) {
      console.error('Error fetching logs:', error);
      setConnectionStatus('error');
      alert(`❌ Failed to connect to backend!\n\n🔗 Trying to connect to: ${API_BASE}\n\n✅ Make sure Flask is running:\n   cd backend\n   python app.py`);
    } finally {
      setLoading(false);
    }
  };

  const fetchVoiceProfiles = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/voice-profiles`);
      const data = await response.json();
      setVoiceProfiles(data.profiles || []);
    } catch (error) {
      console.error('Error fetching voice profiles:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (e) => {
    const files = Array.from(e.target.files);
    if (files.length === 0) return;

    setUploading(true);
    const formData = new FormData();
    
    files.forEach(file => {
      formData.append('files', file);
    });

    try {
      const response = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) throw new Error('Upload failed');
      await response.json();
      await fetchLogs();
    } catch (error) {
      console.error('Error uploading files:', error);
      alert(`❌ Upload failed!\n\n🔗 Backend URL: ${API_BASE}\n\n✅ Make sure Flask is running on port 5000`);
    } finally {
      setUploading(false);
    }
  };

  const viewDetails = async (logId) => {
    try {
      const response = await fetch(`${API_BASE}/recordings/${logId}`);
      const data = await response.json();
      setSelectedLog(data);
      setView('details');
    } catch (error) {
      console.error('Error fetching details:', error);
    }
  };

  const updateSpeakerName = async (speakerIndex, newName) => {
    if (!selectedLog) return;
    
    try {
      const response = await fetch(`${API_BASE}/recordings/${selectedLog.id}/speakers/${speakerIndex}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: newName }),
      });
      
      const data = await response.json();
      setSelectedLog(data);
      setEditingName(null);
      setEditValue('');
      await fetchLogs();
    } catch (error) {
      console.error('Error updating speaker name:', error);
    }
  };

  const filteredLogs = logs.filter(log =>
    log.filename.toLowerCase().includes(search.toLowerCase())
  );

  // DATABASE VIEW
  if (view === 'database') {
    return (
      <div style={{
        minHeight: '100vh',
        background: 'linear-gradient(to bottom right, #eff6ff, #e0e7ff)',
        padding: '24px'
      }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
          <div style={{ display: 'flex', gap: '16px', marginBottom: '24px' }}>
            <button
              onClick={() => setView('logs')}
              style={{
                padding: '12px 24px',
                background: 'white',
                borderRadius: '8px',
                border: '2px solid #e5e7eb',
                cursor: 'pointer',
                fontWeight: '600',
                color: '#374151'
              }}
            >
              📋 Recordings
            </button>
            <button
              style={{
                padding: '12px 24px',
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                borderRadius: '8px',
                border: 'none',
                fontWeight: '600',
                color: 'white'
              }}
            >
              🗄️ Database
            </button>
          </div>

          <div style={{
            background: 'white',
            borderRadius: '12px',
            boxShadow: '0 10px 15px rgba(0,0,0,0.1)',
            padding: '24px',
            marginBottom: '24px'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
              <h2 style={{ fontSize: '24px', fontWeight: 'bold', color: '#312e81', display: 'flex', alignItems: 'center', gap: '12px' }}>
                <Database size={28} /> Voice Profiles Database
              </h2>
              <button
                onClick={fetchVoiceProfiles}
                style={{
                  padding: '10px 20px',
                  background: '#4f46e5',
                  color: 'white',
                  borderRadius: '8px',
                  border: 'none',
                  cursor: 'pointer',
                  fontWeight: '600',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px'
                }}
              >
                <Activity size={18} />
                Refresh
              </button>
            </div>

            <div style={{ marginBottom: '24px', padding: '16px', background: '#f0f9ff', borderRadius: '8px', border: '2px solid #bae6fd' }}>
              <h3 style={{ fontSize: '16px', fontWeight: '600', color: '#0c4a6e', marginBottom: '8px' }}>
                📊 Database Statistics
              </h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px', marginTop: '12px' }}>
                <div style={{ textAlign: 'center', padding: '12px', background: 'white', borderRadius: '8px' }}>
                  <div style={{ fontSize: '28px', fontWeight: 'bold', color: '#4f46e5' }}>{voiceProfiles.length}</div>
                  <div style={{ fontSize: '14px', color: '#6b7280' }}>Total Voice Profiles</div>
                </div>
                <div style={{ textAlign: 'center', padding: '12px', background: 'white', borderRadius: '8px' }}>
                  <div style={{ fontSize: '28px', fontWeight: 'bold', color: '#10b981' }}>
                    {voiceProfiles.reduce((sum, p) => sum + p.total_recordings, 0)}
                  </div>
                  <div style={{ fontSize: '14px', color: '#6b7280' }}>Total Recordings</div>
                </div>
                <div style={{ textAlign: 'center', padding: '12px', background: 'white', borderRadius: '8px' }}>
                  <div style={{ fontSize: '28px', fontWeight: 'bold', color: '#f59e0b' }}>256D</div>
                  <div style={{ fontSize: '14px', color: '#6b7280' }}>Embedding Size</div>
                </div>
              </div>
            </div>

            <h3 style={{ fontSize: '18px', fontWeight: '600', color: '#374151', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <User size={20} /> Stored Voice Profiles ({voiceProfiles.length})
            </h3>

            {loading ? (
              <div style={{ textAlign: 'center', padding: '40px' }}>
                <Loader style={{ animation: 'spin 1s linear infinite', color: '#4f46e5' }} size={40} />
              </div>
            ) : voiceProfiles.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '40px', color: '#9ca3af' }}>
                <Database size={48} style={{ margin: '0 auto 16px', opacity: 0.3 }} />
                <p>No voice profiles yet. Upload audio files to create profiles!</p>
              </div>
            ) : (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ background: '#f9fafb', borderBottom: '2px solid #e5e7eb' }}>
                      <th style={{ padding: '12px', textAlign: 'left', fontWeight: '600', color: '#374151' }}>ID</th>
                      <th style={{ padding: '12px', textAlign: 'left', fontWeight: '600', color: '#374151' }}>Speaker Name</th>
                      <th style={{ padding: '12px', textAlign: 'left', fontWeight: '600', color: '#374151' }}>First Seen</th>
                      <th style={{ padding: '12px', textAlign: 'left', fontWeight: '600', color: '#374151' }}>Last Seen</th>
                      <th style={{ padding: '12px', textAlign: 'left', fontWeight: '600', color: '#374151' }}>Recordings</th>
                      <th style={{ padding: '12px', textAlign: 'left', fontWeight: '600', color: '#374151' }}>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {voiceProfiles.map((profile) => (
                      <tr key={profile.id} style={{ borderBottom: '1px solid #e5e7eb' }}>
                        <td style={{ padding: '12px', color: '#6366f1', fontWeight: '600' }}>#{profile.id}</td>
                        <td style={{ padding: '12px', fontWeight: '500', color: '#1f2937' }}>{profile.name}</td>
                        <td style={{ padding: '12px', color: '#6b7280', fontSize: '14px' }}>
                          {new Date(profile.first_seen).toLocaleDateString('en-IN')}
                        </td>
                        <td style={{ padding: '12px', color: '#6b7280', fontSize: '14px' }}>
                          {new Date(profile.last_seen).toLocaleDateString('en-IN')}
                        </td>
                        <td style={{ padding: '12px' }}>
                          <span style={{
                            padding: '4px 12px',
                            background: '#dbeafe',
                            color: '#1e40af',
                            borderRadius: '12px',
                            fontSize: '14px',
                            fontWeight: '600'
                          }}>
                            {profile.total_recordings}
                          </span>
                        </td>
                        <td style={{ padding: '12px' }}>
                          <span style={{
                            padding: '4px 12px',
                            background: '#d1fae5',
                            color: '#065f46',
                            borderRadius: '12px',
                            fontSize: '12px',
                            fontWeight: '600'
                          }}>
                            ✓ Active
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          <div style={{
            background: 'white',
            borderRadius: '12px',
            boxShadow: '0 10px 15px rgba(0,0,0,0.1)',
            padding: '24px'
          }}>
            <h3 style={{ fontSize: '18px', fontWeight: '600', color: '#374151', marginBottom: '12px' }}>
              🔬 How Voice Embeddings Work
            </h3>
            <ul style={{ color: '#6b7280', lineHeight: '1.8', paddingLeft: '24px' }}>
              <li>Each voice is converted into a <strong>256-dimensional vector</strong> (voice fingerprint)</li>
              <li>Embeddings are stored as <strong>BLOB data</strong> in the database</li>
              <li>Audio files are <strong>deleted after processing</strong> - only embeddings remain</li>
              <li>Speaker matching uses <strong>multi-metric similarity</strong> (threshold: 60%)</li>
              <li>Same voice = high similarity score (&gt;72% = strong match)</li>
            </ul>
          </div>
        </div>
      </div>
    );
  }

  // DETAILS VIEW
  if (view === 'details' && selectedLog) {
    return (
      <div style={{
        minHeight: '100vh',
        background: 'linear-gradient(to bottom right, #eff6ff, #e0e7ff)',
        padding: '24px'
      }}>
        <div style={{ maxWidth: '896px', margin: '0 auto' }}>
          <button
            onClick={() => setView('logs')}
            style={{
              marginBottom: '24px',
              padding: '8px 16px',
              background: 'white',
              borderRadius: '8px',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
              border: 'none',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              color: '#374151'
            }}
          >
            <X size={20} />
            Back to Logs
          </button>

          <div style={{
            background: 'white',
            borderRadius: '12px',
            boxShadow: '0 10px 15px rgba(0,0,0,0.1)',
            padding: '24px',
            marginBottom: '24px'
          }}>
            <h2 style={{ fontSize: '24px', fontWeight: 'bold', color: '#312e81', marginBottom: '16px' }}>
              Recording Details
            </h2>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', color: '#374151' }}>
              <div>
                <span style={{ fontWeight: '600' }}>Filename:</span> {selectedLog.filename}
              </div>
              <div>
                <span style={{ fontWeight: '600' }}>Uploaded On:</span> {new Date(selectedLog.upload_date).toLocaleDateString('en-IN', { timeZone: 'Asia/Kolkata' })}
              </div>
              <div>
                <span style={{ fontWeight: '600' }}>Upload Time:</span> {new Date(selectedLog.upload_date).toLocaleTimeString('en-IN', { timeZone: 'Asia/Kolkata', hour12: true })}
              </div>
              <div>
                <span style={{ fontWeight: '600' }}>Total Speakers:</span> {selectedLog.speakers.length}
              </div>
            </div>
          </div>

          <h3 style={{ fontSize: '20px', fontWeight: 'bold', color: '#312e81', marginBottom: '16px' }}>
            Speakers
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {selectedLog.speakers.map((speaker, index) => (
              <div key={index} style={{
                background: 'white',
                borderRadius: '12px',
                boxShadow: '0 10px 15px rgba(0,0,0,0.1)',
                padding: '24px'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: '12px' }}>
                  {editingName === index ? (
                    <div style={{ display: 'flex', gap: '8px', flex: 1 }}>
                      <input
                        type="text"
                        value={editValue}
                        onChange={(e) => setEditValue(e.target.value)}
                        style={{
                          flex: 1,
                          padding: '8px 12px',
                          border: '1px solid #d1d5db',
                          borderRadius: '8px',
                          outline: 'none'
                        }}
                        autoFocus
                      />
                      <button
                        onClick={() => updateSpeakerName(index, editValue)}
                        style={{
                          padding: '8px 16px',
                          background: '#10b981',
                          color: 'white',
                          borderRadius: '8px',
                          border: 'none',
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '8px'
                        }}
                      >
                        <Save size={18} />
                      </button>
                      <button
                        onClick={() => {
                          setEditingName(null);
                          setEditValue('');
                        }}
                        style={{
                          padding: '8px 16px',
                          background: '#d1d5db',
                          color: '#374151',
                          borderRadius: '8px',
                          border: 'none',
                          cursor: 'pointer'
                        }}
                      >
                        <X size={18} />
                      </button>
                    </div>
                  ) : (
                    <>
                      <h4 style={{ fontSize: '18px', fontWeight: 'bold', color: '#312e81' }}>
                        {speaker.name}
                      </h4>
                      <button
                        onClick={() => {
                          setEditingName(index);
                          setEditValue(speaker.name);
                        }}
                        style={{
                          padding: '4px 12px',
                          background: '#dbeafe',
                          color: '#1e40af',
                          borderRadius: '8px',
                          border: 'none',
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '8px'
                        }}
                      >
                        <Edit2 size={16} />
                        Edit
                      </button>
                    </>
                  )}
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '8px', color: '#374151' }}>
                  <div>
                    <span style={{ fontWeight: '600' }}>Confidence:</span> {speaker.confidence}%
                  </div>
                </div>
                {speaker.voice_profile_id && (
                  <div style={{ marginTop: '12px', padding: '8px', background: '#f0f9ff', borderRadius: '6px', fontSize: '14px', color: '#0369a1' }}>
                    🔗 Linked to Voice Profile #{speaker.voice_profile_id}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  // MAIN LOGS VIEW
  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(to bottom right, #eff6ff, #e0e7ff)',
      padding: '24px'
    }}>
      <div style={{ maxWidth: '1152px', margin: '0 auto' }}>
        <div style={{
          background: 'white',
          borderRadius: '12px',
          boxShadow: '0 10px 15px rgba(0,0,0,0.1)',
          padding: '24px',
          marginBottom: '24px'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
            <h1 style={{ fontSize: '30px', fontWeight: 'bold', color: '#312e81' }}>
              Vocald - Call Recordings
            </h1>
            <button
              onClick={() => { setView('database'); fetchVoiceProfiles(); }}
              style={{
                padding: '12px 24px',
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                color: 'white',
                borderRadius: '8px',
                border: 'none',
                cursor: 'pointer',
                fontWeight: '600',
                display: 'flex',
                alignItems: 'center',
                gap: '8px'
              }}
            >
              <Database size={20} />
              View Database
            </button>
          </div>

          {/* Connection Status */}
          <div style={{ 
            padding: '10px 14px', 
            background: connectionStatus === 'connected' ? '#d1fae5' : connectionStatus === 'error' ? '#fee2e2' : '#fef3c7',
            borderRadius: '8px', 
            fontSize: '13px', 
            color: connectionStatus === 'connected' ? '#065f46' : connectionStatus === 'error' ? '#991b1b' : '#92400e',
            marginBottom: '16px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            fontWeight: '500'
          }}>
            {connectionStatus === 'connected' && '✅ Connected'}
            {connectionStatus === 'error' && '❌ Connection Error'}
            {connectionStatus === 'checking' && '⏳ Checking connection...'}
            <span style={{ fontSize: '11px', opacity: 0.7 }}>• {API_BASE}</span>
          </div>
          
          <div style={{ display: 'flex', gap: '16px', marginBottom: '24px' }}>
            <div style={{ flex: 1, position: 'relative' }}>
              <Search style={{
                position: 'absolute',
                left: '12px',
                top: '50%',
                transform: 'translateY(-50%)',
                color: '#9ca3af'
              }} size={20} />
              <input
                type="text"
                placeholder="Search recordings..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                style={{
                  width: '100%',
                  paddingLeft: '40px',
                  paddingRight: '16px',
                  paddingTop: '12px',
                  paddingBottom: '12px',
                  border: '1px solid #d1d5db',
                  borderRadius: '8px',
                  outline: 'none'
                }}
              />
            </div>
            
            <label style={{
              padding: '12px 24px',
              background: '#4f46e5',
              color: 'white',
              borderRadius: '8px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              fontWeight: '600',
              border: 'none'
            }}>
              {uploading ? (
                <>
                  <Loader style={{ animation: 'spin 1s linear infinite' }} size={20} />
                  Uploading...
                </>
              ) : (
                <>
                  <Upload size={20} />
                  Upload Audio
                </>
              )}
              <input
                type="file"
                multiple
                accept="audio/mp3,audio/wav,audio/*"
                onChange={handleFileUpload}
                style={{ display: 'none' }}
                disabled={uploading}
              />
            </label>
          </div>
        </div>

        {loading ? (
          <div style={{
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            padding: '80px 0'
          }}>
            <Loader style={{ animation: 'spin 1s linear infinite', color: '#4f46e5' }} size={48} />
          </div>
        ) : filteredLogs.length === 0 ? (
          <div style={{
            background: 'white',
            borderRadius: '12px',
            boxShadow: '0 10px 15px rgba(0,0,0,0.1)',
            padding: '48px',
            textAlign: 'center',
            color: '#6b7280'
          }}>
            <Upload size={48} style={{ margin: '0 auto 16px', color: '#d1d5db' }} />
            <p style={{ fontSize: '18px' }}>No recordings found. Upload audio files to get started.</p>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {filteredLogs.map((log) => (
              <div
                key={log.id}
                onClick={() => viewDetails(log.id)}
                style={{
                  background: 'white',
                  borderRadius: '12px',
                  boxShadow: '0 10px 15px rgba(0,0,0,0.1)',
                  padding: '24px',
                  cursor: 'pointer',
                  transition: 'box-shadow 0.2s'
                }}
                onMouseEnter={(e) => e.currentTarget.style.boxShadow = '0 20px 25px rgba(0,0,0,0.15)'}
                onMouseLeave={(e) => e.currentTarget.style.boxShadow = '0 10px 15px rgba(0,0,0,0.1)'}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div style={{ flex: 1 }}>
                    <h3 style={{ fontSize: '20px', fontWeight: 'bold', color: '#312e81', marginBottom: '8px' }}>
                      {log.filename}
                    </h3>
                    <div style={{ display: 'flex', gap: '24px', color: '#4b5563' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <Clock size={16} />
                        <span>Uploaded: {new Date(log.upload_date).toLocaleDateString('en-IN', { timeZone: 'Asia/Kolkata' })} {new Date(log.upload_date).toLocaleTimeString('en-IN', { timeZone: 'Asia/Kolkata', hour12: true })}</span>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <Users size={16} />
                        <span>{log.total_speakers} speakers</span>
                      </div>
                    </div>
                  </div>
                  <ChevronRight style={{ color: '#9ca3af' }} size={24} />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}