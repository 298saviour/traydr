'use client';

import { useState, useEffect, useRef } from 'react';
import PageHeader from '@/components/PageHeader';
import StatCard from '@/components/StatCard';
import Card, { CardBody, CardHeader, CardTitle } from '@/components/Card';
import Button from '@/components/Button';
import { Activity, TrendingUp, TrendingDown, Target, RefreshCw, Sparkles, MessageCircle, Clock, Trash2, Play, Square, X, Send } from 'lucide-react';
import { parseConfidence } from '@/lib/utils';

interface Signal {
  id: number;
  pair: string;
  recommendation: string;
  confidence: number | string;
  entry_price: string;
  stop_loss: string;
  take_profit: string;
  lot_size_100_risk?: string;
  lot_size_200_risk?: string;
  status: string;
  analysis_markdown?: string;
  risk_amount?: string;
}

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

interface LogEntry {
  id: string;
  timestamp: string;
  action: string;
  status: 'in_progress' | 'completed' | 'error';
  details: string;
}

export default function LiveSignalsPage() {
  const [signals, setSignals] = useState<Signal[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedPair, setSelectedPair] = useState('EUR/USD');
  const [analysisResult, setAnalysisResult] = useState<any>(null);
  const [progressLogs, setProgressLogs] = useState<LogEntry[]>([]);
  const [isAutomationRunning, setIsAutomationRunning] = useState(false);
  const [automationInterval, setAutomationInterval] = useState<NodeJS.Timeout | null>(null);
  const [selectedSignal, setSelectedSignal] = useState<Signal | null>(null);
  const [chatSignal, setChatSignal] = useState<Signal | null>(null);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const [lastSignalId, setLastSignalId] = useState<number | null>(null);
  const statusPollInterval = useRef<NodeJS.Timeout | null>(null);

  const pairs = [
    { code: 'EUR/USD', name: 'Euro vs US Dollar' },
    { code: 'GBP/USD', name: 'British Pound vs US Dollar' },
    { code: 'USD/JPY', name: 'US Dollar vs Japanese Yen' },
    { code: 'XAU/USD', name: 'Gold vs US Dollar' },
    { code: 'EUR/GBP', name: 'Euro vs British Pound' },
    { code: 'USD/CAD', name: 'US Dollar vs Canadian Dollar' },
    { code: 'AUD/USD', name: 'Australian Dollar vs US Dollar' },
    { code: 'USD/CHF', name: 'US Dollar vs Swiss Franc' },
    { code: 'NZD/USD', name: 'New Zealand Dollar vs US Dollar' },
    { code: 'EUR/JPY', name: 'Euro vs Japanese Yen' },
    { code: 'GBP/JPY', name: 'British Pound vs Japanese Yen' },
    { code: 'AUD/JPY', name: 'Australian Dollar vs Japanese Yen' },
    { code: 'ETH/USD', name: 'Ethereum' },
    { code: 'BTC/USD', name: 'Bitcoin' },
  ];

  useEffect(() => {
    refreshSignals();
    checkAutomationStatus(); // Check if automation is running on load
    const interval = setInterval(refreshSignals, 10000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    return () => {
      if (automationInterval) {
        clearInterval(automationInterval);
      }
      if (statusPollInterval.current) {
        clearInterval(statusPollInterval.current);
      }
    };
  }, [automationInterval]);

  // Poll backend status when automation is running
  useEffect(() => {
    if (isAutomationRunning) {
      startStatusPolling();
    } else {
      stopStatusPolling();
    }
    return () => stopStatusPolling();
  }, [isAutomationRunning]);

  const parseLogMessage = (msg: string): { action: string; status: LogEntry['status'] } => {
    let action = msg;
    let status: LogEntry['status'] = 'in_progress';
    
    if (msg.includes('✓') || msg.includes('complete') || msg.includes('generated') || msg.includes('Saved')) {
      status = 'completed';
    } else if (msg.includes('✗') || msg.toLowerCase().includes('error') || msg.toLowerCase().includes('failed') || msg.includes('Missing')) {
      status = 'error';
    } else if (msg.includes('Fetching') || msg.includes('Starting') || msg.includes('Calculating') || msg.includes('Pausing')) {
      status = 'in_progress';
    }
    
    // Extract action from message
    if (msg.includes('[Phase A]') || msg.includes('[Phase B]') || msg.includes('[Confirmation Phase]')) {
      const match = msg.match(/\[(?:Phase [AB]|Confirmation Phase)\]\s*(.+?)(?:\s+for|\s*$)/);
      action = match ? match[1].trim() : msg;
    } else if (msg.includes(':')) {
      action = msg.split(':')[0].trim();
    }
    
    return { action, status };
  };

  const formatTimestamp = (isoString: string): string => {
    try {
      return new Date(isoString).toLocaleString('en-US', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false,
        timeZone: 'Africa/Lagos'
      }) + ' WAT';
    } catch {
      return isoString;
    }
  };

  const checkAutomationStatus = async () => {
    try {
      const response = await fetch('/status');
      const data = await response.json();
      
      if (data.success && data.status) {
        const status = data.status;
        
        // Check if automation is running
        if (status.state === 'cycle_running' || status.state === 'resting') {
          setIsAutomationRunning(true);
          if (status.pair) {
            setSelectedPair(status.pair);
          }
        }
        
        // Load existing logs
        if (status.progress_log && Array.isArray(status.progress_log)) {
          const logs = status.progress_log.map((log: any) => {
            const { action, status: logStatus } = parseLogMessage(log.msg);
            return {
              id: log.ts,
              timestamp: formatTimestamp(log.ts),
              action,
              status: logStatus,
              details: log.msg
            };
          });
          setProgressLogs(logs.reverse()); // Show newest first
        }
        
        // Store last signal ID
        if (status.last_signal_id) {
          setLastSignalId(status.last_signal_id);
        }
      }
    } catch (error) {
      console.error('Error checking automation status:', error);
    }
  };

  const startStatusPolling = () => {
    stopStatusPolling(); // Clear any existing interval
    
    const pollStatus = async () => {
      try {
        const response = await fetch('/status');
        const data = await response.json();
        
        if (data.success && data.status) {
          const status = data.status;
          
          // Update logs from backend
          if (status.progress_log && Array.isArray(status.progress_log)) {
            const logs = status.progress_log.map((log: any) => {
              const { action, status: logStatus } = parseLogMessage(log.msg);
              return {
                id: log.ts,
                timestamp: formatTimestamp(log.ts),
                action,
                status: logStatus,
                details: log.msg
              };
            });
            setProgressLogs(logs.reverse());
          }
          
          // Check if automation finished
          if (status.state === 'idle' && status.last_signal_id) {
            // Check if new signal was generated
            if (status.last_signal_id !== lastSignalId) {
              setLastSignalId(status.last_signal_id);
              // Refresh signals to show new one
              setTimeout(() => refreshSignals(), 1000);
            }
            
            // Stop automation if it's idle
            if (isAutomationRunning) {
              setIsAutomationRunning(false);
            }
          }
          
          // Handle errors
          if (status.last_error) {
            console.error('Automation error:', status.last_error);
          }
        }
      } catch (error) {
        console.error('Error polling status:', error);
      }
    };
    
    // Poll immediately, then every 2 seconds
    pollStatus();
    statusPollInterval.current = setInterval(pollStatus, 2000);
  };

  const stopStatusPolling = () => {
    if (statusPollInterval.current) {
      clearInterval(statusPollInterval.current);
      statusPollInterval.current = null;
    }
  };

  const clearLogs = () => {
    setProgressLogs([]);
  };

  const refreshSignals = async () => {
    try {
      const response = await fetch('/api/signals', { cache: 'no-store' });
      const data = await response.json();
      if (data.success && Array.isArray(data.signals)) {
        setSignals(data.signals);
      }
    } catch (error) {
      console.error('Error refreshing signals:', error);
    }
  };

  const startAutomation = async () => {
    if (!selectedPair) {
      alert('Please select a currency pair');
      return;
    }

    if (isAutomationRunning) {
      return;
    }

    // Clear previous logs
    setProgressLogs([]);
    setIsAutomationRunning(true);

    try {
      const response = await fetch('/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          pair: selectedPair,
          outputsize: 'compact'
        })
      });

      const data = await response.json();
      
      if (data.success) {
        // Backend will handle everything
        // Status polling will update logs automatically
      } else {
        setIsAutomationRunning(false);
        alert('Error starting automation: ' + (data.error || 'Unknown error'));
      }
    } catch (error) {
      console.error('Error starting automation:', error);
      setIsAutomationRunning(false);
      alert('Unable to connect to backend server. Please check your connection.');
    }
  };

  const stopAutomation = async () => {
    if (!isAutomationRunning) {
      return;
    }

    try {
      const response = await fetch('/stop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });

      const data = await response.json();
      if (data.success) {
        setIsAutomationRunning(false);
        if (automationInterval) {
          clearInterval(automationInterval);
          setAutomationInterval(null);
        }
        // Logs will be updated by status polling
      } else {
        alert('Error stopping automation: ' + (data.error || 'Unknown error'));
      }
    } catch (error) {
      console.error('Error stopping automation:', error);
      alert('Connection error or server unavailable');
    }
  };

  const openSignalDetails = (signal: Signal) => {
    setSelectedSignal(signal);
  };

  const closeSignalDetails = () => {
    setSelectedSignal(null);
  };

  const openSignalChat = (signal: Signal) => {
    setChatSignal(signal);
    // Load chat history from localStorage
    const chatKey = `chat_signal_${signal.id}`;
    const savedChat = localStorage.getItem(chatKey);
    if (savedChat) {
      setChatMessages(JSON.parse(savedChat));
    } else {
      setChatMessages([]);
    }
  };

  const closeSignalChat = () => {
    setChatSignal(null);
    setChatMessages([]);
    setChatInput('');
  };

  const formatAIResponse = (content: string): JSX.Element => {
    // Check if response already has AI Response: prefix
    const hasPrefix = content.trim().startsWith('AI Response:');
    const mainContent = hasPrefix ? content.replace(/^AI Response:\s*/i, '').trim() : content;
    
    // Split into paragraphs
    const paragraphs = mainContent.split('\n\n').filter(p => p.trim());
    
    return (
      <div className="space-y-3">
        <div className="font-semibold text-blue-400 text-sm mb-2">AI Response:</div>
        {paragraphs.map((paragraph, idx) => {
          const trimmed = paragraph.trim();
          
          // Check if it's a bullet list
          if (trimmed.includes('\n•') || trimmed.includes('\n-') || trimmed.startsWith('•') || trimmed.startsWith('-')) {
            const items = trimmed.split('\n').filter(line => line.trim());
            return (
              <ul key={idx} className="space-y-2 ml-4">
                {items.map((item, i) => {
                  const cleanItem = item.replace(/^[•\-]\s*/, '').trim();
                  if (!cleanItem) return null;
                  return (
                    <li key={i} className="text-sm text-gray-100 leading-relaxed">
                      <span className="text-blue-400 mr-2">•</span>
                      {cleanItem}
                    </li>
                  );
                })}
              </ul>
            );
          }
          
          // Check if it's a bold header (starts with **)
          if (trimmed.startsWith('**') && trimmed.includes('**:')) {
            const parts = trimmed.split('**');
            return (
              <div key={idx} className="text-sm">
                {parts.map((part, i) => {
                  if (i % 2 === 1) {
                    return <strong key={i} className="text-white font-semibold">{part}</strong>;
                  }
                  return <span key={i} className="text-gray-100">{part}</span>;
                })}
              </div>
            );
          }
          
          // Regular paragraph
          return (
            <p key={idx} className="text-sm text-gray-100 leading-relaxed">
              {trimmed}
            </p>
          );
        })}
      </div>
    );
  };

  const sendChatMessage = async () => {
    if (!chatInput.trim() || !chatSignal) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: chatInput,
      timestamp: new Date().toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: true
      })
    };

    const updatedMessages = [...chatMessages, userMessage];
    setChatMessages(updatedMessages);
    const currentInput = chatInput;
    setChatInput('');
    setChatLoading(true);

    // Add minimum delay for typing indicator (2-5 seconds)
    const minDelay = 2000 + Math.random() * 3000; // 2-5 seconds
    const startTime = Date.now();

    try {
      console.log('Sending chat message:', {
        signal_id: chatSignal.id,
        question: currentInput,
        signal_pair: chatSignal.pair
      });

      const response = await fetch('/api/ask-claude', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          signal_id: chatSignal.id,
          question: currentInput
        })
      });

      console.log('Response status:', response.status);
      console.log('Response ok:', response.ok);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      console.log('Response data:', data);
      
      // Ensure minimum delay for typing indicator
      const elapsed = Date.now() - startTime;
      if (elapsed < minDelay) {
        await new Promise(resolve => setTimeout(resolve, minDelay - elapsed));
      }
      
      if (data.answer) {
        console.log('AI answer received, length:', data.answer.length);
        const aiMessage: ChatMessage = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: data.answer,
          timestamp: new Date().toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: true
          })
        };
        
        const finalMessages = [...updatedMessages, aiMessage];
        setChatMessages(finalMessages);
        
        // Save to localStorage
        const chatKey = `chat_signal_${chatSignal.id}`;
        localStorage.setItem(chatKey, JSON.stringify(finalMessages));
        console.log('Chat saved to localStorage');
      } else if (data.error) {
        console.error('Backend returned error:', data.error);
        // Show error message
        const errorMessage: ChatMessage = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: `Error: ${data.error}. Please try again.`,
          timestamp: new Date().toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: true
          })
        };
        const finalMessages = [...updatedMessages, errorMessage];
        setChatMessages(finalMessages);
      } else {
        console.warn('No answer or error in response:', data);
        throw new Error('No answer received from backend');
      }
    } catch (error: any) {
      // Comprehensive error logging
      console.error('=== CHAT ERROR START ===');
      console.error('Error object:', error);
      console.error('Error type:', typeof error);
      console.error('Error constructor:', error?.constructor?.name);
      
      // Try to extract all possible error information
      try {
        console.error('Error details:', JSON.stringify({
          message: error?.message || 'No message',
          name: error?.name || 'No name',
          stack: error?.stack || 'No stack',
          toString: error?.toString() || 'No toString'
        }, null, 2));
      } catch (stringifyError) {
        console.error('Could not stringify error:', stringifyError);
        console.error('Raw error properties:', Object.keys(error || {}));
      }
      
      console.error('=== CHAT ERROR END ===');
      
      // Ensure minimum delay even on error
      const elapsed = Date.now() - startTime;
      if (elapsed < minDelay) {
        await new Promise(resolve => setTimeout(resolve, minDelay - elapsed));
      }
      
      // Show error message to user
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `Sorry, I encountered an error: ${error?.message || 'Unknown error'}. Please make sure the backend server is running and try again.`,
        timestamp: new Date().toLocaleTimeString('en-US', {
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit',
          hour12: true
        })
      };
      const finalMessages = [...updatedMessages, errorMessage];
      setChatMessages(finalMessages);
    } finally {
      setChatLoading(false);
    }
  };

  const activeSignals = signals.filter(s => s.status === 'active' && s.recommendation !== 'NO TRADE');
  const buySignals = activeSignals.filter(s => s.recommendation === 'BUY');
  const sellSignals = activeSignals.filter(s => s.recommendation === 'SELL');
  
  const avgConfidence = activeSignals.length > 0
    ? Math.round(activeSignals.reduce((sum, s) => {
        return sum + parseConfidence(s.confidence);
      }, 0) / activeSignals.length)
    : 0;

  return (
    <div className="min-h-screen bg-gray-900">
      <PageHeader
        title="Live Forex Signals"
        subtitle="Real-time trading signals powered by AI analysis"
        action={
          <Button onClick={refreshSignals} variant="secondary">
            <RefreshCw className="w-4 h-4" />
            Refresh
          </Button>
        }
      />

      <div className="p-6 lg:p-8 space-y-8">
        {/* Stats Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          <StatCard
            title="Active Signals"
            value={activeSignals.length}
            icon={Activity}
            iconColor="text-blue-500"
          />
          <StatCard
            title="Buy Signals"
            value={buySignals.length}
            icon={TrendingUp}
            iconColor="text-green-500"
          />
          <StatCard
            title="Sell Signals"
            value={sellSignals.length}
            icon={TrendingDown}
            iconColor="text-red-500"
          />
          <StatCard
            title="Avg Confidence"
            value={`${avgConfidence}%`}
            icon={Target}
            iconColor="text-yellow-500"
          />
        </div>

        {/* Signal Generator */}
        <Card>
          <CardHeader>
            <div className="text-center">
              <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full mb-4">
                <Sparkles className="w-8 h-8 text-white" />
              </div>
              <CardTitle className="text-2xl">AI Signal Generator</CardTitle>
              <p className="text-gray-400 mt-2">Select a currency pair and let AI analyze the market</p>
            </div>
          </CardHeader>
          <CardBody className="space-y-6">
            <div className="max-w-md mx-auto">
              <label className="block text-sm font-medium text-gray-400 mb-2">
                Currency Pair
              </label>
              <select
                value={selectedPair}
                onChange={(e) => setSelectedPair(e.target.value)}
                disabled={isAutomationRunning}
                className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {pairs.map(pair => (
                  <option key={pair.code} value={pair.code}>
                    {pair.code} - {pair.name}
                  </option>
                ))}
              </select>
            </div>
            <div className="flex flex-col sm:flex-row justify-center gap-4">
              <button
                onClick={startAutomation}
                disabled={isAutomationRunning}
                className="inline-flex items-center justify-center gap-2 px-6 py-3 bg-green-600 hover:bg-green-700 text-white font-medium rounded-lg transition-all duration-200 shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-green-600"
              >
                <Play className="w-5 h-5" />
                {isAutomationRunning ? 'Automation Running' : 'Analyze with AI (Start)'}
              </button>
              <button
                onClick={stopAutomation}
                disabled={!isAutomationRunning}
                className="inline-flex items-center justify-center gap-2 px-6 py-3 bg-red-600 hover:bg-red-700 text-white font-medium rounded-lg transition-all duration-200 shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-red-600"
              >
                <Square className="w-5 h-5" />
                Stop
              </button>
            </div>
            {isAutomationRunning && (
              <div className="text-center">
                <div className="inline-flex items-center gap-2 px-4 py-2 bg-green-500/20 border border-green-500/30 rounded-lg">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                  <span className="text-sm text-green-400 font-medium">
                    Automation Active - Generating signals for {selectedPair}
                  </span>
                </div>
              </div>
            )}
          </CardBody>
        </Card>

        {/* Progress Log */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Clock className="w-5 h-5 text-blue-500" />
                <CardTitle>Progress Log</CardTitle>
              </div>
              {progressLogs.length > 0 && (
                <button
                  onClick={clearLogs}
                  className="flex items-center gap-2 px-3 py-1.5 text-sm text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition-colors"
                >
                  <Trash2 className="w-4 h-4" />
                  Clear
                </button>
              )}
            </div>
          </CardHeader>
          <CardBody>
            {progressLogs.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                <Clock className="w-12 h-12 mx-auto mb-3 opacity-50" />
                <p className="text-sm">Automation logs will appear here when you start signal generation.</p>
                <p className="text-xs mt-2 text-gray-600">Click "Analyze with AI (Start)" to begin.</p>
              </div>
            ) : (
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {progressLogs.map((log, index) => (
                  <div
                    key={`${log.id}-${index}`}
                    className="flex items-start gap-3 p-3 bg-gray-900/50 rounded-lg border border-gray-700 hover:border-gray-600 transition-colors"
                  >
                    <div className="flex-shrink-0 mt-1">
                      {log.status === 'in_progress' && (
                        <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse" />
                      )}
                      {log.status === 'completed' && (
                        <div className="w-2 h-2 bg-green-400 rounded-full" />
                      )}
                      {log.status === 'error' && (
                        <div className="w-2 h-2 bg-red-400 rounded-full" />
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between gap-2 mb-1">
                        <span className="text-sm font-semibold text-white truncate">
                          {log.action}
                        </span>
                        <span
                          className={`text-xs font-medium px-2 py-0.5 rounded-full flex-shrink-0 ${
                            log.status === 'in_progress'
                              ? 'bg-yellow-500/20 text-yellow-400'
                              : log.status === 'completed'
                              ? 'bg-green-500/20 text-green-400'
                              : 'bg-red-500/20 text-red-400'
                          }`}
                        >
                          {log.status === 'in_progress' && 'In Progress'}
                          {log.status === 'completed' && 'Completed'}
                          {log.status === 'error' && 'Error'}
                        </span>
                      </div>
                      <p className="text-xs text-gray-400 mb-1">{log.details}</p>
                      <p className="text-xs text-gray-600 font-mono">{log.timestamp}</p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardBody>
        </Card>

        {/* AI Analysis Result */}
        {analysisResult && (
          <Card>
            <CardHeader className="bg-gradient-to-r from-blue-900/50 to-purple-900/50">
              <div className="flex items-center justify-between">
                <CardTitle>AI Analysis: {analysisResult.pair}</CardTitle>
                <span className="px-3 py-1 bg-blue-600 rounded-full text-sm font-medium">
                  {analysisResult.timeframe}
                </span>
              </div>
            </CardHeader>
            <CardBody>
              <div
                className="prose prose-invert max-w-none"
                dangerouslySetInnerHTML={{ __html: analysisResult.analysis }}
              />
            </CardBody>
          </Card>
        )}

        {/* Signals Grid */}
        <div>
          <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-2">
            <Activity className="w-6 h-6 text-blue-500" />
            Active Signals
          </h2>
          {signals.length === 0 ? (
            <Card>
              <CardBody className="text-center py-12">
                <div className="text-6xl mb-4">📊</div>
                <h3 className="text-xl font-semibold text-white mb-2">No Active Signals</h3>
                <p className="text-gray-400">Use the AI Signal Generator above to analyze markets and create signals</p>
              </CardBody>
            </Card>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {signals.map((signal) => (
                <SignalCard 
                  key={signal.id} 
                  signal={signal}
                  onViewDetails={openSignalDetails}
                  onAskAI={openSignalChat}
                />
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Signal Details Modal */}
      {selectedSignal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
          <div className="bg-gray-800 rounded-lg shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="sticky top-0 bg-gray-800 border-b border-gray-700 p-6 flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-bold text-white">{selectedSignal.pair}</h2>
                <p className="text-gray-400 text-sm mt-1">Signal Details</p>
              </div>
              <button
                onClick={closeSignalDetails}
                className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
              >
                <X className="w-6 h-6 text-gray-400" />
              </button>
            </div>
            <div className="p-6 space-y-6">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-gray-400 mb-1">Currency Pair</p>
                  <p className="text-lg font-semibold text-white">{selectedSignal.pair}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-400 mb-1">Type</p>
                  <span className={`inline-block px-3 py-1 rounded-full text-sm font-semibold ${
                    selectedSignal.recommendation === 'BUY' 
                      ? 'bg-green-500/20 text-green-400' 
                      : 'bg-red-500/20 text-red-400'
                  }`}>
                    {selectedSignal.recommendation}
                  </span>
                </div>
                <div>
                  <p className="text-sm text-gray-400 mb-1">Confidence</p>
                  <p className="text-lg font-semibold text-white">{parseConfidence(selectedSignal.confidence)}%</p>
                </div>
                <div>
                  <p className="text-sm text-gray-400 mb-1">Status</p>
                  <p className="text-lg font-semibold text-blue-400 capitalize">{selectedSignal.status}</p>
                </div>
              </div>
              <div className="border-t border-gray-700 pt-4">
                <h3 className="text-lg font-semibold text-white mb-3">Price Levels</h3>
                <div className="grid grid-cols-1 gap-3">
                  <div className="flex justify-between items-center p-3 bg-gray-900/50 rounded-lg">
                    <span className="text-gray-400">Entry Price</span>
                    <span className="text-white font-semibold">{selectedSignal.entry_price || 'N/A'}</span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-gray-900/50 rounded-lg">
                    <span className="text-gray-400">Stop Loss</span>
                    <span className="text-red-400 font-semibold">{selectedSignal.stop_loss || 'N/A'}</span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-gray-900/50 rounded-lg">
                    <span className="text-gray-400">Take Profit</span>
                    <span className="text-green-400 font-semibold">{selectedSignal.take_profit || 'N/A'}</span>
                  </div>
                </div>
              </div>
              {(selectedSignal.lot_size_100_risk || selectedSignal.risk_amount) && (
                <div className="border-t border-gray-700 pt-4">
                  <h3 className="text-lg font-semibold text-white mb-3">Risk Management</h3>
                  <div className="grid grid-cols-1 gap-3">
                    {selectedSignal.risk_amount && (
                      <div className="flex justify-between items-center p-3 bg-gray-900/50 rounded-lg">
                        <span className="text-gray-400">Risk Amount</span>
                        <span className="text-white font-semibold">${selectedSignal.risk_amount}</span>
                      </div>
                    )}
                    {selectedSignal.lot_size_100_risk && (
                      <div className="flex justify-between items-center p-3 bg-gray-900/50 rounded-lg">
                        <span className="text-gray-400">Lot Size ($100 Risk)</span>
                        <span className="text-white font-semibold">{selectedSignal.lot_size_100_risk}</span>
                      </div>
                    )}
                    {selectedSignal.lot_size_200_risk && (
                      <div className="flex justify-between items-center p-3 bg-gray-900/50 rounded-lg">
                        <span className="text-gray-400">Lot Size ($200 Risk)</span>
                        <span className="text-white font-semibold">{selectedSignal.lot_size_200_risk}</span>
                      </div>
                    )}
                  </div>
                </div>
              )}
              {selectedSignal.analysis_markdown && (
                <div className="border-t border-gray-700 pt-4">
                  <h3 className="text-lg font-semibold text-white mb-3">Market Analysis</h3>
                  <div className="p-4 bg-gray-900/50 rounded-lg">
                    <p className="text-gray-300 text-sm leading-relaxed whitespace-pre-wrap">
                      {selectedSignal.analysis_markdown}
                    </p>
                  </div>
                </div>
              )}
              <div className="flex gap-3">
                <button
                  onClick={() => {
                    closeSignalDetails();
                    openSignalChat(selectedSignal);
                  }}
                  className="flex-1 px-4 py-3 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors"
                >
                  Ask AI About This Signal
                </button>
                <button
                  onClick={closeSignalDetails}
                  className="px-4 py-3 bg-red-600 hover:bg-red-700 text-white font-medium rounded-lg transition-colors"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Signal Chat Modal */}
      {chatSignal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
          <div className="bg-gray-800 rounded-lg shadow-2xl max-w-2xl w-full h-[600px] flex flex-col">
            <div className="bg-gray-800 border-b border-gray-700 p-4 flex items-center justify-between">
              <div>
                <h2 className="text-xl font-bold text-white">AI Chat: {chatSignal.pair}</h2>
                <p className="text-gray-400 text-sm">{chatSignal.recommendation} Signal</p>
              </div>
              <button
                onClick={closeSignalChat}
                className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
              >
                <X className="w-5 h-5 text-gray-400" />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-4 space-y-3">
              {chatMessages.length === 0 ? (
                <div className="text-center py-12 text-gray-500">
                  <MessageCircle className="w-16 h-16 mx-auto mb-3 opacity-50" />
                  <p>Start a conversation about this signal</p>
                </div>
              ) : (
                chatMessages.map((msg) => (
                  <div
                    key={msg.id}
                    className={`flex ${
                      msg.role === 'user' ? 'justify-end' : 'justify-start'
                    }`}
                  >
                    <div
                      className={`max-w-[85%] rounded-lg p-4 ${
                        msg.role === 'user'
                          ? 'bg-blue-600 text-white'
                          : 'bg-gray-700'
                      }`}
                    >
                      {msg.role === 'user' ? (
                        <p className="text-sm text-white leading-relaxed">{msg.content}</p>
                      ) : (
                        formatAIResponse(msg.content)
                      )}
                      <p className="text-xs mt-3 text-gray-400 opacity-80">{msg.timestamp}</p>
                    </div>
                  </div>
                ))
              )}
              {chatLoading && (
                <div className="flex justify-start">
                  <div className="bg-gray-700 rounded-lg p-4">
                    <div className="flex items-center gap-2">
                      <span className="text-sm text-gray-400">AI is typing</span>
                      <div className="flex gap-1">
                        <div className="w-2 h-2 bg-gray-400 rounded-full typing-dot" />
                        <div className="w-2 h-2 bg-gray-400 rounded-full typing-dot" />
                        <div className="w-2 h-2 bg-gray-400 rounded-full typing-dot" />
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
            <div className="border-t border-gray-700 p-4">
              <div className="flex gap-2">
                <input
                  type="text"
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && sendChatMessage()}
                  placeholder="Ask about this signal..."
                  className="flex-1 px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  disabled={chatLoading}
                />
                <button
                  onClick={sendChatMessage}
                  disabled={chatLoading || !chatInput.trim()}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Send className="w-5 h-5" />
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

interface SignalCardProps {
  signal: Signal;
  onViewDetails: (signal: Signal) => void;
  onAskAI: (signal: Signal) => void;
}

function SignalCard({ signal, onViewDetails, onAskAI }: SignalCardProps) {
  if (signal.recommendation === 'NO TRADE') {
    return (
      <Card>
        <CardBody>
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-lg font-bold text-white">{signal.pair}</h3>
            <span className="px-3 py-1 bg-gray-700 text-gray-300 rounded-full text-xs font-semibold">
              NO TRADE
            </span>
          </div>
          <p className="text-sm text-gray-400">
            {signal.analysis_markdown || 'Market conditions are not favorable for trading.'}
          </p>
        </CardBody>
      </Card>
    );
  }

  const isBuy = signal.recommendation === 'BUY';
  const confidence = parseConfidence(signal.confidence);

  return (
    <Card className="hover:border-blue-500 transition-all cursor-pointer" onClick={() => onViewDetails(signal)}>
      <CardBody className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-bold text-white">{signal.pair}</h3>
          <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
            isBuy ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
          }`}>
            {signal.recommendation}
          </span>
        </div>

        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-gray-400">Confidence</span>
            <span className="text-white font-semibold">{confidence}%</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-gray-400">Entry</span>
            <span className="text-white font-semibold">{signal.entry_price || 'N/A'}</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-gray-400">Stop Loss</span>
            <span className="text-red-400 font-semibold">{signal.stop_loss || 'N/A'}</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-gray-400">Take Profit</span>
            <span className="text-green-400 font-semibold">{signal.take_profit || 'N/A'}</span>
          </div>
          {signal.lot_size_100_risk && (
            <>
              <div className="border-t border-gray-700 pt-2 mt-2" />
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">Lot Size ($100)</span>
                <span className="text-white font-semibold">{signal.lot_size_100_risk}</span>
              </div>
              {signal.lot_size_200_risk && (
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Lot Size ($200)</span>
                  <span className="text-white font-semibold">{signal.lot_size_200_risk}</span>
                </div>
              )}
            </>
          )}
        </div>

        <Button 
          fullWidth 
          variant="primary" 
          size="sm"
          onClick={(e) => {
            e.stopPropagation();
            onAskAI(signal);
          }}
        >
          <MessageCircle className="w-4 h-4" />
          Ask AI
        </Button>
      </CardBody>
    </Card>
  );
}
