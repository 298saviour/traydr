'use client';

import { useState, useEffect } from 'react';
import PageHeader from '@/components/PageHeader';
import Card, { CardBody } from '@/components/Card';
import Button from '@/components/Button';
import { MessageCircle, ChevronDown, ChevronUp, X, Send } from 'lucide-react';

interface Signal {
  id: number;
  pair: string;
  recommendation: string;
  status: string;
  entry_price: string;
  stop_loss: string;
  take_profit: string;
  take_profit_1?: string;
  confidence: number | string;
  risk_amount?: string;
  lot_size?: string;
  created_at: string;
  updated_at: string;
  analysis_markdown?: string;
  signal_id?: string;
}

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

const filters = [
  { id: 'all', label: 'All Signals' },
  { id: 'active', label: 'Active' },
  { id: 'hit_tp', label: 'Hit TP' },
  { id: 'hit_sl', label: 'Hit SL' },
  { id: 'expired', label: 'Expired' },
];

export default function SignalHistoryPage() {
  const [signals, setSignals] = useState<Signal[]>([]);
  const [activeFilter, setActiveFilter] = useState('all');
  const [expandedSignal, setExpandedSignal] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [chatSignal, setChatSignal] = useState<Signal | null>(null);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);

  useEffect(() => {
    loadSignalHistory(activeFilter);
  }, [activeFilter]);

  const loadSignalHistory = async (filter: string) => {
    setLoading(true);
    try {
      const url = filter !== 'all'
        ? `/api/signals/history?status=${filter}`
        : '/api/signals/history';
      
      const response = await fetch(url, { cache: 'no-store' });
      const data = await response.json();
      
      if (data.success && Array.isArray(data.signals)) {
        setSignals(data.signals);
      } else {
        setSignals([]);
      }
    } catch (error) {
      console.error('Error loading signal history:', error);
      setSignals([]);
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateString: string) => {
    try {
      return new Date(dateString).toLocaleString();
    } catch {
      return dateString;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status?.toLowerCase()) {
      case 'active':
        return 'bg-blue-500/20 text-blue-400';
      case 'hit_tp':
        return 'bg-green-500/20 text-green-400';
      case 'hit_sl':
        return 'bg-red-500/20 text-red-400';
      case 'expired':
        return 'bg-gray-500/20 text-gray-400';
      default:
        return 'bg-gray-500/20 text-gray-400';
    }
  };

  const getRecommendationColor = (recommendation: string) => {
    if (recommendation === 'BUY') return 'bg-green-500/20 text-green-400';
    if (recommendation === 'SELL') return 'bg-red-500/20 text-red-400';
    return 'bg-gray-500/20 text-gray-400';
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

  return (
    <div className="min-h-screen bg-gray-900">
      <PageHeader
        title="Signal History"
        subtitle="Complete record of all generated trading signals"
      />

      <div className="p-6 lg:p-8 space-y-6">
        {/* Filters */}
        <div className="flex flex-wrap gap-3">
          {filters.map((filter) => (
            <button
              key={filter.id}
              onClick={() => setActiveFilter(filter.id)}
              className={`px-4 py-2 rounded-lg font-medium transition-all ${
                activeFilter === filter.id
                  ? 'bg-blue-600 text-white shadow-lg'
                  : 'bg-gray-800 text-gray-300 hover:bg-gray-700 border border-gray-700'
              }`}
            >
              {filter.label}
            </button>
          ))}
        </div>

        {/* Signals List */}
        {loading ? (
          <Card>
            <CardBody className="text-center py-12">
              <div className="animate-spin w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full mx-auto mb-4" />
              <p className="text-gray-400">Loading signals...</p>
            </CardBody>
          </Card>
        ) : signals.length === 0 ? (
          <Card>
            <CardBody className="text-center py-12">
              <div className="text-6xl mb-4">📋</div>
              <h3 className="text-xl font-semibold text-white mb-2">No Signal History</h3>
              <p className="text-gray-400">
                Generate signals from the Live Signals page to see your trading history here
              </p>
            </CardBody>
          </Card>
        ) : (
          <div className="space-y-4">
            {signals.map((signal) => (
              <Card key={signal.id} className="hover:border-blue-500 transition-all">
                <CardBody>
                  {/* Header */}
                  <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-4">
                    <div>
                      <h3 className="text-2xl font-bold text-white mb-1">
                        {signal.pair || 'Unknown'}
                      </h3>
                      <p className="text-sm text-gray-500 font-mono">
                        {signal.signal_id || `#${signal.id}`}
                      </p>
                    </div>
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-sm text-gray-400">
                        {formatDate(signal.created_at)}
                      </span>
                      <span className={`px-3 py-1 rounded-full text-xs font-semibold ${getStatusColor(signal.status)}`}>
                        {signal.status?.toUpperCase() || 'N/A'}
                      </span>
                      <span className={`px-3 py-1 rounded-full text-xs font-semibold ${getRecommendationColor(signal.recommendation)}`}>
                        {signal.recommendation || 'N/A'}
                      </span>
                    </div>
                  </div>

                  {/* Details Grid */}
                  <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4 mb-4">
                    <div className="bg-gray-900/50 rounded-lg p-3 border border-gray-700">
                      <p className="text-xs text-gray-500 uppercase font-semibold mb-1">Entry</p>
                      <p className="text-base font-semibold text-white">{signal.entry_price || 'N/A'}</p>
                    </div>
                    <div className="bg-gray-900/50 rounded-lg p-3 border border-gray-700">
                      <p className="text-xs text-gray-500 uppercase font-semibold mb-1">Stop Loss</p>
                      <p className="text-base font-semibold text-red-400">{signal.stop_loss || 'N/A'}</p>
                    </div>
                    <div className="bg-gray-900/50 rounded-lg p-3 border border-gray-700">
                      <p className="text-xs text-gray-500 uppercase font-semibold mb-1">Take Profit</p>
                      <p className="text-base font-semibold text-green-400">
                        {signal.take_profit_1 || signal.take_profit || 'N/A'}
                      </p>
                    </div>
                    <div className="bg-gray-900/50 rounded-lg p-3 border border-gray-700">
                      <p className="text-xs text-gray-500 uppercase font-semibold mb-1">Confidence</p>
                      <p className="text-base font-semibold text-white">{signal.confidence || 'N/A'}</p>
                    </div>
                    {signal.risk_amount && (
                      <div className="bg-gray-900/50 rounded-lg p-3 border border-gray-700">
                        <p className="text-xs text-gray-500 uppercase font-semibold mb-1">Risk</p>
                        <p className="text-base font-semibold text-white">{signal.risk_amount}</p>
                      </div>
                    )}
                    {signal.lot_size && (
                      <div className="bg-gray-900/50 rounded-lg p-3 border border-gray-700">
                        <p className="text-xs text-gray-500 uppercase font-semibold mb-1">Lot Size</p>
                        <p className="text-base font-semibold text-white">{signal.lot_size}</p>
                      </div>
                    )}
                  </div>

                  {/* Analysis Section */}
                  {signal.analysis_markdown && (
                    <div className="mb-4">
                      <button
                        onClick={() => setExpandedSignal(expandedSignal === signal.id ? null : signal.id)}
                        className="flex items-center gap-2 text-blue-400 hover:text-blue-300 font-medium text-sm"
                      >
                        {expandedSignal === signal.id ? (
                          <>
                            <ChevronUp className="w-4 h-4" />
                            Hide Analysis
                          </>
                        ) : (
                          <>
                            <ChevronDown className="w-4 h-4" />
                            Show Market Analysis
                          </>
                        )}
                      </button>
                      {expandedSignal === signal.id && (
                        <div className="mt-3 p-4 bg-blue-900/20 border border-blue-800/30 rounded-lg">
                          <div
                            className="prose prose-invert prose-sm max-w-none text-gray-300"
                            dangerouslySetInnerHTML={{ __html: signal.analysis_markdown }}
                          />
                        </div>
                      )}
                    </div>
                  )}

                  {/* Footer */}
                  <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 pt-4 border-t border-gray-700">
                    <p className="text-xs text-gray-500 font-mono">
                      Updated: {formatDate(signal.updated_at)}
                    </p>
                    <button
                      onClick={() => openSignalChat(signal)}
                      className="inline-flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors text-sm"
                    >
                      <MessageCircle className="w-4 h-4" />
                      Ask AI
                    </button>
                  </div>
                </CardBody>
              </Card>
            ))}
          </div>
        )}
      </div>

      {/* Signal Chat Modal */}
      {chatSignal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
          <div className="bg-gray-800 rounded-lg shadow-2xl max-w-2xl w-full h-[600px] flex flex-col">
            <div className="bg-gray-800 border-b border-gray-700 p-4 flex items-center justify-between">
              <div>
                <h2 className="text-xl font-bold text-white">AI Chat: {chatSignal.pair}</h2>
                <p className="text-gray-400 text-sm">{chatSignal.recommendation} Signal - {formatDate(chatSignal.created_at)}</p>
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
