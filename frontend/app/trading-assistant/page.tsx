'use client';

import { useState, useEffect, useRef } from 'react';
import PageHeader from '@/components/PageHeader';
import Card, { CardBody } from '@/components/Card';
import Button from '@/components/Button';
import { Send, Bot, User } from 'lucide-react';

interface Message {
  role: 'user' | 'bot';
  content: string;
}

const quickActions = [
  'What are the market conditions?',
  'Which pairs should I watch?',
  'Show me the latest signals',
  'Analyze EUR/USD trends',
];

export default function TradingAssistantPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Load chat history from localStorage
    loadChatHistory();
  }, []);

  const loadChatHistory = () => {
    try {
      const savedChat = localStorage.getItem('trading_assistant_chat');
      if (savedChat) {
        const parsed = JSON.parse(savedChat);
        if (Array.isArray(parsed)) {
          setMessages(parsed);
        }
      }
    } catch (error) {
      console.error('Failed to load chat history:', error);
    }
  };

  const saveChatHistory = (msgs: Message[]) => {
    try {
      localStorage.setItem('trading_assistant_chat', JSON.stringify(msgs));
    } catch (error) {
      console.error('Failed to save chat history:', error);
    }
  };

  const sendMessage = async (message?: string) => {
    const messageToSend = message || input.trim();
    if (!messageToSend) return;

    const userMessage: Message = { role: 'user', content: messageToSend };
    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);
    saveChatHistory(updatedMessages);
    setInput('');
    setLoading(true);

    try {
      const response = await fetch('/api/general-chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: messageToSend })
      });

      const data = await response.json();
      
      if (response.ok) {
        const botMessage: Message = { role: 'bot', content: data.answer };
        const finalMessages = [...updatedMessages, botMessage];
        setMessages(finalMessages);
        saveChatHistory(finalMessages);
      } else {
        const errorMessage: Message = {
          role: 'bot',
          content: data.error || 'Sorry, I encountered an error.'
        };
        const finalMessages = [...updatedMessages, errorMessage];
        setMessages(finalMessages);
        saveChatHistory(finalMessages);
      }
    } catch (error) {
      console.error('Chat error:', error);
      const errorMessage: Message = {
        role: 'bot',
        content: 'Sorry, there was a connection error.'
      };
      const finalMessages = [...updatedMessages, errorMessage];
      setMessages(finalMessages);
      saveChatHistory(finalMessages);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
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

  return (
    <div className="min-h-screen bg-gray-900 flex flex-col">
      <PageHeader
        title="Trading Assistant"
        subtitle="General forex discussions, market analysis, and trading strategies"
      />

      <div className="flex-1 p-6 lg:p-8">
        <Card className="h-[calc(100vh-16rem)] flex flex-col">
          {/* Quick Actions */}
          <div className="px-6 py-4 border-b border-gray-700">
            <div className="flex flex-wrap gap-2">
              {quickActions.map((action, index) => (
                <button
                  key={index}
                  onClick={() => sendMessage(action)}
                  className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 text-gray-300 hover:text-white rounded-lg text-sm transition-colors border border-gray-600"
                >
                  {action}
                </button>
              ))}
            </div>
          </div>

          {/* Messages */}
          <CardBody className="flex-1 overflow-y-auto space-y-4">
            {messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-center">
                <div className="w-20 h-20 bg-blue-600 rounded-full flex items-center justify-center mb-4">
                  <Bot className="w-10 h-10 text-white" />
                </div>
                <h3 className="text-xl font-semibold text-white mb-2">
                  Welcome to Trading Assistant
                </h3>
                <p className="text-gray-400 max-w-md">
                  Ask me anything about forex markets, trading strategies, technical analysis, or general market insights.
                </p>
              </div>
            ) : (
              <>
                {messages.map((message, index) => (
                  <div
                    key={index}
                    className={`flex gap-3 ${
                      message.role === 'user' ? 'flex-row-reverse' : 'flex-row'
                    }`}
                  >
                    <div className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center ${
                      message.role === 'user' ? 'bg-blue-600' : 'bg-gray-700'
                    }`}>
                      {message.role === 'user' ? (
                        <User className="w-5 h-5 text-white" />
                      ) : (
                        <Bot className="w-5 h-5 text-white" />
                      )}
                    </div>
                    <div className={`flex-1 max-w-[85%] ${
                      message.role === 'user' ? 'text-right' : 'text-left'
                    }`}>
                      <div className={`inline-block px-4 py-3 rounded-lg ${
                        message.role === 'user'
                          ? 'bg-blue-600 text-white'
                          : 'bg-gray-700'
                      }`}>
                        {message.role === 'user' ? (
                          <p className="text-sm text-white leading-relaxed">{message.content}</p>
                        ) : (
                          formatAIResponse(message.content)
                        )}
                      </div>
                    </div>
                  </div>
                ))}
                {loading && (
                  <div className="flex gap-3">
                    <div className="flex-shrink-0 w-10 h-10 rounded-full bg-gray-700 flex items-center justify-center">
                      <Bot className="w-5 h-5 text-white" />
                    </div>
                    <div className="flex-1">
                      <div className="inline-block px-4 py-3 rounded-lg bg-gray-700">
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
                  </div>
                )}
                <div ref={messagesEndRef} />
              </>
            )}
          </CardBody>

          {/* Input */}
          <div className="px-6 py-4 border-t border-gray-700">
            <div className="flex gap-3">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask about forex markets, strategies, or analysis..."
                className="flex-1 px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
                disabled={loading}
              />
              <Button
                onClick={() => sendMessage()}
                disabled={loading || !input.trim()}
                variant="primary"
              >
                <Send className="w-5 h-5" />
                Send
              </Button>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
