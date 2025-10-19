'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Zap, Bot, History, TrendingUp, X, Menu } from 'lucide-react';
import { useState } from 'react';

const navigation = [
  { name: 'Live Signals', href: '/', icon: Zap },
  { name: 'Trading Assistant', href: '/trading-assistant', icon: Bot },
  { name: 'Signal History', href: '/signal-history', icon: History },
  { name: 'Performance', href: '/performance', icon: TrendingUp },
];

export default function Sidebar() {
  const pathname = usePathname();
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      {/* Mobile menu button */}
      <button
        onClick={() => setIsOpen(true)}
        className="lg:hidden fixed top-4 left-4 z-40 p-2 bg-gray-800 rounded-lg text-white hover:bg-gray-700"
      >
        <Menu className="w-6 h-6" />
      </button>

      {/* Backdrop */}
      {isOpen && (
        <div
          className="lg:hidden fixed inset-0 bg-black/50 backdrop-blur-sm z-40"
          onClick={() => setIsOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`
          fixed top-0 left-0 z-50 h-screen w-64 bg-gray-800 border-r border-gray-700
          transition-transform duration-300 ease-in-out
          ${isOpen ? 'translate-x-0' : '-translate-x-full'}
          lg:translate-x-0
        `}
      >
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b border-gray-700">
            <div className="flex items-center gap-3">
              <div className="text-2xl">📈</div>
              <div>
                <h1 className="text-xl font-bold text-white">Traydr</h1>
                <p className="text-xs text-gray-400">Trading Signals</p>
              </div>
            </div>
            <button
              onClick={() => setIsOpen(false)}
              className="lg:hidden text-gray-400 hover:text-white"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Navigation */}
          <nav className="flex-1 px-4 py-6 space-y-1">
            <div className="mb-4">
              <p className="px-3 text-xs font-semibold text-gray-500 uppercase tracking-wider">
                Trading
              </p>
            </div>
            {navigation.map((item) => {
              const isActive = pathname === item.href;
              const Icon = item.icon;
              return (
                <Link
                  key={item.name}
                  href={item.href}
                  onClick={() => setIsOpen(false)}
                  className={`
                    flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium
                    transition-colors duration-200
                    ${
                      isActive
                        ? 'bg-blue-600 text-white'
                        : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                    }
                  `}
                >
                  <Icon className="w-5 h-5" />
                  <span>{item.name}</span>
                </Link>
              );
            })}
          </nav>

          {/* System Status */}
          <div className="p-4 border-t border-gray-700">
            <div className="bg-gray-900 rounded-lg p-4 space-y-3">
              <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
                System Status
              </p>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-400">System Active</span>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                </div>
              </div>
              <div className="text-xs text-gray-500">
                Next signal generation: Every hour
              </div>
            </div>
          </div>

          {/* User Profile */}
          <div className="p-4 border-t border-gray-700">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-blue-600 rounded-full flex items-center justify-center font-semibold">
                T
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-white truncate">Trader</p>
                <p className="text-xs text-gray-400 truncate">Professional Account</p>
              </div>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
}
