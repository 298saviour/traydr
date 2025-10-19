'use client';

import { useState, useEffect } from 'react';
import PageHeader from '@/components/PageHeader';
import StatCard from '@/components/StatCard';
import Card, { CardBody, CardHeader, CardTitle } from '@/components/Card';
import { Activity, TrendingUp, Target, Award, CheckCircle, XCircle, Clock, AlertCircle } from 'lucide-react';

interface PerformanceData {
  total_signals: number;
  win_rate: number;
  net_pips: number;
  avg_confidence: number;
  status_breakdown: {
    hit_tp: number;
    hit_sl: number;
    active: number;
    expired: number;
  };
  pair_performance: Array<{
    pair: string;
    total: number;
    wins: number;
    losses: number;
    win_rate: number;
  }>;
}

export default function PerformancePage() {
  const [data, setData] = useState<PerformanceData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadPerformanceData();
  }, []);

  const loadPerformanceData = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/performance', { cache: 'no-store' });
      const result = await response.json();
      
      if (result.success) {
        setData(result.data);
      }
    } catch (error) {
      console.error('Error loading performance data:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-900">
        <PageHeader
          title="Performance Analytics"
          subtitle="Track your signal performance metrics"
        />
        <div className="p-6 lg:p-8">
          <Card>
            <CardBody className="text-center py-12">
              <div className="animate-spin w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full mx-auto mb-4" />
              <p className="text-gray-400">Loading performance data...</p>
            </CardBody>
          </Card>
        </div>
      </div>
    );
  }

  const totalSignals = data?.total_signals || 0;
  const winRate = data?.win_rate || 0;
  const netPips = data?.net_pips || 0;
  const avgConfidence = data?.avg_confidence || 0;
  const statusBreakdown = data?.status_breakdown || { hit_tp: 0, hit_sl: 0, active: 0, expired: 0 };
  const pairPerformance = data?.pair_performance || [];

  return (
    <div className="min-h-screen bg-gray-900">
      <PageHeader
        title="Performance Analytics"
        subtitle="Track your signal performance metrics"
      />

      <div className="p-6 lg:p-8 space-y-8">
        {/* Key Metrics */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          <StatCard
            title="Total Signals"
            value={totalSignals}
            icon={Activity}
            iconColor="text-blue-500"
          />
          <StatCard
            title="Win Rate"
            value={isNaN(winRate) ? '0%' : `${winRate.toFixed(1)}%`}
            icon={TrendingUp}
            iconColor="text-green-500"
            trend={winRate > 50 ? { value: 'Above average', isPositive: true } : undefined}
          />
          <StatCard
            title="Net Pips"
            value={isNaN(netPips) ? '0' : netPips.toFixed(0)}
            icon={Target}
            iconColor="text-yellow-500"
          />
          <StatCard
            title="Avg Confidence"
            value={isNaN(avgConfidence) ? '0/10' : `${avgConfidence.toFixed(1)}/10`}
            icon={Award}
            iconColor="text-purple-500"
          />
        </div>

        {/* Signal Status Breakdown */}
        <Card>
          <CardHeader>
            <CardTitle>Signal Status Breakdown</CardTitle>
          </CardHeader>
          <CardBody>
            <div className="space-y-4">
              <StatusItem
                label="Hit Take Profit"
                count={statusBreakdown.hit_tp}
                total={totalSignals}
                color="green"
                icon={CheckCircle}
              />
              <StatusItem
                label="Hit Stop Loss"
                count={statusBreakdown.hit_sl}
                total={totalSignals}
                color="red"
                icon={XCircle}
              />
              <StatusItem
                label="Active"
                count={statusBreakdown.active}
                total={totalSignals}
                color="blue"
                icon={Activity}
              />
              <StatusItem
                label="Expired"
                count={statusBreakdown.expired}
                total={totalSignals}
                color="gray"
                icon={Clock}
              />
            </div>
          </CardBody>
        </Card>

        {/* Pair Performance */}
        {pairPerformance.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Performance by Currency Pair</CardTitle>
            </CardHeader>
            <CardBody>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-gray-700">
                      <th className="text-left py-3 px-4 text-sm font-semibold text-gray-400">Pair</th>
                      <th className="text-center py-3 px-4 text-sm font-semibold text-gray-400">Total</th>
                      <th className="text-center py-3 px-4 text-sm font-semibold text-gray-400">Wins</th>
                      <th className="text-center py-3 px-4 text-sm font-semibold text-gray-400">Losses</th>
                      <th className="text-center py-3 px-4 text-sm font-semibold text-gray-400">Win Rate</th>
                    </tr>
                  </thead>
                  <tbody>
                    {pairPerformance.map((pair, index) => (
                      <tr key={index} className="border-b border-gray-700/50 hover:bg-gray-700/30">
                        <td className="py-3 px-4 font-semibold text-white">{pair.pair}</td>
                        <td className="py-3 px-4 text-center text-gray-300">{pair.total}</td>
                        <td className="py-3 px-4 text-center text-green-400">{pair.wins}</td>
                        <td className="py-3 px-4 text-center text-red-400">{pair.losses}</td>
                        <td className="py-3 px-4 text-center">
                          <span className={`px-2 py-1 rounded-full text-xs font-semibold ${
                            pair.win_rate >= 60 ? 'bg-green-500/20 text-green-400' :
                            pair.win_rate >= 40 ? 'bg-yellow-500/20 text-yellow-400' :
                            'bg-red-500/20 text-red-400'
                          }`}>
                            {pair.win_rate.toFixed(1)}%
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardBody>
          </Card>
        )}

        {/* Empty State */}
        {totalSignals === 0 && (
          <Card>
            <CardBody className="text-center py-12">
              <AlertCircle className="w-16 h-16 text-gray-600 mx-auto mb-4" />
              <h3 className="text-xl font-semibold text-white mb-2">No Performance Data Yet</h3>
              <p className="text-gray-400">
                Start generating signals to see your performance analytics here
              </p>
            </CardBody>
          </Card>
        )}
      </div>
    </div>
  );
}

interface StatusItemProps {
  label: string;
  count: number;
  total: number;
  color: 'green' | 'red' | 'blue' | 'gray';
  icon: React.ElementType;
}

function StatusItem({ label, count, total, color, icon: Icon }: StatusItemProps) {
  const percentage = total > 0 ? ((count / total) * 100).toFixed(1) : '0.0';
  
  const colorClasses = {
    green: 'bg-green-500',
    red: 'bg-red-500',
    blue: 'bg-blue-500',
    gray: 'bg-gray-500',
  };

  const bgColorClasses = {
    green: 'bg-green-500/10',
    red: 'bg-red-500/10',
    blue: 'bg-blue-500/10',
    gray: 'bg-gray-500/10',
  };

  return (
    <div className={`flex items-center gap-4 p-4 rounded-lg ${bgColorClasses[color]}`}>
      <div className={`w-3 h-3 rounded-full ${colorClasses[color]}`} />
      <div className="flex-1">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-300">{label}</span>
          <div className="flex items-center gap-3">
            <span className="text-2xl font-bold text-white">{count}</span>
            <span className="text-sm text-gray-400">{percentage}%</span>
          </div>
        </div>
        <div className="w-full bg-gray-700 rounded-full h-2">
          <div
            className={`h-2 rounded-full ${colorClasses[color]}`}
            style={{ width: `${percentage}%` }}
          />
        </div>
      </div>
    </div>
  );
}
