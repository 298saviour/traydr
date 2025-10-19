import { ReactNode } from 'react';
import { LucideIcon } from 'lucide-react';

interface StatCardProps {
  title: string;
  value: string | number;
  icon: LucideIcon;
  trend?: {
    value: string;
    isPositive: boolean;
  };
  iconColor?: string;
}

export default function StatCard({ title, value, icon: Icon, trend, iconColor = 'text-blue-500' }: StatCardProps) {
  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700 p-6 hover:border-blue-500 transition-all duration-200 hover:shadow-lg">
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-gray-400 mb-1">{title}</p>
          <p className="text-3xl font-bold text-white">{value}</p>
          {trend && (
            <p className={`text-sm mt-2 ${trend.isPositive ? 'text-green-500' : 'text-red-500'}`}>
              {trend.value}
            </p>
          )}
        </div>
        <div className={`p-3 rounded-lg bg-gray-900 ${iconColor}`}>
          <Icon className="w-8 h-8" />
        </div>
      </div>
    </div>
  );
}
