import { type ClassValue, clsx } from 'clsx';

export function cn(...inputs: ClassValue[]) {
  return clsx(inputs);
}

export function formatDate(dateString: string): string {
  try {
    const date = new Date(dateString);
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return dateString;
  }
}

export function formatCurrency(value: number, decimals: number = 2): string {
  return value.toFixed(decimals);
}

export function formatPercentage(value: number, decimals: number = 1): string {
  return `${value.toFixed(decimals)}%`;
}

export function parseConfidence(confidence: any): number {
  if (!confidence) return 0;
  
  if (typeof confidence === 'number') {
    if (confidence <= 1) return Math.round(confidence * 100);
    if (confidence <= 100) return Math.round(confidence);
    return 0;
  }
  
  const str = String(confidence).trim();
  
  // Handle percentage strings
  if (str.endsWith('%')) {
    return parseInt(str.replace('%', '')) || 0;
  }
  
  // Handle fraction strings
  if (str.includes('/')) {
    const [num, den] = str.split('/');
    const n = parseFloat(num);
    const d = parseFloat(den || '10');
    if (d > 0) return Math.round((n / d) * 100);
  }
  
  // Handle text values
  const textMap: Record<string, number> = {
    low: 33,
    medium: 66,
    med: 66,
    high: 85,
  };
  
  if (textMap[str.toLowerCase()]) {
    return textMap[str.toLowerCase()];
  }
  
  // Try parsing as float
  const parsed = parseFloat(str);
  if (!isNaN(parsed)) {
    return parsed <= 1 ? Math.round(parsed * 100) : Math.round(parsed);
  }
  
  return 0;
}

export function getStatusColor(status: string): string {
  switch (status?.toLowerCase()) {
    case 'active':
      return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
    case 'hit_tp':
      return 'bg-green-500/20 text-green-400 border-green-500/30';
    case 'hit_sl':
      return 'bg-red-500/20 text-red-400 border-red-500/30';
    case 'expired':
      return 'bg-gray-500/20 text-gray-400 border-gray-500/30';
    default:
      return 'bg-gray-500/20 text-gray-400 border-gray-500/30';
  }
}

export function getRecommendationColor(recommendation: string): string {
  if (recommendation === 'BUY') return 'bg-green-500/20 text-green-400 border-green-500/30';
  if (recommendation === 'SELL') return 'bg-red-500/20 text-red-400 border-red-500/30';
  return 'bg-gray-500/20 text-gray-400 border-gray-500/30';
}
