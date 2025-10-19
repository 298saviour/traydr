/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:5000/api/:path*',
      },
      {
        source: '/analyze',
        destination: 'http://localhost:5000/analyze',
      },
      {
        source: '/stop',
        destination: 'http://localhost:5000/stop',
      },
      {
        source: '/status',
        destination: 'http://localhost:5000/status',
      },
    ];
  },
};

export default nextConfig;
