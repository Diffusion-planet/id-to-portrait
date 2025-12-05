/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    remotePatterns: [
      {
        protocol: 'http',
        hostname: 'localhost',
        port: '8007',
        pathname: '/**',
      },
    ],
  },
}

module.exports = nextConfig
