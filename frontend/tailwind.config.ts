import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Pretendard', '-apple-system', 'BlinkMacSystemFont', 'system-ui', 'sans-serif'],
      },
      letterSpacing: {
        tighter: '-0.025em',
      },
      colors: {
        primary: '#000000',
        secondary: '#6366f1',
        border: '#e5e5e5',
        muted: '#737373',
      },
    },
  },
  plugins: [],
}

export default config
