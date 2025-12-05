import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Prometheus',
  description: 'Identity-Preserving Image Generation',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="ko">
      <body className="antialiased">
        {children}
      </body>
    </html>
  )
}
