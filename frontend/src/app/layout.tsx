import './globals.css';
import type { Metadata, Viewport } from 'next';
import { Inter } from 'next/font/google';
import Sidebar from '@/components/layout/Sidebar';
import FloatingMebuki from '@/components/layout/FloatingMebuki';
import MobileHeader from '@/components/layout/MobileHeader';
import { SidebarProvider } from '@/context/SidebarContext';
import ContentWrapper from '@/components/layout/ContentWrapper';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: '温水式 リース審査アシスタント',
  description: 'AI駆動のリース与信判定・分析システム',
  manifest: '/manifest.json',
  appleWebApp: {
    capable: true,
    statusBarStyle: 'black-translucent',
    title: 'リース審査AI',
  },
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 5,
  viewportFit: 'cover',
  themeColor: '#3b82f6',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ja">
      <body className={`${inter.className} bg-[#f8fafc] text-slate-800 antialiased`}>
        <SidebarProvider>
          <div className="flex min-h-screen relative overflow-x-hidden">
            {/* 左側のサイドバー */}
            <Sidebar />
            
            <div className="flex-1 flex flex-col min-w-0">
              {/* スマホ画面のみ表示されるヘッダー */}
              <MobileHeader />

              {/* メインコンテンツエリア (ContentWrapperでマージン制御) */}
              <ContentWrapper>
                {children}
              </ContentWrapper>
            </div>
            
            {/* 全画面共通フローティングBOT (めぶきちゃん) */}
            <FloatingMebuki />
          </div>
        </SidebarProvider>
      </body>
    </html>
  );
}
