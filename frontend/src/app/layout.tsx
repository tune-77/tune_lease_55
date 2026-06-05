import './globals.css';
import type { Metadata, Viewport } from 'next';
import Script from 'next/script';
import Sidebar from '@/components/layout/Sidebar';
import FloatingMebuki from '@/components/layout/FloatingMebuki';
import MobileHeader from '@/components/layout/MobileHeader';
import { SidebarProvider } from '@/context/SidebarContext';
import { ThemeProvider } from '@/context/ThemeContext';
import ContentWrapper from '@/components/layout/ContentWrapper';

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
    <html lang="ja" suppressHydrationWarning>
      <body className="text-slate-800 antialiased">
        <Script id="sw-cleanup" strategy="beforeInteractive">{`
          (function () {
            try {
              if ('serviceWorker' in navigator) {
                navigator.serviceWorker.getRegistrations().then(function (registrations) {
                  registrations.forEach(function (registration) {
                    registration.unregister();
                  });
                }).catch(function () {});
              }
              if ('caches' in window) {
                caches.keys().then(function (keys) {
                  keys.forEach(function (key) {
                    caches.delete(key);
                  });
                }).catch(function () {});
              }
            } catch (e) {}
          })();
        `}</Script>
        <ThemeProvider>
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
        </ThemeProvider>
      </body>
    </html>
  );
}
