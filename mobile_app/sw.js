const CACHE_NAME = 'mebuki-v1';
const STATIC_ASSETS = ['/', '/index.html', '/mebuki.png'];

// API パスは常にネットワーク優先（キャッシュしない）
const NETWORK_ONLY_PATTERNS = [
  /^\/predict/,
  /^\/cases/,
  /^\/chat/,
  /^\/advisor/,
  /^\/health/,
  /^\/manifest\.json/,
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(STATIC_ASSETS))
  );
  self.skipWaiting();
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);

  // 同一オリジン以外はスルー
  if (url.origin !== self.location.origin) return;

  // API パスはネットワークのみ
  if (NETWORK_ONLY_PATTERNS.some(p => p.test(url.pathname))) {
    event.respondWith(fetch(event.request));
    return;
  }

  // 静的アセット: cache-first
  event.respondWith(
    caches.match(event.request).then(cached => {
      if (cached) return cached;
      return fetch(event.request).then(response => {
        if (response.ok) {
          const clone = response.clone();
          caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
        }
        return response;
      });
    })
  );
});
