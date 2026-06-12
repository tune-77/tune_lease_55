"use client";

import { useEffect } from "react";

export default function ServiceWorkerCleanup() {
  useEffect(() => {
    try {
      if ("serviceWorker" in navigator) {
        navigator.serviceWorker
          .getRegistrations()
          .then((registrations) => {
            registrations.forEach((registration) => {
              registration.unregister();
            });
          })
          .catch(() => {});
      }
      if ("caches" in window) {
        caches
          .keys()
          .then((keys) => {
            keys.forEach((key) => {
              caches.delete(key);
            });
          })
          .catch(() => {});
      }
    } catch {
      // ignore
    }
  }, []);

  return null;
}
