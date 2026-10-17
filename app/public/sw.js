// Cache-first service worker: everything the app needs is stored on the phone.
const VERSION = "scat-id-__BUILD__";
const FILES = [
  "./", "index.html", "styles.css", "app.js", "species.js", "photos.js", "manifest.webmanifest",
  "icons/icon-180.png", "icons/icon-192.png", "icons/icon-512.png",
  "ort/ort.wasm.min.js", "ort/ort-wasm-simd-threaded.mjs", "ort/ort-wasm-simd-threaded.wasm",
  "model/model.json", "model/scat.onnx",
  "guide/deer.jpg", "guide/elk.jpg", "guide/bear.jpg", "guide/cougar.jpg", "guide/coyote.jpg", "guide/bobcat.jpg", "guide/grayfox.jpg", "guide/raccoon.jpg", "guide/ringtail.jpg", "guide/opossum.jpg", "guide/skunk.jpg", "guide/porcupine.jpg", "guide/rabbit.jpg", "guide/otter.jpg", "guide/wolf.jpg", "guide/turkey.jpg", "guide/grouse.jpg", "guide/dog.jpg", "guide/cattle.jpg", "guide/horse.jpg",
];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(VERSION).then((c) => c.addAll(FILES)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});
self.addEventListener("fetch", (e) => {
  if (e.request.method !== "GET") return;
  e.respondWith(caches.match(e.request, { ignoreSearch: true }).then((hit) => hit || fetch(e.request)));
});
