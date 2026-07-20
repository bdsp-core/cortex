// Tiny IndexedDB blob cache for the EEG/spectrogram bundle. The bundle is
// large (up to ~1 GB at 500 segments) and immutable per version, so caching
// the raw ArrayBuffers across reloads avoids re-downloading on every sitting
// and lets a refreshed tab resume instantly (see LOCAL_DEV.md).
//
// Keys are namespaced by bundle version so a new bundle doesn't collide with a
// stale cache. Failures degrade gracefully to network fetches — the cache is
// an optimization, never a correctness dependency.

const DB_NAME = "cortex-bundle";
const STORE = "blobs";
const DB_VERSION = 1;

let _dbPromise: Promise<IDBDatabase | null> | null = null;

function openDb(): Promise<IDBDatabase | null> {
  if (_dbPromise) return _dbPromise;
  _dbPromise = new Promise((resolve) => {
    if (typeof indexedDB === "undefined") return resolve(null);
    let req: IDBOpenDBRequest;
    try {
      req = indexedDB.open(DB_NAME, DB_VERSION);
    } catch {
      return resolve(null);
    }
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE);
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => resolve(null);
  });
  return _dbPromise;
}

export async function idbGet(key: string): Promise<ArrayBuffer | null> {
  const db = await openDb();
  if (!db) return null;
  return new Promise((resolve) => {
    try {
      const tx = db.transaction(STORE, "readonly");
      const req = tx.objectStore(STORE).get(key);
      req.onsuccess = () => resolve((req.result as ArrayBuffer) ?? null);
      req.onerror = () => resolve(null);
    } catch {
      resolve(null);
    }
  });
}

export async function idbPut(key: string, value: ArrayBuffer): Promise<void> {
  const db = await openDb();
  if (!db) return;
  return new Promise((resolve) => {
    try {
      const tx = db.transaction(STORE, "readwrite");
      tx.objectStore(STORE).put(value, key);
      tx.oncomplete = () => resolve();
      tx.onerror = () => resolve();
      tx.onabort = () => resolve();
    } catch {
      resolve();
    }
  });
}

// Fetch an ArrayBuffer, preferring the IndexedDB cache. On a cache miss the
// network result is stored for next time. Network errors propagate (the
// caller needs the bytes); cache errors are swallowed.
export async function cachedArrayBuffer(url: string, cacheKey: string): Promise<ArrayBuffer> {
  const hit = await idbGet(cacheKey);
  if (hit) return hit;
  const res = await fetch(url);
  // A transient 404/500 returns an HTML/JSON error body; without this check it
  // would be decoded as Int16Array EEG AND written to IndexedDB under the
  // version key — permanently serving garbage for that segment. Only cache a
  // genuine 2xx response.
  if (!res.ok) throw new Error(`fetch ${url} failed (${res.status})`);
  const buf = await res.arrayBuffer();
  void idbPut(cacheKey, buf);
  return buf;
}
