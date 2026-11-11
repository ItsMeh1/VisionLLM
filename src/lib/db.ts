/** Small IndexedDB wrapper. Everything the user imports stays in this database. */

const DB_NAME = 'visionllm'
const DB_VERSION = 1

export type StoreName = 'projects' | 'images' | 'runs' | 'agentRuns' | 'settings'

let cached: Promise<IDBDatabase> | null = null

export function openDB(): Promise<IDBDatabase> {
  if (cached) return cached
  cached = new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION)
    request.onupgradeneeded = () => {
      const db = request.result
      if (!db.objectStoreNames.contains('projects')) db.createObjectStore('projects', { keyPath: 'id' })
      if (!db.objectStoreNames.contains('images')) {
        const store = db.createObjectStore('images', { keyPath: 'id' })
        store.createIndex('projectId', 'projectId')
      }
      if (!db.objectStoreNames.contains('runs')) {
        const store = db.createObjectStore('runs', { keyPath: 'id' })
        store.createIndex('projectId', 'projectId')
      }
      if (!db.objectStoreNames.contains('agentRuns')) {
        const store = db.createObjectStore('agentRuns', { keyPath: 'id' })
        store.createIndex('projectId', 'projectId')
      }
      if (!db.objectStoreNames.contains('settings')) db.createObjectStore('settings', { keyPath: 'key' })
    }
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error)
  })
  return cached
}

function wrap<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error)
  })
}

export async function put<T>(store: StoreName, value: T): Promise<T> {
  const db = await openDB()
  await wrap(db.transaction(store, 'readwrite').objectStore(store).put(value as unknown as object))
  return value
}

export async function putMany<T>(store: StoreName, values: T[]): Promise<void> {
  if (!values.length) return
  const db = await openDB()
  const tx = db.transaction(store, 'readwrite')
  const objectStore = tx.objectStore(store)
  for (const value of values) objectStore.put(value as unknown as object)
  await new Promise<void>((resolve, reject) => {
    tx.oncomplete = () => resolve()
    tx.onerror = () => reject(tx.error)
  })
}

export async function get<T>(store: StoreName, key: string): Promise<T | undefined> {
  const db = await openDB()
  return wrap(db.transaction(store, 'readonly').objectStore(store).get(key) as IDBRequest<T | undefined>)
}

export async function getAll<T>(store: StoreName): Promise<T[]> {
  const db = await openDB()
  return wrap(db.transaction(store, 'readonly').objectStore(store).getAll() as IDBRequest<T[]>)
}

export async function getAllByProject<T>(store: StoreName, projectId: string): Promise<T[]> {
  const db = await openDB()
  const index = db.transaction(store, 'readonly').objectStore(store).index('projectId')
  return wrap(index.getAll(projectId) as IDBRequest<T[]>)
}

export async function del(store: StoreName, key: string): Promise<void> {
  const db = await openDB()
  await wrap(db.transaction(store, 'readwrite').objectStore(store).delete(key))
}

export async function delMany(store: StoreName, keys: string[]): Promise<void> {
  if (!keys.length) return
  const db = await openDB()
  const tx = db.transaction(store, 'readwrite')
  const objectStore = tx.objectStore(store)
  for (const key of keys) objectStore.delete(key)
  await new Promise<void>((resolve, reject) => {
    tx.oncomplete = () => resolve()
    tx.onerror = () => reject(tx.error)
  })
}

export async function clearStore(store: StoreName): Promise<void> {
  const db = await openDB()
  await wrap(db.transaction(store, 'readwrite').objectStore(store).clear())
}

export async function estimateStorage(): Promise<{ usage: number; quota: number } | null> {
  if (!navigator.storage?.estimate) return null
  const { usage = 0, quota = 0 } = await navigator.storage.estimate()
  return { usage, quota }
}
