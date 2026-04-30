export type IndexedDbWriteWorkerPort = Pick<
	Worker,
	"postMessage" | "terminate" | "onmessage"
>;

export type IndexedDbWriteWorkerFactory = (
	source: string,
) => IndexedDbWriteWorkerPort;

export function createDefaultIndexedDbWriteWorkerFactory(
	source: string,
): IndexedDbWriteWorkerPort {
	return new Worker(
		URL.createObjectURL(new Blob([source], { type: "application/javascript" })),
	);
}

export function buildIndexedDbWriteWorkerSource(
	contentHashIndex: string,
	version: number,
): string {
	return `
      const CONTENT_HASH_INDEX = ${JSON.stringify(contentHashIndex)};
      const DB_VERSION = ${version};

      let dbPromise = null;

      function requestToPromise(request){
        return new Promise((resolve,reject)=>{
          request.onsuccess = ()=> resolve(request.result);
          request.onerror = ()=> reject(request.error || new Error('IndexedDB request failed'));
        });
      }

      function transactionDone(transaction){
        return new Promise((resolve,reject)=>{
          transaction.oncomplete = ()=> resolve();
          transaction.onerror = ()=> reject(transaction.error || new Error('IndexedDB transaction failed'));
          transaction.onabort = ()=> reject(transaction.error || new Error('IndexedDB transaction aborted'));
        });
      }

      function openDb(dbName, storeName, version){
        if (dbPromise) return dbPromise;
        dbPromise = new Promise((resolve,reject)=>{
          const request = indexedDB.open(dbName, version || DB_VERSION);
          request.onupgradeneeded = ()=>{
            const database = request.result;
            if (database.objectStoreNames.contains(storeName)){
              try{
                const store = request.transaction?.objectStore(storeName);
                if (store && typeof store.indexNames?.contains === 'function' && !store.indexNames.contains(CONTENT_HASH_INDEX)){
                  store.createIndex(CONTENT_HASH_INDEX, 'contentHash', { unique: false });
                }
              }catch(e){}
              return;
            }
            const created = database.createObjectStore(storeName, { keyPath: 'id' });
            try{ created.createIndex(CONTENT_HASH_INDEX, 'contentHash', { unique: false }); }catch(e){}
          };
          request.onsuccess = ()=> resolve(request.result);
          request.onerror = ()=> reject(request.error || new Error('Failed to open IndexedDB in worker'));
        });
        return dbPromise;
      }

      self.onmessage = async (ev)=>{
        const msg = ev.data;
        try{
          if (!msg) return;
          if (msg.type === 'putBatch'){
            const db = await openDb(msg.dbName, msg.storeName, msg.version);
            const records = msg.records || [];
            const chunkSize = msg.chunkSize || 64;
            for (let i = 0; i < records.length; i += chunkSize){
              const end = Math.min(i + chunkSize, records.length);
              const tx = db.transaction(msg.storeName, 'readwrite');
              const store = tx.objectStore(msg.storeName);
              for (let j = i; j < end; j++){
                const r = records[j];
                let embedding = null;
                if (r && r.embedding && r.embedding.type === 'buffer'){
                  try{ embedding = new Float32Array(r.embedding.buffer, 0, r.embedding.length); }catch(e){ embedding = new Float32Array(r.embedding.buffer); }
                } else if (r && r.embedding && r.embedding.type === 'array'){
                  embedding = r.embedding.array;
                } else {
                  embedding = r.embedding;
                }

                const recToPut = {
                  id: r.id,
                  content: r.content,
                  embedding: embedding,
                  metadata: r.metadata,
                  contentHash: r.contentHash,
                  cacheKey: r.cacheKey,
                  embeddingSpace: r.embeddingSpace,
                };
                try{ store.put(recToPut); }catch(e){}
              }
              await transactionDone(tx);
            }
            self.postMessage({ type: 'putBatchAck', id: msg.id });
          } else if (msg.type === 'init'){
            await openDb(msg.dbName, msg.storeName, msg.version);
            self.postMessage({ type: 'initAck' });
          } else if (msg.type === 'close'){
            if (dbPromise){
              try{ const db = await dbPromise; db.close(); }catch(e){}
              dbPromise = null;
            }
            self.postMessage({ type: 'closeAck', id: msg.id });
          }
        }catch(err){
          try{ self.postMessage({ type: 'putBatchError', id: msg.id, error: String(err) }); }catch(e){}
        }
      };
    `;
}
