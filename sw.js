// 폰 앱처럼 쓰기: 홈 화면 설치 · 갤러리 '공유'로 사진 받기 · 한 번 받은 인식 모델은 다시 내려받지 않기
const SHELL="mg-shell-v2",MODELS="mg-models-v1",SHARE="mg-share";
self.addEventListener("install",e=>{e.waitUntil(caches.open(SHELL).then(c=>c.addAll(["./","./manifest.webmanifest","./icon-192.png","./icon-512.png"])).then(()=>self.skipWaiting()))});
self.addEventListener("activate",e=>{e.waitUntil(caches.keys().then(ks=>Promise.all(ks.filter(k=>![SHELL,MODELS,SHARE].includes(k)).map(k=>caches.delete(k)))).then(()=>self.clients.claim()))});
// 여러 코어 계산(SharedArrayBuffer)을 쓰려면 페이지가 '격리'돼야 해서, 같은 사이트 응답에 격리 헤더를 붙여요
const iso=r=>{if(!r||r.type==="opaque"||r.status===0)return r;const h=new Headers(r.headers);h.set("Cross-Origin-Opener-Policy","same-origin");h.set("Cross-Origin-Embedder-Policy","credentialless");h.set("Cross-Origin-Resource-Policy","same-origin");return new Response(r.body,{status:r.status,statusText:r.statusText,headers:h})};
self.addEventListener("fetch",e=>{
  const req=e.request,url=new URL(req.url);
  if(url.origin!==location.origin)return;
  // 갤러리에서 '공유 → 카운터'로 보낸 사진: 잠깐 보관했다가 앱 화면에서 꺼내 인식해요
  if(req.method==="POST"&&url.pathname.endsWith("/share")){
    e.respondWith((async()=>{
      const fd=await req.formData(),files=fd.getAll("photos").filter(f=>f&&f.size),c=await caches.open(SHARE);
      for(const k of await c.keys())await c.delete(k);
      await Promise.all(files.map((f,i)=>c.put(`./shared/${i}`,new Response(f,{headers:{"content-type":f.type||"image/jpeg","x-name":encodeURIComponent(f.name||`photo${i+1}.jpg`)}}))));
      return Response.redirect(`./?shared=${files.length}`,303);
    })());
    return;
  }
  if(req.method!=="GET")return;
  // 인식 모델(ocr/, ocr2/)은 크고 잘 안 바뀌어서 한 번 받으면 저장해 두고 써요
  if(/\/ocr2?\//.test(url.pathname)){
    e.respondWith(caches.open(MODELS).then(async c=>{const hit=await c.match(req,{ignoreSearch:false});if(hit)return iso(hit);const r=await fetch(req);if(r.ok)c.put(req,r.clone());return iso(r)}));
    return;
  }
  // 앱 화면은 늘 최신을 먼저 받고, 인터넷이 안 되면 저장해 둔 것을 보여줘요
  if(req.mode==="navigate"||url.pathname.endsWith("/index.html")){
    e.respondWith(fetch(req).then(r=>{if(r.ok){const cp=r.clone();caches.open(SHELL).then(c=>c.put("./",cp))}return iso(r)}).catch(()=>caches.match("./").then(iso)));
    return;
  }
  e.respondWith(fetch(req).then(iso));
});
