/* 오픈소스 글자 인식 (PaddleOCR PP-OCRv4 모델 + onnxruntime-web) — 이 기기 안에서만 돌아가요.
 * 1) det.onnx 로 글자가 있는 줄을 찾고  2) rec.onnx 로 각 줄을 읽어요.
 * window.PPOCR.load(say) / PPOCR.recognize(canvas, {detSide}) → [{text,x,y,w,h,conf}] (canvas 좌표)
 */
(function(){
  const BASE=document.currentScript.src.replace(/[^\/]*$/,"");
  let det=null,rec=null,dict=null,ready=null;
  const load=(say)=>ready||(ready=(async()=>{
    if(!window.ort){await new Promise((ok,no)=>{const s=document.createElement("script");s.src=BASE+"ort.wasm.min.js";s.onload=ok;s.onerror=()=>no(new Error("onnxruntime 을 불러오지 못했어요"));document.head.appendChild(s)})}
    ort.env.wasm.wasmPaths=BASE;ort.env.wasm.numThreads=1;ort.env.wasm.proxy=true;   // proxy: 화면이 멈추지 않게 별도 스레드에서 계산
    say&&say("글자 인식 모델을 내려받는 중이에요 (처음 한 번, 약 26MB)...");
    const get=async(f,t)=>{const r=await fetch(BASE+f);if(!r.ok)throw new Error(f+" 다운로드 실패 "+r.status);return t?r.text():r.arrayBuffer()};
    const [d,r,k]=await Promise.all([get("det.onnx"),get("rec.onnx"),get("keys.txt",1)]);
    det=await ort.InferenceSession.create(d,{executionProviders:["wasm"],graphOptimizationLevel:"all"});
    rec=await ort.InferenceSession.create(r,{executionProviders:["wasm"],graphOptimizationLevel:"all"});
    const ch=k.split("\n").map(s=>s.replace(/\r$/,""));if(ch[ch.length-1]==="")ch.pop();
    dict=["#blank"].concat(ch,[" "]);
    return true;
  })());

  const m32=v=>Math.max(32,Math.round(v/32)*32);
  function toBGR(canvas,w,h,mean,std,half){      // 캔버스 → [1,3,h,w] float (BGR 순서, PaddleOCR 방식)
    const c=document.createElement("canvas");c.width=w;c.height=h;
    const x=c.getContext("2d",{willReadFrequently:true});x.fillStyle="#fff";x.fillRect(0,0,w,h);x.drawImage(canvas,0,0,w,h);
    const d=x.getImageData(0,0,w,h).data,n=w*h,t=new Float32Array(3*n);
    for(let i=0,p=0;i<n;i++,p+=4){
      const r=d[p]/255,g=d[p+1]/255,b=d[p+2]/255;
      if(half){t[i]=(b-.5)/.5;t[n+i]=(g-.5)/.5;t[2*n+i]=(r-.5)/.5}
      else{t[i]=(b-mean[0])/std[0];t[n+i]=(g-mean[1])/std[1];t[2*n+i]=(r-mean[2])/std[2]}
    }
    return t;
  }

  // 글자 줄 찾기: 확률 지도를 이진화 → 덩어리(연결 성분) → 박스 → 살짝 넓히기
  async function detect(canvas,side){
    const W=canvas.width,H=canvas.height,s=Math.min(1,side/Math.max(W,H)),w=m32(W*s),h=m32(H*s);
    const inp=toBGR(canvas,w,h,[.485,.456,.406],[.229,.224,.225],false);
    const out=await det.run({[det.inputNames[0]]:new ort.Tensor("float32",inp,[1,3,h,w])});
    const P=out[det.outputNames[0]].data,n=w*h,seen=new Uint8Array(n),stack=new Int32Array(n),boxes=[];
    const TH=.3;
    for(let i=0;i<n;i++){
      if(seen[i]||P[i]<=TH)continue;
      let sp=0,cnt=0,sum=0,x0=w,y0=h,x1=0,y1=0,mx=0,my=0,mxx=0,myy=0,mxy=0;stack[sp++]=i;seen[i]=1;
      while(sp){const q=stack[--sp],qx=q%w,qy=(q-qx)/w;cnt++;sum+=P[q];mx+=qx;my+=qy;mxx+=qx*qx;myy+=qy*qy;mxy+=qx*qy;
        if(qx<x0)x0=qx;if(qx>x1)x1=qx;if(qy<y0)y0=qy;if(qy>y1)y1=qy;
        if(qx>0&&!seen[q-1]&&P[q-1]>TH){seen[q-1]=1;stack[sp++]=q-1}
        if(qx<w-1&&!seen[q+1]&&P[q+1]>TH){seen[q+1]=1;stack[sp++]=q+1}
        if(qy>0&&!seen[q-w]&&P[q-w]>TH){seen[q-w]=1;stack[sp++]=q-w}
        if(qy<h-1&&!seen[q+w]&&P[q+w]>TH){seen[q+w]=1;stack[sp++]=q+w}}
      const bw=x1-x0+1,bh=y1-y0+1;
      if(cnt<12||Math.min(bw,bh)<3||sum/cnt<.55)continue;
      const dist=(bw*bh)*1.6/(2*(bw+bh)),k=1/s;                    // DB 후처리의 unclip
      const px=Math.min(bh*.35,dist);   // 줄 끝 글자가 잘리지 않도록 가로로 조금 더 넓혀요
      const box={x0:Math.max(0,(x0-dist-px)*k),y0:Math.max(0,(y0-dist)*k),x1:Math.min(W,(x1+dist+px)*k),y1:Math.min(H,(y1+dist)*k)};
      // 비스듬한 글줄: 픽셀 분포의 기울기(2차 모멘트)로 각도와 길이·두께를 구해, 읽을 때 똑바로 돌려서 잘라요
      const ax=mx/cnt,ay=my/cnt,cxx=mxx/cnt-ax*ax,cyy=myy/cnt-ay*ay,cxy=mxy/cnt-ax*ay;
      const ang=.5*Math.atan2(2*cxy,cxx-cyy),tr=(cxx+cyy)/2,dd=Math.sqrt(Math.max(0,(cxx-cyy)*(cxx-cyy)/4+cxy*cxy));
      const L=Math.sqrt(12*(tr+dd)),T=Math.sqrt(12*Math.max(0,tr-dd));
      if(Math.abs(ang)>.05&&Math.abs(ang)<.7&&T>=2&&L>T*2.5){const d2=L*T*1.6/(2*(L+T));
        box.ang=ang;box.cx=ax*k;box.cy=ay*k;box.L=(L+2*d2+2*Math.min(T*.35,d2))*k;box.T=(T+2*d2)*k}
      boxes.push(box);
    }
    return boxes;
  }

  async function readLine(canvas,b){
    let bw=b.x1-b.x0,bh=b.y1-b.y0;if(bw<4||bh<4)return null;
    const vert=bh>bw*1.6;                                           // 세로로 서 있는 글자는 눕혀서 읽어요
    const c=document.createElement("canvas"),H=48;
    const sw=vert?bh:bw,sh=vert?bw:bh,W=Math.min(1600,Math.max(24,Math.ceil(H*sw/sh)));
    c.width=W;c.height=H;const x=c.getContext("2d",{willReadFrequently:true});x.fillStyle="#fff";x.fillRect(0,0,W,H);
    if(b.ang!=null){const Wr=Math.min(1600,Math.max(24,Math.ceil(H*b.L/b.T)));c.width=Wr;x.fillStyle="#fff";x.fillRect(0,0,Wr,H);
      x.translate(Wr/2,H/2);x.scale(Wr/b.L,H/b.T);x.rotate(-b.ang);x.translate(-b.cx,-b.cy);x.drawImage(canvas,0,0);x.setTransform(1,0,0,1,0,0);
      const inp=toBGR(c,Wr,H,null,null,true);return await runRec(inp,Wr,H)}
    if(vert){x.translate(0,H);x.rotate(-Math.PI/2);x.drawImage(canvas,b.x0,b.y0,bw,bh,0,0,H,W)}
    else x.drawImage(canvas,b.x0,b.y0,bw,bh,0,0,W,H);
    const inp=toBGR(c,W,H,null,null,true);
    return await runRec(inp,W,H);
  }
  async function runRec(inp,W,H){
    const out=await rec.run({[rec.inputNames[0]]:new ort.Tensor("float32",inp,[1,3,H,W])});
    const o=out[rec.outputNames[0]],T=o.dims[1],C=o.dims[2],D=o.data;
    let last=-1,txt="",cs=0,cn=0;const pos=[];
    for(let t=0;t<T;t++){let bi=0,bv=-1;const off=t*C;for(let k=0;k<C;k++){const v=D[off+k];if(v>bv){bv=v;bi=k}}
      if(bi!==0&&bi!==last){const ch=dict[bi]||"";txt+=ch;for(let j=0;j<ch.length;j++)pos.push((t+.5)/T);cs+=bv;cn++}last=bi}
    return {text:txt,conf:cn?cs/cn:0,pos};   // pos: 글자마다 줄 안에서의 가로 위치(0~1)
  }

  async function recognize(canvas,opt){
    await load();opt=opt||{};
    const boxes=await detect(canvas,opt.detSide||1920),lines=[];
    boxes.sort((a,b)=>a.y0-b.y0||a.x0-b.x0);
    for(const b of boxes){
      let r=null;try{r=await readLine(canvas,b)}catch(e){continue}
      if(r&&r.text.trim()&&r.conf>=.3)lines.push({text:r.text,conf:r.conf,x:(b.x0+b.x1)/2,y:(b.y0+b.y1)/2,w:b.x1-b.x0,h:b.y1-b.y0,pos:b.ang==null&&!(b.y1-b.y0>(b.x1-b.x0)*1.6)?r.pos:null});
    }
    return lines;
  }
  // ---------- 손글씨 숫자 찾기 ----------
  // 인쇄 글자·바코드가 아닌 '손글씨 잉크 덩어리'를 찾고, 한 글자씩 잘라 작은 숫자 인식기(공개 MNIST 손글씨 데이터로 학습)로 읽어요.
  // printed: recognize() 가 읽은 줄들, exclude: [{x,y,r}] (바코드 위치) — 모두 canvas 좌표
  let DG=null;
  async function loadDigits(){
    if(DG)return DG;
    const j=await (await fetch(BASE+"digits.json")).json();
    const un=b64=>{const s=atob(b64),a=new Int8Array(s.length);for(let i=0;i<s.length;i++)a[i]=s.charCodeAt(i)<<24>>24;return a};
    DG={h:j.h,s1:j.s1,s2:j.s2,b1:Float32Array.from(j.b1),b2:Float32Array.from(j.b2),w1:un(j.w1),w2:un(j.w2)};return DG;
  }
  // 한 글자(잉크 픽셀 목록) → 28x28 (MNIST와 같은 방식: 20px 로 줄여서 무게중심을 가운데로)
  function prep28(b,W){
    const w=b.x1-b.x0,h=b.y1-b.y0,c=document.createElement("canvas");c.width=w;c.height=h;
    const x=c.getContext("2d"),im=x.createImageData(w,h);
    for(const k of b.pix){const px=(k%W)-b.x0,py=((k/W)|0)-b.y0,o=(py*w+px)*4;im.data[o]=im.data[o+1]=im.data[o+2]=255;im.data[o+3]=255}
    x.putImageData(im,0,0);
    const sc=20/Math.max(w,h),nw=Math.max(1,Math.round(w*sc)),nh=Math.max(1,Math.round(h*sc)),c2=document.createElement("canvas");c2.width=nw;c2.height=nh;
    const x2=c2.getContext("2d",{willReadFrequently:true});x2.fillStyle="#000";x2.fillRect(0,0,nw,nh);x2.imageSmoothingQuality="high";x2.drawImage(c,0,0,nw,nh);
    const d=x2.getImageData(0,0,nw,nh).data;let sum=0,cy=0,cx=0;
    for(let i=0;i<nh;i++)for(let j=0;j<nw;j++){const v=d[(i*nw+j)*4]/255;sum+=v;cy+=v*i;cx+=v*j}
    if(!sum)return null;cy/=sum;cx/=sum;
    const out=new Float32Array(784),oy=Math.round(14-cy),ox=Math.round(14-cx);
    for(let i=0;i<nh;i++)for(let j=0;j<nw;j++){const yy=i+oy,xx=j+ox;if(yy>=0&&yy<28&&xx>=0&&xx<28)out[yy*28+xx]=d[(i*nw+j)*4]/255}
    return out;
  }
  // 사용자가 고쳐 준 글자 예시 (kNN): [{l:'0'..'9' 또는 'x'(숫자 아님), v:Float32Array(784)}]
  let USER=[];
  function setUserSamples(arr){USER=(arr||[]).map(a=>{let n=0;for(let i=0;i<784;i++)n+=a.v[i]*a.v[i];return {l:a.l,v:a.v,n:Math.sqrt(n)||1}})}
  function knn(v){
    if(USER.length<3)return null;
    let vn=0;for(let i=0;i<784;i++)vn+=v[i]*v[i];vn=Math.sqrt(vn)||1;
    const sims=USER.map(u=>{let d=0;for(let i=0;i<784;i++)d+=u.v[i]*v[i];return {l:u.l,s:d/(u.n*vn)}}).sort((a,b)=>b.s-a.s).slice(0,3);
    if(sims[0].s<.8)return null;                       // 비슷한 예시가 없으면 기본 모델에 맡겨요
    const w={};for(const t of sims){if(t.s>=sims[0].s-.08)w[t.l]=(w[t.l]||0)+t.s}
    let bl=null,bv=-1;for(const k in w)if(w[k]>bv){bv=w[k];bl=k}
    return {l:bl,s:sims[0].s};
  }
  function classify1(v){
    const m=DG,h=new Float32Array(m.h);
    for(let i=0;i<784;i++){const a=v[i];if(!a)continue;const o=i*m.h;for(let k=0;k<m.h;k++)h[k]+=a*m.w1[o+k]*m.s1}
    for(let k=0;k<m.h;k++){h[k]+=m.b1[k];if(h[k]<0)h[k]=0}
    const z=new Float32Array(10);for(let c=0;c<10;c++){let t=m.b2[c];for(let k=0;k<m.h;k++)t+=h[k]*m.w2[k*10+c]*m.s2;z[c]=t}
    let mx=-1e9;for(const t of z)if(t>mx)mx=t;let sum=0;const p=z.map(t=>{const e=Math.exp(t-mx);sum+=e;return e});
    let bi=0;for(let c=1;c<10;c++)if(p[c]>p[bi])bi=c;
    const base={d:bi,p:p[bi]/sum,src:"model"},u=knn(v);
    if(u){if(u.l==="x")return {d:-1,p:u.s,src:"user"};return {d:+u.l,p:Math.max(u.s,.85),src:"user"}}   // 사용자 예시가 있으면 그걸 우선
    return base;
  }
  async function handwriting(canvas,printed,exclude){
    await load();await loadDigits();
    const W=canvas.width,H=canvas.height,M=Math.max(W,H),n=W*H;
    const d=canvas.getContext("2d",{willReadFrequently:true}).getImageData(0,0,W,H).data,g=new Uint8Array(n);
    for(let i=0,p=0;i<n;i++,p+=4)g[i]=(d[p]*.3+d[p+1]*.59+d[p+2]*.11)|0;
    // 적응형 이진화: 주변 평균보다 확실히 어두운 점 = 잉크 (그림자가 있어도 동작)
    const I=new Float64Array((W+1)*(H+1));
    for(let y=0;y<H;y++){let row=0;for(let x=0;x<W;x++){row+=g[y*W+x];I[(y+1)*(W+1)+x+1]=I[y*(W+1)+x+1]+row}}
    const hw=Math.max(12,Math.round(M*.012)),ink=new Uint8Array(n);
    for(let y=0;y<H;y++){const y0=Math.max(0,y-hw),y1=Math.min(H,y+hw+1);
      for(let x=0;x<W;x++){const x0=Math.max(0,x-hw),x1=Math.min(W,x+hw+1),a=((y1-y0)*(x1-x0)),
        s2=I[y1*(W+1)+x1]-I[y0*(W+1)+x1]-I[y1*(W+1)+x0]+I[y0*(W+1)+x0];
        if(g[y*W+x]<s2/a*.62&&g[y*W+x]<120)ink[y*W+x]=1}}
    const lab=new Uint8Array(n),stack=new Int32Array(n),blobs=[];
    for(let i=0;i<n;i++){
      if(!ink[i]||lab[i])continue;
      let sp=0,cnt=0,x0=W,y0=H,x1=0,y1=0;const pix=[];stack[sp++]=i;lab[i]=1;
      while(sp){const q=stack[--sp],qx=q%W,qy=(q-qx)/W;cnt++;pix.push(q);
        if(qx<x0)x0=qx;if(qx>x1)x1=qx;if(qy<y0)y0=qy;if(qy>y1)y1=qy;
        for(let dy=-1;dy<=1;dy++){const yy=qy+dy;if(yy<0||yy>=H)continue;for(let dx=-1;dx<=1;dx++){const xx=qx+dx;if(xx<0||xx>=W)continue;const k=yy*W+xx;if(ink[k]&&!lab[k]){lab[k]=1;stack[sp++]=k}}}}
      const bw=x1-x0+1,bh=y1-y0+1;
      if(cnt<M*.0012||bh<M*.011||bh>M*.075||bw>M*.075)continue;       // 너무 작은 점·너무 큰 덩어리 제외
      if(cnt/(bw*bh)>.55)continue;                                     // 속이 꽉 찬 덩어리(그림자·바코드) 제외
      if(bw>bh*1.6)continue;                                           // 가로로 긴 것(줄·밑줄·글자 일부) 제외
      blobs.push({x0,y0,x1:x1+1,y1:y1+1,cx:(x0+x1)/2,cy:(y0+y1)/2,h:bh,w:bw,pix,t:cnt/Math.max(bh,bw)});
    }
    const inBox=(b,list)=>list.some(l=>b.cx>l.x-l.w/2&&b.cx<l.x+l.w/2&&b.cy>l.y-l.h/2&&b.cy<l.y+l.h/2);
    const pr=(printed||[]).filter(l=>l.conf>=.6&&l.text.replace(/\s/g,"").length>=3);
    let cand=blobs.filter(b=>!inBox(b,pr)&&!(exclude||[]).some(e=>Math.hypot(b.cx-e.x,b.cy-e.y)<e.r));
    // 같은 높이로 줄지어 늘어선 덩어리 4개 이상 = 인쇄된 글자나 1차원 바코드 → 손글씨가 아니에요 (손글씨 숫자는 보통 1~3자)
    {const sorted=cand.slice().sort((a,b)=>a.cx-b.cx),drop=new Set();
     for(const b of sorted){const chain=[b];let grew=true;
       while(grew){grew=false;for(const o of sorted){if(chain.includes(o))continue;
         if(chain.some(c=>Math.abs(c.cy-o.cy)<Math.max(c.h,o.h)*.4&&Math.abs(c.h-o.h)<Math.max(c.h,o.h)*.3&&Math.abs(c.cx-o.cx)<Math.max(c.h,o.h)*2.6)){chain.push(o);grew=true}}}
       if(chain.length>=4)chain.forEach(c=>drop.add(c))}
     cand=cand.filter(b=>!drop.has(b))}
    // 한 글자씩 읽기
    const ch=[];
    for(const b of cand){const v=prep28(b,W);if(!v)continue;const r=classify1(v);if(r.d<0)continue;if(r.p>=.55)ch.push({...b,d:r.d,p:r.p,v,how:r.src})}
    // 가까운 글자끼리 묶어서 한 자리/두 자리 숫자로 (왼쪽→오른쪽)
    ch.sort((a,b)=>a.x0-b.x0);const groups=[];
    for(const b of ch){
      const G=groups.find(G=>Math.abs(G.cy-b.cy)<Math.max(G.h,b.h)*.6&&b.x0-G.x1<Math.max(G.h,b.h)*.9&&b.x0-G.x1>-Math.max(G.h,b.h)*.3&&G.items.length<2&&Math.abs(G.h-b.h)<Math.max(G.h,b.h)*.45);
      if(G){G.items.push(b);G.x1=Math.max(G.x1,b.x1);G.y0=Math.min(G.y0,b.y0);G.y1=Math.max(G.y1,b.y1);G.h=G.y1-G.y0;G.cy=(G.y0+G.y1)/2}
      else groups.push({items:[b],x0:b.x0,x1:b.x1,y0:b.y0,y1:b.y1,h:b.h,cy:b.cy});
    }
    return groups.map(G=>{const t=G.items.map(i=>i.d).join(""),p=Math.min(...G.items.map(i=>i.p));
      return {text:t,num:+t,conf:p,x:(G.x0+G.x1)/2,y:(G.y0+G.y1)/2,w:G.x1-G.x0,h:G.h,items:G.items.map(i=>({d:i.d,p:i.p,v:i.v,how:i.how}))}});
  }
  window.PPOCR={load,recognize,handwriting,setUserSamples};

  // ---------- 날짜 구간 나누기: 시트에 손으로 그은 가로 구분선 찾기 ----------
  // 길고 가는 (두께 6px 이하) 가로 잉크 선만 찾아요. 반환: [{x0,x1,ya,yb}] (ya: 왼쪽 끝 y, yb: 오른쪽 끝 y) — canvas 좌표
  async function separators(canvas){
    const W=canvas.width,H=canvas.height,n=W*H;
    const d=canvas.getContext("2d",{willReadFrequently:true}).getImageData(0,0,W,H).data,g=new Uint8Array(n);
    for(let i=0,p=0;i<n;i++,p+=4)g[i]=(d[p]*.3+d[p+1]*.59+d[p+2]*.11)|0;
    const I=new Float64Array((W+1)*(H+1));
    for(let y=0;y<H;y++){let row=0;for(let x=0;x<W;x++){row+=g[y*W+x];I[(y+1)*(W+1)+x+1]=I[y*(W+1)+x+1]+row}}
    const hw=Math.max(12,Math.round(Math.max(W,H)*.012)),ink=new Uint8Array(n);
    for(let y=0;y<H;y++){const y0=Math.max(0,y-hw),y1=Math.min(H,y+hw+1);
      for(let x=0;x<W;x++){const x0=Math.max(0,x-hw),x1=Math.min(W,x+hw+1),a=((y1-y0)*(x1-x0)),
        s2=I[y1*(W+1)+x1]-I[y0*(W+1)+x1]-I[y1*(W+1)+x0]+I[y0*(W+1)+x0];
        if(g[y*W+x]<s2/a*.7&&g[y*W+x]<135)ink[y*W+x]=1}}
    // 가로로 끊긴 틈(펜이 떼진 곳) 메우기
    const gap=Math.round(W*.008)+4;
    for(let y=0;y<H;y++){let last=-1e9;for(let x=0;x<W;x++){if(ink[y*W+x]){if(x-last>1&&x-last<=gap)for(let k=last+1;k<x;k++)ink[y*W+k]=2;last=x}}}
    const lab=new Uint8Array(n),stack=new Int32Array(n),out=[];
    for(let i=0;i<n;i++){
      if(!ink[i]||lab[i])continue;
      let sp=0,cnt=0,x0=W,y0=H,x1=0,y1=0;const pts=[];stack[sp++]=i;lab[i]=1;
      while(sp){const q=stack[--sp],qx=q%W,qy=(q-qx)/W;cnt++;pts.push(q);
        if(qx<x0)x0=qx;if(qx>x1)x1=qx;if(qy<y0)y0=qy;if(qy>y1)y1=qy;
        for(let dy=-1;dy<=1;dy++){const yy=qy+dy;if(yy<0||yy>=H)continue;for(let dx=-1;dx<=1;dx++){const xx=qx+dx;if(xx<0||xx>=W)continue;const k=yy*W+xx;if(ink[k]&&!lab[k]){lab[k]=1;stack[sp++]=k}}}}
      const bw=x1-x0+1,bh=y1-y0+1;
      // 곧은 선뿐 아니라 계단처럼 꺾인 선(스티커 둘레를 따라 그은 경계)도 받아요: 선이 가늘고(획 두께 7px 이하) 가로로 충분히 길면 돼요
      const straight=bh<=H*.07&&bw/bh>=14&&cnt/bw<=6.5,bent=bw>=W*.25&&bh<=H*.5&&cnt/(bw+bh)<=7;
      if(bw<W*.2||!(straight||bent))continue;
      const e=Math.max(6,Math.round(bw*.08));let sa=0,na=0,sb=0,nb=0;
      for(const q of pts){const qx=q%W,qy=(q-qx)/W;if(qx<x0+e){sa+=qy;na++}if(qx>x1-e){sb+=qy;nb++}}
      const col=new Map();for(const q of pts){const qx=q%W,qy=(q-qx)/W,k=Math.round(qx/6);(col.get(k)||col.set(k,[]).get(k)).push(qy)}
      const prof=[...col.entries()].sort((a,b)=>a[0]-b[0]).map(([k,ys])=>{ys.sort((a,b)=>a-b);return [k*6,ys[ys.length>>1]]});
      out.push({x0,x1,ya:na?sa/na:y0,yb:nb?sb/nb:y1,w:bw,prof,bent:!straight});
    }
    return out;
  }
  window.PPOCR=Object.assign(window.PPOCR||{},{separators});

  // ---------- 바를 정(正) 막대 세기 ----------
  // 이름 줄(예: Human, Safety) 오른쪽의 손글씨 덩어리를 찾아, 글자마다 선의 길이로 막대 수(一=1, 丅=2, 下=3, 止=4, 正=5)를 어림해요.
  function thin(m,w,h){   // Zhang-Suen 세선화 (작은 조각용)
    const g=new Uint8Array(m);let ch=true,guard=0;
    const at=(x,y)=>(x<0||y<0||x>=w||y>=h)?0:g[y*w+x];
    while(ch&&guard++<60){ch=false;
      for(let pass=0;pass<2;pass++){const del=[];
        for(let y=1;y<h-1;y++)for(let x=1;x<w-1;x++){
          if(!g[y*w+x])continue;
          const p2=at(x,y-1),p3=at(x+1,y-1),p4=at(x+1,y),p5=at(x+1,y+1),p6=at(x,y+1),p7=at(x-1,y+1),p8=at(x-1,y),p9=at(x-1,y-1);
          const B=p2+p3+p4+p5+p6+p7+p8+p9;if(B<2||B>6)continue;
          const seq=[p2,p3,p4,p5,p6,p7,p8,p9,p2];let A=0;for(let i=0;i<8;i++)if(!seq[i]&&seq[i+1])A++;if(A!==1)continue;
          if(pass===0){if(p2*p4*p6!==0||p4*p6*p8!==0)continue}else{if(p2*p4*p8!==0||p2*p6*p8!==0)continue}
          del.push(y*w+x)}
        if(del.length){ch=true;for(const k of del)g[k]=0}}}
    return g;
  }
  async function tally(canvas,notes,printed){
    await load();
    const W=canvas.width,H=canvas.height,M=Math.max(W,H),n=W*H;
    const d=canvas.getContext("2d",{willReadFrequently:true}).getImageData(0,0,W,H).data,g=new Uint8Array(n);
    for(let i=0,p=0;i<n;i++,p+=4)g[i]=(d[p]*.3+d[p+1]*.59+d[p+2]*.11)|0;
    const I=new Float64Array((W+1)*(H+1));
    for(let y=0;y<H;y++){let row=0;for(let x=0;x<W;x++){row+=g[y*W+x];I[(y+1)*(W+1)+x+1]=I[y*(W+1)+x+1]+row}}
    const hw=Math.max(12,Math.round(M*.012));
    const inkAt=(x,y)=>{const y0=Math.max(0,y-hw),y1=Math.min(H,y+hw+1),x0=Math.max(0,x-hw),x1=Math.min(W,x+hw+1),a=(y1-y0)*(x1-x0),s2=I[y1*(W+1)+x1]-I[y0*(W+1)+x1]-I[y1*(W+1)+x0]+I[y0*(W+1)+x0];return g[y*W+x]<s2/a*.62&&g[y*W+x]<125};
    const pr=(printed||[]).filter(l=>l.conf>=.6&&l.text.replace(/\s/g,"").length>=3);
    const out=[];
    for(const nt of notes){
      const bh=Math.max(nt.h,M*.015),bx0=Math.round(nt.x+nt.w/2),bx1=Math.min(W,nt.xmax||W,Math.round(bx0+Math.max(M*.16,bh*7))),by0=Math.max(0,Math.round(nt.y-bh*(nt.xmax?.75:1.3))),by1=Math.min(H,Math.round(nt.y+bh*(nt.xmax?.75:1.3)));
      if(bx1-bx0<10)continue;
      const w=bx1-bx0,h=by1-by0,ink=new Uint8Array(w*h);
      for(let y=0;y<h;y++)for(let x=0;x<w;x++)if(inkAt(bx0+x,by0+y))ink[y*w+x]=1;
      const lab=new Uint8Array(w*h),blobs=[];
      for(let i=0;i<w*h;i++){
        if(!ink[i]||lab[i])continue;const st=[i];lab[i]=1;let cnt=0,x0=w,y0=h,x1=0,y1=0;const pts=[];
        while(st.length){const q=st.pop(),qx=q%w,qy=(q-qx)/w;cnt++;pts.push(q);if(qx<x0)x0=qx;if(qx>x1)x1=qx;if(qy<y0)y0=qy;if(qy>y1)y1=qy;
          for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){const xx=qx+dx,yy=qy+dy;if(xx<0||yy<0||xx>=w||yy>=h)continue;const k=yy*w+xx;if(ink[k]&&!lab[k]){lab[k]=1;st.push(k)}}}
        const bw=x1-x0+1,bhh=y1-y0+1;
        if(cnt<bh*.2||Math.max(bw,bhh)<bh*.3)continue;                                   // 점·잡티
        const cx=bx0+(x0+x1)/2,cy=by0+(y0+y1)/2;
        if(pr.some(l=>cx>l.x-l.w/2&&cx<l.x+l.w/2&&cy>l.y-l.h/2&&cy<l.y+l.h/2))continue;      // 인쇄 글씨
        blobs.push({x0,y0,x1:x1+1,y1:y1+1,pts,bw,bh:bhh});
      }
      blobs.sort((a,b)=>a.x0-b.x0);
      const groups=[];
      for(const b of blobs){const G=groups[groups.length-1];
        if(G&&b.x0-G.x1<bh*.7){G.items.push(b);G.x1=Math.max(G.x1,b.x1);G.y0=Math.min(G.y0,b.y0);G.y1=Math.max(G.y1,b.y1)}
        else groups.push({items:[b],x0:b.x0,x1:b.x1,y0:b.y0,y1:b.y1})}
      const gl=[];
      for(const G of groups){
        const gw=G.x1-G.x0,gh=G.y1-G.y0;if(gw<2||gh<2)continue;
        const m=new Uint8Array(gw*gh);for(const b of G.items)for(const q of b.pts){const qx=q%w-G.x0,qy=((q/w)|0)-G.y0;if(qx>=0&&qy>=0&&qx<gw&&qy<gh)m[qy*gw+qx]=1}
        const sk=thin(m,gw,gh);let L=0,ends=0,br=0;for(let i=0;i<sk.length;i++)L+=sk[i];
        for(let y=0;y<gh;y++)for(let x=0;x<gw;x++){if(!sk[y*gw+x])continue;let nb=0;for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){if(!dx&&!dy)continue;const xx=x+dx,yy=y+dy;if(xx>=0&&yy>=0&&xx<gw&&yy<gh&&sk[yy*gw+xx])nb++}if(nb===1)ends++;else if(nb>=3)br++}
        gl.push({w:gw,h:gh,len:L,ends,br,ratio:L/Math.max(bh,gh,gw*.5),x:bx0+(G.x0+G.x1)/2,y:by0+(G.y0+G.y1)/2,x0:bx0+G.x0,x1:bx0+G.x1});
      }
      out.push({name:nt.text,x:nt.x,y:nt.y,glyphs:gl,bh});
    }
    return out;
  }
  window.PPOCR=Object.assign(window.PPOCR||{},{tally});
  // ---------- 정(正) 막대 획 수 세기 ----------
  // 막대는 획 수가 곧 개수예요(一=1, 丅=2, 下=3, 正=5). 잉크 덩어리마다 뼈대(세선화)의 끝점 수 - 1 = 획 수로 셉니다.
  // req: {x0,x1,y,bh,split} (canvas 좌표). split보다 왼쪽에 중심이 있는 덩어리는 이름 글씨라 빼요.
  async function strokes(canvas,req,printed){
    const W=canvas.width,H=canvas.height,M=Math.max(W,H),bh=req.bh;
    const x0=Math.max(0,Math.round(req.x0)),x1=Math.min(W,Math.round(req.x1)),y0=Math.max(0,Math.round(req.y-bh*1.3)),y1=Math.min(H,Math.round(req.y+bh*1.4));
    const w=x1-x0,h=y1-y0;if(w<4||h<4)return {n:0,comps:[]};
    const pad=Math.round(M*.015),X0=Math.max(0,x0-pad),Y0=Math.max(0,y0-pad),X1=Math.min(W,x1+pad),Y1=Math.min(H,y1+pad),WW=X1-X0,HH=Y1-Y0;
    const d=canvas.getContext("2d",{willReadFrequently:true}).getImageData(X0,Y0,WW,HH).data,g=new Float32Array(WW*HH);
    const red=new Uint8Array(WW*HH);for(let i=0,p=0;i<WW*HH;i++,p+=4){g[i]=d[p]*.3+d[p+1]*.59+d[p+2]*.11;if(d[p]>d[p+1]+45&&d[p]>d[p+2]+35)red[i]=1}   // 빨간 펜(정정 표시)은 빼요
    const I=new Float64Array((WW+1)*(HH+1));for(let y=0;y<HH;y++){let r=0;for(let x=0;x<WW;x++){r+=g[y*WW+x];I[(y+1)*(WW+1)+x+1]=I[y*(WW+1)+x+1]+r}}
    const hw=Math.max(8,Math.round(bh*.35));
    const ink0=new Uint8Array(w*h);
    for(let y=0;y<h;y++)for(let x=0;x<w;x++){const gx=x+x0-X0,gy=y+y0-Y0,a0=Math.max(0,gx-hw),a1=Math.min(WW,gx+hw+1),b0=Math.max(0,gy-hw),b1=Math.min(HH,gy+hw+1);
      const mean=(I[b1*(WW+1)+a1]-I[b0*(WW+1)+a1]-I[b1*(WW+1)+a0]+I[b0*(WW+1)+a0])/((a1-a0)*(b1-b0));const v=g[gy*WW+gx];
      if(v<mean*.7&&v<150&&!red[gy*WW+gx])ink0[y*w+x]=1}
    const pr=(printed||[]).filter(l=>l.conf>=.6&&String(l.text).replace(/\s/g,"").length>=3);
    for(const l of pr){const a0=Math.max(0,Math.round(l.x-l.w/2-x0)),a1=Math.min(w,Math.round(l.x+l.w/2-x0)),b0=Math.max(0,Math.round(l.y-l.h/2-y0)),b1=Math.min(h,Math.round(l.y+l.h/2-y0));for(let y=b0;y<b1;y++)for(let x=a0;x<a1;x++)ink0[y*w+x]=0}
    const ink=new Uint8Array(w*h);   // 2px 넓혀서 끊긴 획을 이어요
    for(let y=0;y<h;y++)for(let x=0;x<w;x++)if(ink0[y*w+x])for(let dy=-2;dy<=2;dy++)for(let dx=-2;dx<=2;dx++){const xx=x+dx,yy=y+dy;if(xx>=0&&yy>=0&&xx<w&&yy<h)ink[yy*w+xx]=1}
    const lab=new Int32Array(w*h),comps=[];let nl=0;
    for(let i=0;i<w*h;i++){if(!ink[i]||lab[i])continue;nl++;const st=[i];lab[i]=nl;const pts=[];let cx0=w,cy0=h,cx1=0,cy1=0,sx=0;
      while(st.length){const q=st.pop(),qx=q%w,qy=(q-qx)/w;pts.push(q);sx+=qx;if(qx<cx0)cx0=qx;if(qx>cx1)cx1=qx;if(qy<cy0)cy0=qy;if(qy>cy1)cy1=qy;
        for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){const xx=qx+dx,yy=qy+dy;if(xx<0||yy<0||xx>=w||yy>=h)continue;const k=yy*w+xx;if(ink[k]&&!lab[k]){lab[k]=nl;st.push(k)}}}
      const bw=cx1-cx0+1,bhh=cy1-cy0+1;
      if(Math.max(bw,bhh)<bh*.22)continue;                                   // 점·잡티
      if(cx0===0&&x0>0&&sx/pts.length<w*.1)continue;                          // 왼쪽 가장자리에 걸린 이름 글씨 조각
      comps.push({pts,cx0,cy0,cx1,cy1,bw,bh:bhh,mx:x0+sx/pts.length})}
    let n=0;const out=[];
    for(const c of comps){
      if(c.mx<req.split)continue;
      const my=y0+(c.cy0+c.cy1)/2;if(Math.abs(my-req.y)>bh*.95)continue;   // 다른 줄 글씨
      if(c.cy0===0||c.cy1===h-1){if(c.bh>bh*1.2)continue}
      if(c.bw<bh*.35&&c.bh<bh*.8&&my>req.y+bh*.15)continue;                // 쉼표·마침표
      const gw=c.bw+2,gh=c.bh+2,m=new Uint8Array(gw*gh);for(const q of c.pts){const qx=q%w-c.cx0+1,qy=((q/w)|0)-c.cy0+1;m[qy*gw+qx]=1}
      const sk=thin(m,gw,gh),thr=Math.max(3,Math.round(bh*.12));
      const nb=(k)=>{const x=k%gw,y=(k-x)/gw;const r=[];for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){if(!dx&&!dy)continue;const xx=x+dx,yy=y+dy;if(xx>=0&&yy>=0&&xx<gw&&yy<gh&&sk[yy*gw+xx])r.push(yy*gw+xx)}return r};
      for(let round=0;round<2;round++){   // 짧은 잔가지 제거
        const ends=[];for(let k=0;k<sk.length;k++)if(sk[k]&&nb(k).length===1)ends.push(k);
        for(const e of ends){if(!sk[e])continue;const path=[e];let prev=-1,cur=e,ok=false;
          while(path.length<=thr){const ns=nb(cur).filter(k=>k!==prev&&!path.includes(k));if(ns.length!==1){ok=ns.length>=2;break}prev=cur;cur=ns[0];if(nb(cur).length>=3){ok=true;break}path.push(cur)}
          if(ok&&path.length<thr)for(const k of path)sk[k]=0}}
      let ends=0,L=0;const bp=[];for(let k=0;k<sk.length;k++)if(sk[k]){L++;const q=nb(k).length;if(q===1)ends++;else if(q>=3)bp.push(k)}
      if(L<bh*.2)continue;
      // 교차점(획이 만나는 곳) 수: 가까운 갈래 점은 하나로 묶어요. 막대는 획 수 = 교차점 + 1 (一 0, 丅 1, 下 2, 正 4)
      const used=new Set();let J=0;const R=Math.max(3,Math.round(bh*.12));
      for(const k of bp){if(used.has(k))continue;J++;const st=[k];used.add(k);while(st.length){const q=st.pop(),qx=q%gw,qy=(q-qx)/gw;for(const o of bp){if(used.has(o))continue;const ox=o%gw,oy=(o-ox)/gw;if(Math.abs(ox-qx)<=R&&Math.abs(oy-qy)<=R){used.add(o);st.push(o)}}}}
      const dash=c.bw>c.bh*2.5&&c.bh<bh*.5,s=dash?1:J>=4?5:Math.max(1,Math.min(5,ends-1));   // 굵은 선은 뼈대에 가짜 갈래가 생겨서, 교차점이 4개 이상(正)일 때만 교차점으로 판단해요
      n+=s;out.push({x:c.mx,w:c.bw,h:c.bh,len:L,ends,J,s});
    }
    return {n,comps:out};
  }
  window.PPOCR=Object.assign(window.PPOCR||{},{strokes});
})();
