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
      let sp=0,cnt=0,sum=0,x0=w,y0=h,x1=0,y1=0;stack[sp++]=i;seen[i]=1;
      while(sp){const q=stack[--sp],qx=q%w,qy=(q-qx)/w;cnt++;sum+=P[q];
        if(qx<x0)x0=qx;if(qx>x1)x1=qx;if(qy<y0)y0=qy;if(qy>y1)y1=qy;
        if(qx>0&&!seen[q-1]&&P[q-1]>TH){seen[q-1]=1;stack[sp++]=q-1}
        if(qx<w-1&&!seen[q+1]&&P[q+1]>TH){seen[q+1]=1;stack[sp++]=q+1}
        if(qy>0&&!seen[q-w]&&P[q-w]>TH){seen[q-w]=1;stack[sp++]=q-w}
        if(qy<h-1&&!seen[q+w]&&P[q+w]>TH){seen[q+w]=1;stack[sp++]=q+w}}
      const bw=x1-x0+1,bh=y1-y0+1;
      if(cnt<12||Math.min(bw,bh)<3||sum/cnt<.55)continue;
      const dist=(bw*bh)*1.6/(2*(bw+bh)),k=1/s;                    // DB 후처리의 unclip
      const px=Math.min(bh*.35,dist);   // 줄 끝 글자가 잘리지 않도록 가로로 조금 더 넓혀요
      boxes.push({x0:Math.max(0,(x0-dist-px)*k),y0:Math.max(0,(y0-dist)*k),x1:Math.min(W,(x1+dist+px)*k),y1:Math.min(H,(y1+dist)*k)});
    }
    return boxes;
  }

  async function readLine(canvas,b){
    let bw=b.x1-b.x0,bh=b.y1-b.y0;if(bw<4||bh<4)return null;
    const vert=bh>bw*1.6;                                           // 세로로 서 있는 글자는 눕혀서 읽어요
    const c=document.createElement("canvas"),H=48;
    const sw=vert?bh:bw,sh=vert?bw:bh,W=Math.min(1600,Math.max(24,Math.ceil(H*sw/sh)));
    c.width=W;c.height=H;const x=c.getContext("2d",{willReadFrequently:true});x.fillStyle="#fff";x.fillRect(0,0,W,H);
    if(vert){x.translate(0,H);x.rotate(-Math.PI/2);x.drawImage(canvas,b.x0,b.y0,bw,bh,0,0,H,W)}
    else x.drawImage(canvas,b.x0,b.y0,bw,bh,0,0,W,H);
    const inp=toBGR(c,W,H,null,null,true);
    const out=await rec.run({[rec.inputNames[0]]:new ort.Tensor("float32",inp,[1,3,H,W])});
    const o=out[rec.outputNames[0]],T=o.dims[1],C=o.dims[2],D=o.data;
    let last=-1,txt="",cs=0,cn=0;
    for(let t=0;t<T;t++){let bi=0,bv=-1;const off=t*C;for(let k=0;k<C;k++){const v=D[off+k];if(v>bv){bv=v;bi=k}}
      if(bi!==0&&bi!==last){txt+=dict[bi]||"";cs+=bv;cn++}last=bi}
    return {text:txt,conf:cn?cs/cn:0};
  }

  async function recognize(canvas,opt){
    await load();opt=opt||{};
    const boxes=await detect(canvas,opt.detSide||1920),lines=[];
    boxes.sort((a,b)=>a.y0-b.y0||a.x0-b.x0);
    for(const b of boxes){
      let r=null;try{r=await readLine(canvas,b)}catch(e){continue}
      if(r&&r.text.trim()&&r.conf>=.3)lines.push({text:r.text,conf:r.conf,x:(b.x0+b.x1)/2,y:(b.y0+b.y1)/2,w:b.x1-b.x0,h:b.y1-b.y0});
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
  function classify1(v){
    const m=DG,h=new Float32Array(m.h);
    for(let i=0;i<784;i++){const a=v[i];if(!a)continue;const o=i*m.h;for(let k=0;k<m.h;k++)h[k]+=a*m.w1[o+k]*m.s1}
    for(let k=0;k<m.h;k++){h[k]+=m.b1[k];if(h[k]<0)h[k]=0}
    const z=new Float32Array(10);for(let c=0;c<10;c++){let t=m.b2[c];for(let k=0;k<m.h;k++)t+=h[k]*m.w2[k*10+c]*m.s2;z[c]=t}
    let mx=-1e9;for(const t of z)if(t>mx)mx=t;let sum=0;const p=z.map(t=>{const e=Math.exp(t-mx);sum+=e;return e});
    let bi=0;for(let c=1;c<10;c++)if(p[c]>p[bi])bi=c;return {d:bi,p:p[bi]/sum};
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
    for(const b of cand){const v=prep28(b,W);if(!v)continue;const r=classify1(v);if(r.p>=.8)ch.push({...b,d:r.d,p:r.p})}
    // 가까운 글자끼리 묶어서 한 자리/두 자리 숫자로 (왼쪽→오른쪽)
    ch.sort((a,b)=>a.x0-b.x0);const groups=[];
    for(const b of ch){
      const G=groups.find(G=>Math.abs(G.cy-b.cy)<Math.max(G.h,b.h)*.6&&b.x0-G.x1<Math.max(G.h,b.h)*.9&&b.x0-G.x1>-Math.max(G.h,b.h)*.3&&G.items.length<2&&Math.abs(G.h-b.h)<Math.max(G.h,b.h)*.45);
      if(G){G.items.push(b);G.x1=Math.max(G.x1,b.x1);G.y0=Math.min(G.y0,b.y0);G.y1=Math.max(G.y1,b.y1);G.h=G.y1-G.y0;G.cy=(G.y0+G.y1)/2}
      else groups.push({items:[b],x0:b.x0,x1:b.x1,y0:b.y0,y1:b.y1,h:b.h,cy:b.cy});
    }
    return groups.map(G=>{const t=G.items.map(i=>i.d).join(""),p=Math.min(...G.items.map(i=>i.p));
      return {text:t,num:+t,conf:p,x:(G.x0+G.x1)/2,y:(G.y0+G.y1)/2,w:G.x1-G.x0,h:G.h}});
  }
  window.PPOCR={load,recognize,handwriting};

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
      if(bw<W*.2||bh>H*.07||bw/bh<14||cnt/bw>6.5)continue;
      const e=Math.max(6,Math.round(bw*.08));let sa=0,na=0,sb=0,nb=0;
      for(const q of pts){const qx=q%W,qy=(q-qx)/W;if(qx<x0+e){sa+=qy;na++}if(qx>x1-e){sb+=qy;nb++}}
      out.push({x0,x1,ya:na?sa/na:y0,yb:nb?sb/nb:y1,w:bw});
    }
    return out;
  }
  window.PPOCR=Object.assign(window.PPOCR||{},{separators});
})();
