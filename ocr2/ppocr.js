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
  window.PPOCR={load,recognize};
})();
