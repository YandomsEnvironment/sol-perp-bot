"use strict";
const $=id=>document.getElementById(id);
const S={h:"15m",tf:"1m",wallet:"",busy:false,range:null,data:null};
const C={up:"#167d62",down:"#c8494e",grid:"#f0f0f3",axis:"#76767c",bg:"#ffffff"};
const appearanceDefaults={theme:"system",style:"candles",height:"standard",grid:true,priceLine:true,signals:true};
let appearance={...appearanceDefaults};
try {
  const stored=JSON.parse(localStorage.getItem("marketBrief.appearance")||"{}");
  for(const [key,allowed] of Object.entries({theme:["system","light","dark"],style:["candles","line"],height:["compact","standard","tall"],grid:[true,false],priceLine:[true,false],signals:[true,false]})) {
    if(allowed.includes(stored?.[key]))appearance[key]=stored[key];
  }
} catch (_) {}
const systemAppearance=matchMedia("(prefers-color-scheme: dark)");
function applyAppearance() {
  const dark=appearance.theme==="dark"||(appearance.theme==="system"&&systemAppearance.matches);
  document.documentElement.dataset.theme=dark?"dark":"light";
  const css=getComputedStyle(document.documentElement);
  Object.assign(C,{up:css.getPropertyValue("--green").trim(),down:css.getPropertyValue("--red").trim(),bg:css.getPropertyValue("--surface").trim(),axis:css.getPropertyValue("--muted").trim(),grid:css.getPropertyValue("--chart-grid").trim()});
  document.querySelector('meta[name="theme-color"]').content=css.getPropertyValue("--bg").trim();
  const canvas=$("chart");
  if(canvas)canvas.style.height=({compact:"300px",standard:"",tall:"560px"})[appearance.height];
  if($("themeSelect"))$("themeSelect").value=appearance.theme;
  for(const [id,key] of [["chartStyle","style"],["chartHeight","height"],["chartGrid","grid"],["chartPriceLine","priceLine"],["chartSignals","signals"]]) {
    if(!$(id))continue;
    if(typeof appearance[key]==="boolean")$(id).checked=appearance[key];else $(id).value=appearance[key];
  }
  requestAnimationFrame(draw);
}
function saveAppearance() {
  try {localStorage.setItem("marketBrief.appearance",JSON.stringify(appearance));$("appearanceStatus").textContent="Appearance saved on this browser.";}
  catch (_) {$("appearanceStatus").textContent="Applied for this visit; browser storage is unavailable.";}
  applyAppearance();
}
if(document.documentElement)applyAppearance();
systemAppearance.addEventListener("change",()=>{if(appearance.theme==="system")applyAppearance();});
const n=v=>v==null||!Number.isFinite(Number(v))?null:Number(v);
const esc=v=>String(v??"").replace(/[&<>"\']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","\'":"&#39;"}[c]));
function valueTone(value) {const v=n(value);return v===null||v===0?"":v>0?"up":"down";}
function directionTone(value) {return {LONG:"up",SHORT:"down"}[String(value).toUpperCase()]||"";}
function money(v,d=2){v=n(v);return v==null?"\u2014":v.toLocaleString("en-US",{minimumFractionDigits:d,maximumFractionDigits:d});}
function pct(v,d=2){v=n(v);return v==null?"\u2014":`${v>0?"+":""}${(v*100).toFixed(d)}%`;}
function time(ms){return new Date(ms).toLocaleTimeString("en-GB",{hour:"2-digit",minute:"2-digit",second:"2-digit",timeZone:"UTC"});}
function tab(which){const a=which==="overview";for(const [id,on] of [["overviewPanel",a],["detailsPanel",!a]])$(id).hidden=!on;for(const [id,on] of [["overviewTab",a],["detailsTab",!a]]){$(id).classList.toggle("active",on);$(id).setAttribute("aria-selected",String(on));$(id).tabIndex=on?0:-1;}location.hash=a?"overview":"research";if(a)requestAnimationFrame(draw);else refreshDetails();}
async function update(){if(S.busy){S.pendingRefresh=true;return;}S.pendingRefresh=false;S.busy=true;try{const q=new URLSearchParams({hz:S.h,tf:S.tf});const r=await fetch(`/api/snapshot?${q}`,{cache:"no-store"}),d=await r.json();if(!r.ok)throw Error(d.error||r.statusText);if(d.tf!==S.tf){S.pendingRefresh=true;return;}S.data=d;render(d);$("updatedLabel").textContent=`Updated ${time(d.updated_at)} UTC`;$("feedStatus").textContent=d.feed_connected?"Live market":"Reconnecting";$("feedDot").classList.toggle("offline",!d.feed_connected);$("chartRangeLabel").textContent=`${d.candles.length.toLocaleString()} candles / ${d.tf}`;$("chartTimeframe").textContent=d.tf;$("refreshChip").textContent=`Refreshes every ${d.refresh_seconds}s`;if(!S.range)S.range=[Math.max(0,d.candles.length-({"1m":360,"5m":288,"15m":192,"1h":168,"4h":126}[S.tf]||180)),d.candles.length-1];else if(S.range[1]>=d.candles.length-2)S.range[1]=d.candles.length-1;draw();}catch(e){$("feedStatus").textContent="Data unavailable";$("feedDot").classList.add("offline");$("updatedLabel").textContent=`Update unavailable: ${String(e.message).slice(0,120)}`;}finally{S.busy=false;if(S.pendingRefresh)update();}}
function render(d){$("priceValue").textContent=`$${money(d.price,d.price<1?5:2)}`;$("priceChange").textContent=`${pct(d.price_change_15m)} (15m)`;$("priceChange").className=`price-change ${valueTone(d.price_change_15m)}`;$("fundingValue").textContent=d.funding==null?"?":`${(Number(d.funding)*100).toFixed(4)}%`;$("oiValue").textContent=d.open_interest_usd==null?"?":`$${(d.open_interest_usd/1e6).toFixed(1)}M`;
$("atrValue").textContent=pct(d.atr_pct,3);$("latencyText").textContent=d.tick_age_ms==null?"Tick unavailable":`Latency: tick ${Math.round(d.tick_age_ms)}ms`;const f=d.flow||{};$("flowText").textContent=f.buy_share==null?"Taker flow 15m: ?":`Taker flow 15m: ${(f.buy_share*100).toFixed(0)}% aggressive buys`;$("flowSizes").textContent=f.buy_share==null?"":`(${Math.round(f.buy_size)} vs ${Math.round(f.sell_size)} SOL)`;$("clockText").textContent=`${time(d.updated_at)} UTC`;
$("decisionHorizon").textContent=`${d.horizon_min} min`;$("verdictValue").textContent=d.decision==="STAND ASIDE"?"Stand aside":d.decision;$("modelLean").textContent=d.model_available===false?"Unavailable":`${d.model_lean} / ${(d.model_probability*100).toFixed(0)}%`;$("modelLean").className=d.model_available===false?"":directionTone(d.model_lean);$("evidenceLean").className=directionTone(d.evidence_lean);const t=d.evidence_tally;$("evidenceLean").textContent=`${d.evidence_lean} / ${t.LONG}L ${t.SHORT}S ${t.NEUTRAL}N`;$("decisionReason").textContent=d.data_quality_errors?.length?d.data_quality_errors.join(". "):d.gate_reason;$("modelState").textContent=d.gate_pass?"Validation passed":"Validation not passed";$("validationNote").textContent=`${d.qualifying_trades} qualifying walk-forward trades. Raw model probabilities are not proven win probabilities. No order execution.`;
$("detailVoteSummary").className=`vote-summary ${directionTone(d.evidence_lean)}`;$("detailVoteSummary").textContent=`${d.evidence_lean} · ${t.LONG}L / ${t.SHORT}S / ${t.NEUTRAL}N`;
$("evidenceRows").innerHTML=(d.evidence||[]).map(x=>`<tr class="indicator-row ${directionTone(x.vote)}"><td>${esc(x.signal)}</td><td class="mono">${esc(x.reading)}</td><td><span class="vote-pill ${String(x.vote).toLowerCase()}">${esc(x.vote)}</span></td><td>${esc(x.why)}</td></tr>`).join("")||'<tr><td colspan="4" class="empty-cell">No evidence available.</td></tr>';
const p=d.probabilities||{};const valid=[p.long,p.flat,p.short].every(v=>d.model_available!==false&&typeof v==="number"&&Number.isFinite(v)&&v>=0&&v<=1)&&Math.abs(p.long+p.flat+p.short-1)<0.01;const bars=[["Long",p.long,C.up],["Flat",p.flat,"#7d8992"],["Short",p.short,C.down]].map(([k,v,col])=>`<div class="prob-row"><span class="prob-label">${k}</span><div class="prob-track" role="meter" aria-label="Raw ${k.toLowerCase()} probability" aria-valuemin="0" aria-valuemax="100" ${valid?`aria-valuenow="${(v*100).toFixed(1)}"`:'aria-valuetext="Unavailable"'}><div class="prob-fill" style="width:${valid?v*100:0}%;background:${col}"></div></div><span class="prob-value">${valid?(v*100).toFixed(1)+"%":"Unavailable"}</span></div>`).join("");$("probabilityBars").innerHTML=bars;$("overviewProbabilityBars").innerHTML=bars;

$("walletAddress").textContent=d.wallet?`${d.wallet.slice(0,5)}...${d.wallet.slice(-5)}`:"Not configured";
if(!$("detailsPanel").hidden)refreshDetails();}
let detailsLoaded=0;
function refreshDetails(){if(Date.now()-detailsLoaded<30000)return;detailsLoaded=Date.now();loadScoring();}
let accountLoaded=0, accountWallet=null, accountBusy=false;
function accountTotals(trades) {
  const number=value=>value===null||value===undefined||String(value).trim()===""?null:n(value);
  const decreases=trades.filter(x=>String(x.action).toLowerCase()==="decrease");
  const pnls=decreases.map(x=>number(x.pnl)), fees=trades.map(x=>number(x.fee));
  const gross=pnls.every(x=>x!==null)?pnls.reduce((a,b)=>a+b,0):null;
  const totalFees=fees.every(x=>x!==null)?fees.reduce((a,b)=>a+b,0):null;
  return {gross,fees:totalFees,net:gross!==null&&totalFees!==null?gross-totalFees:null,decreases:decreases.length,
    profits:pnls.every(x=>x!==null)?pnls.reduce((a,b)=>a+Math.max(0,b),0):null,
    losses:pnls.every(x=>x!==null)?pnls.reduce((a,b)=>a+Math.min(0,b),0):null};
}
function accountReset(message) {
  $("accountStatus").textContent=message;
  $("accountSummary").innerHTML="";
  $("tradeRows").innerHTML='<tr><td colspan="8" class="empty-cell">No current account data available.</td></tr>';
}
async function loadAccount(address, force=false) {
  if(accountBusy)return;
  if(!force&&accountWallet===address&&Date.now()-accountLoaded<30000)return;
  accountWallet=address;accountLoaded=Date.now();
  $("walletAddress").textContent=address?`${address.slice(0,5)}...${address.slice(-5)}`:"Not configured";
  accountReset(address?"Loading public account activity...":"Add a public wallet in Research & activity to view profits and trades.");
  if(!address)return;
  accountBusy=true;
  try {
    const response=await fetch(`/api/account?address=${encodeURIComponent(address)}`,{cache:"no-store"});
    const data=await response.json();
    if(address!==S.wallet)return;
    if(!response.ok||!Array.isArray(data.trades?.dataList))throw Error(data.error||"Trade history unavailable; totals cannot be calculated.");
    const trades=data.trades.dataList.filter(x=>x&&typeof x==="object").sort((a,b)=>(n(b.createdTime)||0)-(n(a.createdTime)||0));
    const totals=accountTotals(trades), positions=Array.isArray(data.positions?.dataList)?data.positions.dataList.length:null;
    const usd=value=>value===null?"Unavailable":`${value<0?"-":""}$${money(Math.abs(value))}`;
    $("accountStatus").textContent=`${trades.length} returned events / ${totals.decreases} reductions or closes / ${positions===null?"open positions unavailable":positions+" open positions"}. Recent-window totals, not lifetime. Refreshes every 30s.`;
    $("accountSummary").innerHTML=[["Reported profits",totals.profits],["Reported losses",totals.losses],["Reported P&L total",totals.gross],["Reported fees (all events)",totals.fees],["Estimated net after fees",totals.net]].map(([label,value])=>`<div class="summary-chip ${valueTone(label==="Reported fees (all events)"&&value!==null?-value:value)}"><span>${label}</span><strong class="${valueTone(label==="Reported fees (all events)"&&value!==null?-value:value)}">${usd(value)}</strong></div>`).join("");
    $("tradeRows").innerHTML=trades.slice(0,20).map(x=>{
      const result=accountTotals([x]);
      const stamp=n(x.createdTime);
      const date=stamp?new Date(stamp*1000):null;
      return `<tr class="trade-result ${valueTone(result.net)}"><td>${date&&Number.isFinite(date.getTime())?esc(date.toISOString().slice(0,16).replace("T"," ")):"?"}</td><td>${esc(x.positionName)}</td><td class="${directionTone(x.side)}">${esc(x.side)}</td><td>${esc(x.action)}</td><td class="mono">${x.price==null?"?":"$"+money(x.price)}</td><td class="mono ${valueTone(result.fees===null?null:-result.fees)}">${usd(result.fees)}</td><td class="mono ${valueTone(result.gross)}">${String(x.action).toLowerCase()==="decrease"?usd(result.gross):"?"}</td><td class="mono ${valueTone(result.net)}">${usd(result.net)}</td></tr>`;
    }).join("")||'<tr><td colspan="8" class="empty-cell">No recent trades returned for this wallet.</td></tr>';
  } catch(error) {if(address===S.wallet)accountReset(`Account activity unavailable: ${error.message}`);}
  finally {accountBusy=false;if(address!==S.wallet)loadAccount(S.wallet,true);}
}
async function loadScoring(){try{const r=await fetch("/api/details",{cache:"no-store"}),d=await r.json();if(!r.ok)throw Error(d.error);const rows=d.scored||[];$("scoringSummary").textContent=rows.length?`${rows.length} matured signals · directional movement only; not fee-adjusted.`:"No matured signals yet.";$("scoringRows").innerHTML=rows.slice(-15).reverse().map(x=>`<tr><td>${esc(String(x.time||"").slice(0,19))}</td><td class="${directionTone(x.pred)}">${esc(x.pred)}</td><td>${(Number(x.p_long)*100).toFixed(1)}%</td><td>${(Number(x.p_short)*100).toFixed(1)}%</td><td>$${money(x.price)}</td><td>$${money(x.px_after)}</td><td class="${valueTone(x.chg)}">${pct(x.chg,3)}</td></tr>`).join("")||'<tr><td colspan="7" class="empty-cell">No matured signals.</td></tr>';}catch(e){$("scoringSummary").textContent=`Signal scoring unavailable: ${e.message}`;}}
function draw(){const canvas=$("chart");if(!canvas||$("overviewPanel").hidden)return;const box=canvas.getBoundingClientRect(),dpr=Math.min(2,devicePixelRatio||1);canvas.width=Math.floor(box.width*dpr);canvas.height=Math.floor(box.height*dpr);const c=canvas.getContext("2d");c.setTransform(dpr,0,0,dpr,0,0);c.fillStyle=C.bg;c.fillRect(0,0,box.width,box.height);const all=S.data?.candles||[];if(all.length<2){c.fillStyle=C.axis;c.textAlign="center";c.fillText("WAITING FOR CANDLES",box.width/2,box.height/2);return;}let [a,b]=S.range||[Math.max(0,all.length-180),all.length-1];a=Math.max(0,Math.min(a,all.length-2));b=Math.max(a+1,Math.min(b,all.length-1));S.range=[a,b];const v=all.slice(a,b+1),pad={l:12,r:76,t:20,b:28},w=box.width-pad.l-pad.r,h=box.height-pad.t-pad.b,vals=v.flatMap(x=>[+x.high,+x.low]);vals.push(+S.data.price);let lo=Math.min(...vals),hi=Math.max(...vals),span=(hi-lo)||1;lo-=span*.06;hi+=span*.06;span=hi-lo;const y=q=>pad.t+(hi-q)/span*h,x=i=>pad.l+(i-a+.5)/(b-a+1)*w;const step=w/(b-a+1),bw=Math.max(1,Math.min(12,step*.64));c.font="10px -apple-system, Segoe UI, sans-serif";
for(let i=0;i<=5;i++){const yy=pad.t+i*h/5;c.strokeStyle=appearance.grid?C.grid:"transparent";c.beginPath();c.moveTo(pad.l,yy);c.lineTo(pad.l+w,yy);c.stroke();c.fillStyle=C.axis;c.textAlign="right";c.fillText((hi-i*span/5).toFixed(2),box.width-8,yy+3);}const nt=Math.min(6,Math.max(2,Math.floor(w/140)));for(let i=0;i<=nt;i++){const ix=Math.round(a+i*(b-a)/nt),xx=x(ix);c.strokeStyle=appearance.grid?C.grid:"transparent";c.beginPath();c.moveTo(xx,pad.t);c.lineTo(xx,pad.t+h);c.stroke();const dt=new Date(all[ix].timestamp);c.fillStyle=C.axis;c.textAlign="center";c.fillText(`${String(dt.getUTCHours()).padStart(2,"0")}:${String(dt.getUTCMinutes()).padStart(2,"0")}`,xx,box.height-9);}
if(appearance.style==="line"){c.strokeStyle=C.up;c.lineWidth=1.8;c.beginPath();v.forEach((q,j)=>{if(j)c.lineTo(x(a+j),y(+q.close));else c.moveTo(x(a+j),y(+q.close));});c.stroke();c.lineWidth=1;}else v.forEach((q,j)=>{const ix=a+j,xx=x(ix),up=+q.close>=+q.open,col=up?C.up:C.down;c.strokeStyle=col;c.lineWidth=1;c.beginPath();c.moveTo(xx,y(+q.high));c.lineTo(xx,y(+q.low));c.stroke();const top=y(Math.max(+q.open,+q.close)),bottom=y(Math.min(+q.open,+q.close));c.fillStyle=col;c.fillRect(xx-bw/2,top,bw,Math.max(1,bottom-top));});if(appearance.priceLine){const lp=+S.data.price,ly=y(lp);c.setLineDash([2,4]);c.strokeStyle=lp>=+v.at(-1).open?C.up:C.down;c.beginPath();c.moveTo(pad.l,ly);c.lineTo(pad.l+w,ly);c.stroke();c.setLineDash([]);c.fillStyle=lp>=+v.at(-1).open?C.up:C.down;c.fillRect(pad.l+w+2,ly-9,66,18);c.fillStyle="#fff";c.textAlign="center";c.fillText(money(lp),pad.l+w+35,ly+3);
}
const tfms=({"1m":60,"5m":300,"15m":900,"1h":3600,"4h":14400}[S.tf]||60)*1000;for(const m of (appearance.signals?S.data.signals||[]:[])){if(m.time<all[a].timestamp||m.time>(all[b].timestamp+tfms))continue;const ix=Math.max(a,Math.min(b,Math.round(a+(m.time-all[a].timestamp)/(all[b].timestamp-all[a].timestamp||1)*(b-a)))),xx=x(ix),yy=y(+m.price)+(m.pred==="SHORT"?-14:14);c.beginPath();c.arc(xx,yy,8,0,Math.PI*2);c.fillStyle=m.pred==="SHORT"?"#e94f59":"#16b985";c.fill();c.fillStyle="#fff";c.textAlign="center";c.textBaseline="middle";c.fillText(m.pred==="SHORT"?"S":"B",xx,yy);c.textBaseline="alphabetic";}S.plot={pad,w,h};}
function syncChartIntervals() {
  document.querySelectorAll("#chartIntervals button").forEach(button => {
    button.setAttribute("aria-pressed", String(button.dataset.tf === S.tf));
    button.disabled = !!S.tfSaving;
  });
}
function init() {
  applyAppearance();
  $("themeSelect").onchange=()=>{appearance.theme=$("themeSelect").value;saveAppearance();};
  $("chartSettingsToggle").onclick=()=>{
    const open=$("chartSettings").hidden;
    $("chartSettings").hidden=!open;$("chartSettingsToggle").setAttribute("aria-expanded",String(open));
  };
  for(const [id,key] of [["chartStyle","style"],["chartHeight","height"],["chartGrid","grid"],["chartPriceLine","priceLine"],["chartSignals","signals"]]) {
    $(id).onchange=()=>{appearance[key]=$(id).type==="checkbox"?$(id).checked:$(id).value;saveAppearance();};
  }
  $("resetChartAppearance").onclick=()=>{appearance={...appearanceDefaults,theme:appearance.theme};saveAppearance();};

  const tabs = [$("overviewTab"), $("detailsTab")];
  tabs.forEach((el, index) => {
    el.onclick = () => tab(index ? "details" : "overview");
    el.onkeydown = event => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      const next = event.key === "Home" ? 0 : event.key === "End" ? 1 : 1 - index;
      tab(next ? "details" : "overview");
      tabs[next].focus();
    };
  });
  $("viewEvidence").onclick = () => {tab("overview");$("indicatorsCard").scrollIntoView({behavior:matchMedia("(prefers-reduced-motion: reduce)").matches?"instant":"smooth",block:"start"});$("indicatorsCard").focus({preventScroll:true});};
  document.querySelector(".brand").onclick = event => {event.preventDefault(); tab("overview");};
  $("settingsForm").onsubmit = async event => {
    event.preventDefault();
    if (S.busy || S.tfSaving) {$("settingsStatus").textContent="Please wait for the current market refresh to finish."; return;}
    const button = $("applySettings"), status = $("settingsStatus");
    const payload = {hz: $("horizonSelect").value, tf: S.tf, sol_wallet: $("walletInput").value.trim()};
    status.classList.remove("error");
    if (payload.sol_wallet && !/^[1-9A-HJ-NP-Za-km-z]{32,44}$/.test(payload.sol_wallet)) {
      status.classList.add("error"); status.textContent = "Enter a valid public Solana address, or leave it empty."; return;
    }
    S.tfSaving = true; syncChartIntervals(); button.disabled = true; button.textContent = "Saving..."; status.textContent = "";
    try {
      const response = await fetch("/api/settings", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)});
      const saved = await response.json();
      if (!response.ok) throw Error(saved.error || "Could not save preferences");
      S.h=saved.hz; S.tf=saved.tf; S.wallet=saved.sol_wallet; S.range=null; detailsLoaded=0;
      status.textContent="Preferences saved.";
      loadAccount(S.wallet,true);
      await update();
    } catch(error) {status.classList.add("error"); status.textContent=error.message;}
    finally {S.tfSaving=false; syncChartIntervals(); button.disabled=false; button.textContent="Save preferences";}
  };
  document.querySelectorAll("#chartIntervals button").forEach(button => {
    button.onclick = async () => {
      if (S.tfSaving || button.dataset.tf === S.tf) return;
      const status = $("chartIntervalStatus");
      S.tfSaving = true; syncChartIntervals();
      $("applySettings").disabled = true;
      status.textContent = "Saving chart interval...";
      try {
        const response = await fetch("/api/settings", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({tf:button.dataset.tf})});
        const saved = await response.json();
        if (!response.ok) throw Error(saved.error || "Could not save chart interval");
        S.tf = saved.tf; S.range = null; S.drag = null;
        status.textContent = `${S.tf} candles selected.`;
        await update();
      } catch (error) { status.textContent = `Chart interval unchanged: ${error.message}`; }
      finally { S.tfSaving = false; syncChartIntervals(); $("applySettings").disabled = false; }
    };
  });
  const canvas = $("chart");
  canvas.addEventListener("pointerdown", event => {
    if (event.button!==0 || !S.range || !S.plot) return;
    S.drag={x:event.clientX,start:S.range[0],count:S.range[1]-S.range[0]+1};
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", event => {
    if (!S.drag) return;
    const shift=Math.round(-(event.clientX-S.drag.x)/Math.max(1,S.plot.w)*S.drag.count);
    const left=Math.max(0,Math.min(S.data.candles.length-S.drag.count,S.drag.start+shift));
    S.range=[left,left+S.drag.count-1]; draw();
  });
  for (const name of ["pointerup","pointercancel","lostpointercapture"]) canvas.addEventListener(name,()=>S.drag=null);
  canvas.addEventListener("wheel", event => {
    if (!S.range || !S.data || S.data.candles.length<2) return;
    event.preventDefault();
    const count=S.range[1]-S.range[0]+1, total=S.data.candles.length;
    const next=Math.max(Math.min(30,total),Math.min(total,Math.round(count*(event.deltaY<0?.8:1.25))));
    const rect=canvas.getBoundingClientRect(), anchor=Math.max(0,Math.min(1,(event.clientX-rect.left)/rect.width));
    const left=Math.max(0,Math.min(total-next,Math.round(S.range[0]+(count-next)*anchor)));
    S.range=[left,left+next-1]; draw();
  },{passive:false});
  $("resetZoom").onclick=()=>{if(!S.data?.candles.length)return;const end=S.data.candles.length-1;S.range=[Math.max(0,end-179),end];draw();};
  new ResizeObserver(()=>requestAnimationFrame(draw)).observe(canvas);
  fetch("/api/settings",{cache:"no-store"}).then(async response=>{
    if(!response.ok)throw Error("Preferences unavailable");return response.json();
  }).then(settings=>{
    S.h=settings.hz||"15m";S.tf=["1m","5m","15m"].includes(settings.tf)?settings.tf:"1m";S.wallet=settings.sol_wallet||"";
    syncChartIntervals();$("horizonSelect").value=S.h;$("walletInput").value=S.wallet;
  }).catch(error=>{$("settingsStatus").textContent=error.message;}).finally(()=>{
    if(location.hash==="#research")tab("details");
    loadAccount(S.wallet);
    update();setInterval(()=>{if(!document.hidden){update();loadAccount(S.wallet);}},5000);
  });
  draw();
}
document.addEventListener("DOMContentLoaded",init);
