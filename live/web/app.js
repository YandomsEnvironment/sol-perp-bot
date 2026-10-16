"use strict";
const $=id=>document.getElementById(id);
const S={h:"15m",tf:"1m",wallet:"",busy:false,range:null,data:null,followLive:true,hover:null};
const C={up:"#167d62",down:"#c8494e",grid:"#f0f0f3",axis:"#76767c",bg:"#ffffff",ink:"#ffffff"};
const appearanceDefaults={theme:"system",palette:"classic",up:null,down:null,accent:null,gridColor:null,style:"candles",height:"standard",grid:true,priceLine:true,signals:true,candleStyle:"solid",bodyWidth:88,bodyOpacity:100,borders:true,wicks:true,wickWidth:1,matchWicks:true,wickUp:null,wickDown:null,crosshair:true};
const colorControls={upColor:"up",downColor:"down",accentColor:"accent",gridColor:"gridColor"};
const wickControls={wickUpColor:"wickUp",wickDownColor:"wickDown"};
const chartControls={chartStyle:"style",chartHeight:"height",chartGrid:"grid",chartPriceLine:"priceLine",chartSignals:"signals",candleStyle:"candleStyle",bodyWidth:"bodyWidth",bodyOpacity:"bodyOpacity",candleBorders:"borders",candleWicks:"wicks",wickWidth:"wickWidth",matchWicks:"matchWicks",chartCrosshair:"crosshair"};
const numericAppearance={bodyWidth:[40,95,1],bodyOpacity:[30,100,1],wickWidth:[0.5,3,0.5]};
const colorPalettes={
  emerald:{light:{up:"#00865a",down:"#ce3247",accent:"#00865a"},dark:{up:"#00d68f",down:"#ff526b",accent:"#00d68f"}},
  mint:{light:{up:"#237952",down:"#b54758",accent:"#237952"},dark:{up:"#9cdbb5",down:"#f1a0ae",accent:"#9cdbb5"}},
  terminal:{light:{up:"#347d13",down:"#cb2b2b",accent:"#347d13"},dark:{up:"#75ef48",down:"#ff5555",accent:"#75ef48"}},
  forest:{light:{up:"#386c4a",down:"#a64435",accent:"#386c4a"},dark:{up:"#8bb69a",down:"#dd9483",accent:"#8bb69a"}},
  crimson:{light:{up:"#00745e",down:"#b31b3c",accent:"#b31b3c"},dark:{up:"#43d9b0",down:"#ff4267",accent:"#ff8099"}},
  ocean:{light:{up:"#1769b5",down:"#b8540b",accent:"#1769b5"},dark:{up:"#72baff",down:"#ffb15c",accent:"#72baff"}},
  violet:{light:{up:"#007e87",down:"#b12e83",accent:"#7851bd"},dark:{up:"#56dce5",down:"#f28acb",accent:"#b7a0fa"}}
};
function validColor(value) {return typeof value==="string" && /^#[0-9a-f]{6}$/i.test(value);}
function normalizeAppearance(stored) {
  const result={...appearanceDefaults};
  for(const [key,allowed] of Object.entries({theme:["system","light","dark"],palette:["classic",...Object.keys(colorPalettes),"custom"],style:["candles","line"],height:["compact","standard","tall"],candleStyle:["solid","hollow-up","hollow"],grid:[true,false],priceLine:[true,false],signals:[true,false],borders:[true,false],wicks:[true,false],matchWicks:[true,false],crosshair:[true,false]})) {
    if(allowed.includes(stored?.[key]))result[key]=stored[key];
  }
  for(const key of [...Object.values(colorControls),...Object.values(wickControls)])if(validColor(stored?.[key]))result[key]=stored[key].toLowerCase();
  for(const [key,[min,max,step]] of Object.entries(numericAppearance)) {
    const value=stored?.[key];
    if(typeof value==="number"&&Number.isFinite(value))result[key]=Math.round(Math.max(min,Math.min(max,value))/step)*step;
  }
  return result;
}
let appearance={...appearanceDefaults};
try {appearance=normalizeAppearance(JSON.parse(localStorage.getItem("marketBrief.appearance")||"{}"));} catch (_) {}
const systemAppearance=matchMedia("(prefers-color-scheme: dark)");
function applyAppearance() {
  const dark=appearance.theme==="dark"||(appearance.theme==="system"&&systemAppearance.matches);
  document.documentElement.dataset.theme=dark?"dark":"light";
  const root=document.documentElement;
  // Read theme defaults before applying a preset or custom overrides.
  for(const property of ["--green","--red","--accent","--focus","--chart-grid","--gain-soft","--loss-soft"])root.style.removeProperty(property);
  const defaults=getComputedStyle(root),preset=colorPalettes[appearance.palette]?.[dark?"dark":"light"]||{};
  for(const [key,property] of Object.entries({up:"--green",down:"--red",accent:"--accent",gridColor:"--chart-grid"})) {
    const color=appearance.palette==="custom"&&validColor(appearance[key])?appearance[key]:preset[key];
    if(color)root.style.setProperty(property,color);
  }
  const accent=defaults.getPropertyValue("--accent").trim();
  root.style.setProperty("--focus",accent);
  const css=getComputedStyle(root);
  Object.assign(C,{up:css.getPropertyValue("--green").trim(),down:css.getPropertyValue("--red").trim(),bg:css.getPropertyValue("--surface").trim(),axis:css.getPropertyValue("--muted").trim(),grid:css.getPropertyValue("--chart-grid").trim(),ink:css.getPropertyValue("--chart-ink").trim()||"#ffffff"});
  document.querySelector('meta[name="theme-color"]').content=css.getPropertyValue("--bg").trim();
  const canvas=$("chart");
  if(canvas)canvas.style.height=({compact:"300px",standard:"",tall:"560px"})[appearance.height];
  if($("themeSelect"))$("themeSelect").value=appearance.theme;
  if($("paletteSelect"))$("paletteSelect").value=appearance.palette;
  const resolved={up:C.up,down:C.down,accent,gridColor:C.grid};
  for(const [id,key] of Object.entries(colorControls))if($(id)) {
    $(id).value=resolved[key];$(`${id}Value`).textContent=resolved[key].toUpperCase();
  }
  root.style.setProperty("--gain-soft",`color-mix(in srgb, ${C.up} 12%, ${C.bg})`);
  root.style.setProperty("--loss-soft",`color-mix(in srgb, ${C.down} 12%, ${C.bg})`);
  for(const [id,key] of Object.entries(chartControls)) {
    if(!$(id))continue;
    if(typeof appearance[key]==="boolean")$(id).checked=appearance[key];else $(id).value=appearance[key];
    if(numericAppearance[key])$(`${id}Value`).textContent=`${appearance[key]}${key==="wickWidth"?"px":"%"}`;
  }
  const candlesDisabled=appearance.style!=="candles";
  for(const id of ["candleStyle","bodyWidth","bodyOpacity","candleBorders","candleWicks"])if($(id))$(id).disabled=candlesDisabled;
  if($("wickWidth"))$("wickWidth").disabled=candlesDisabled||!appearance.wicks;
  if($("matchWicks"))$("matchWicks").disabled=candlesDisabled||!appearance.wicks;
  for(const [id,key] of Object.entries(wickControls))if($(id)) {
    const color=appearance.matchWicks?(key==="wickUp"?C.up:C.down):(appearance[key]||(key==="wickUp"?C.up:C.down));
    $(id).value=color;$(`${id}Value`).textContent=color.toUpperCase();
    $(id).disabled=candlesDisabled||!appearance.wicks||appearance.matchWicks;
  }
  if($("candlePreviewCaption"))$("candlePreviewCaption").textContent=candlesDisabled?"Illustrative line preview · not market data":"Illustrative candles · not market data";
  document.querySelectorAll(".probability-bars").forEach(bars=>{
    bars.querySelectorAll(".prob-fill").forEach((fill,index)=>fill.style.background=[C.up,C.axis,C.down][index]);
  });
  requestAnimationFrame(()=>{draw();draw24h();drawCandlePreview();});
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
async function update(){if(S.busy){S.pendingRefresh=true;return;}S.pendingRefresh=false;S.busy=true;try{const q=new URLSearchParams({hz:S.h,tf:S.tf});const r=await fetch(`/api/snapshot?${q}`,{cache:"no-store"}),d=await r.json();if(!r.ok)throw Error(d.error||r.statusText);if(d.tf!==S.tf || q.get("hz")!==S.h){S.pendingRefresh=true;return;}S.data=d;draw24h();render(d);$("updatedLabel").textContent=`Updated ${time(d.updated_at)} UTC`;$("feedStatus").textContent=d.feed_connected?"Live market":"Reconnecting";$("feedDot").classList.toggle("offline",!d.feed_connected);$("chartRangeLabel").textContent=`${d.candles.length.toLocaleString()} candles / ${d.tf}`;$("chartTimeframe").textContent=d.tf;$("refreshChip").textContent=`Refreshes every ${d.refresh_seconds}s`;if(!S.range)S.range=[Math.max(0,d.candles.length-({"1m":360,"5m":288,"15m":192,"1h":168,"4h":126}[S.tf]||180)),d.candles.length-1];else if(S.followLive){const count=S.range[1]-S.range[0]+1;S.range=[Math.max(0,d.candles.length-count),d.candles.length-1];}draw();}catch(e){$("feedStatus").textContent="Data unavailable";$("feedDot").classList.add("offline");$("updatedLabel").textContent=`Update unavailable: ${String(e.message).slice(0,120)}`;}finally{S.busy=false;if(S.pendingRefresh)update();}}
function render(d){$("priceValue").textContent=`$${money(d.price,d.price<1?5:2)}`;$("priceChange").textContent=`${pct(d.price_change_15m)} (15m)`;$("priceChange").className=`price-change ${valueTone(d.price_change_15m)}`;$("fundingValue").textContent=d.funding==null?"?":`${(Number(d.funding)*100).toFixed(4)}%`;$("oiValue").textContent=d.open_interest_usd==null?"?":`$${(d.open_interest_usd/1e6).toFixed(1)}M`;
$("atrValue").textContent=pct(d.atr_pct,3);$("latencyText").textContent=d.tick_age_ms==null?"Tick unavailable":`Latency: tick ${Math.round(d.tick_age_ms)}ms`;const f=d.flow||{};$("flowText").textContent=f.buy_share==null?"Taker flow 15m: ?":`Taker flow 15m: ${(f.buy_share*100).toFixed(0)}% aggressive buys`;$("flowSizes").textContent=f.buy_share==null?"":`(${Math.round(f.buy_size)} vs ${Math.round(f.sell_size)} SOL)`;$("clockText").textContent=`${time(d.updated_at)} UTC`;
$("decisionHorizon").textContent=d.horizon_min>=1440?`${Math.round(d.horizon_min/1440)} day`:d.horizon_min>=60?`${Math.round(d.horizon_min/60)} hr`:`${d.horizon_min} min`;$("verdictValue").textContent=d.decision==="STAND ASIDE"?"Stand aside":d.decision;$("modelLean").textContent=d.model_available===false?"Unavailable":`${d.model_lean} / ${(d.model_probability*100).toFixed(0)}%`;$("modelLean").className=d.model_available===false?"":directionTone(d.model_lean);$("evidenceLean").className=directionTone(d.evidence_lean);const t=d.evidence_tally;$("evidenceLean").textContent=`${d.evidence_lean} / ${t.LONG}L ${t.SHORT}S ${t.NEUTRAL}N`;$("decisionReason").textContent=d.data_quality_errors?.length?d.data_quality_errors.join(". "):d.gate_reason;$("modelState").textContent=d.gate_pass?"Validation passed":"Validation not passed";$("validationNote").textContent=`${d.qualifying_trades} qualifying walk-forward trades. Raw model probabilities are not proven win probabilities. No order execution.${d.model_id?` Research model: ${d.model_id}. Trained: ${d.model_trained_at||"unknown"}. Target: ${d.model_target||"unknown"}.`:""}`;
$("detailVoteSummary").className=`vote-summary ${directionTone(d.evidence_lean)}`;$("detailVoteSummary").textContent=`${d.evidence_lean} · ${t.LONG}L / ${t.SHORT}S / ${t.NEUTRAL}N`;
$("evidenceRows").innerHTML=(d.evidence||[]).map(x=>`<tr class="indicator-row ${directionTone(x.vote)}"><td>${esc(x.signal)}</td><td class="mono">${esc(x.reading)}</td><td><span class="vote-pill ${String(x.vote).toLowerCase()}">${esc(x.vote)}</span></td><td>${esc(x.why)}</td></tr>`).join("")||'<tr><td colspan="4" class="empty-cell">No evidence available.</td></tr>';
const p=d.probabilities||{};const valid=[p.long,p.flat,p.short].every(v=>d.model_available!==false&&typeof v==="number"&&Number.isFinite(v)&&v>=0&&v<=1)&&Math.abs(p.long+p.flat+p.short-1)<0.01;const bars=[["Long",p.long,C.up],["Flat",p.flat,C.axis],["Short",p.short,C.down]].map(([k,v,col])=>`<div class="prob-row"><span class="prob-label">${k}</span><div class="prob-track" role="meter" aria-label="Raw ${k.toLowerCase()} probability" aria-valuemin="0" aria-valuemax="100" ${valid?`aria-valuenow="${(v*100).toFixed(1)}"`:'aria-valuetext="Unavailable"'}><div class="prob-fill" style="width:${valid?v*100:0}%;background:${col}"></div></div><span class="prob-value">${valid?(v*100).toFixed(1)+"%":"Unavailable"}</span></div>`).join("");$("overviewProbabilityBars").innerHTML=bars;

const adv=d.trade_advisor;if(adv&&$("advNarrative")){$("advBadge").textContent=adv.action_badge||"STAND ASIDE";$("advBadge").className=`advisor-badge ${adv.should_enter?(adv.likely_direction==="LONG"?"enter-long":"enter-short"):"stand-aside"}`;$("advNarrative").textContent=adv.narrative||"";$("advAction").textContent=adv.action||"STAND ASIDE";$("advAction").className=`adv-val ${adv.should_enter?(adv.likely_direction==="LONG"?"up":"down"):""}`;$("advDirection").textContent=`${adv.likely_direction} (${(adv.likely_probability_pct||0).toFixed(0)}%)`;$("advDirection").className=`adv-val ${adv.likely_direction==="LONG"?"up":adv.likely_direction==="SHORT"?"down":""}`;$("advTP").textContent=adv.take_profit_price?`$${money(adv.take_profit_price,2)} (+${(adv.take_profit_pct||0).toFixed(2)}%)`:"—";$("advSL").textContent=adv.stop_loss_price?`$${money(adv.stop_loss_price,2)} (-${(adv.stop_loss_pct||0).toFixed(2)}%)`:"—";$("advRR").textContent=adv.risk_reward_ratio||"2.0 : 1";$("advFee").textContent=adv.fee_hurdle_usd===0?"$0.00 (0 Fees)":(adv.breakeven_price?`BE: $${money(adv.breakeven_price,2)}`:"$0.00");}
$("walletAddress").textContent=d.wallet?`${d.wallet.slice(0,5)}...${d.wallet.slice(-5)}`:"Not configured";
refreshDetails();}
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
  accountReset(address?"Loading public account activity...":"Add a public address in Wallet settings below to view profits and trades.");
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
function candleBodyWidth(step,percent=appearance.bodyWidth) {return Math.min(step,Math.max(0.5,step*percent/100));}
function paintCandle(c,q,xx,y,width) {
  const up=+q.close>=+q.open,color=up?C.up:C.down;
  const hollow=appearance.candleStyle==="hollow"||(appearance.candleStyle==="hollow-up"&&up);
  const top=y(Math.max(+q.open,+q.close)),height=Math.max(1,y(Math.min(+q.open,+q.close))-top);
  c.save();
  if(appearance.wicks) {
    c.strokeStyle=appearance.matchWicks?color:((up?appearance.wickUp:appearance.wickDown)||color);
    c.lineWidth=appearance.wickWidth;c.beginPath();
    c.moveTo(xx,y(+q.high));c.lineTo(xx,top);c.moveTo(xx,top+height);c.lineTo(xx,y(+q.low));c.stroke();
  }
  if(!hollow) {c.globalAlpha=appearance.bodyOpacity/100;c.fillStyle=color;c.fillRect(xx-width/2,top,width,height);c.globalAlpha=1;}
  if(appearance.borders||hollow) {
    c.strokeStyle=color;c.lineWidth=Math.min(1,width);
    c.strokeRect(xx-width/2+c.lineWidth/2,top+c.lineWidth/2,Math.max(0,width-c.lineWidth),Math.max(0,height-c.lineWidth));
  }
  c.restore();
}
function drawCandlePreview() {
  const canvas=$("candlePreview");if(!canvas||!$("appearanceDialog").open)return;
  canvas.setAttribute("aria-label",appearance.style==="line"?"Illustrative price line showing the current appearance settings":"Illustrative bullish and bearish candles showing the current appearance settings");
  const box=canvas.getBoundingClientRect();if(!box.width||!box.height)return;
  const dpr=Math.min(2,devicePixelRatio||1);canvas.width=Math.round(box.width*dpr);canvas.height=Math.round(box.height*dpr);
  const c=canvas.getContext("2d");c.setTransform(dpr,0,0,dpr,0,0);c.fillStyle=C.bg;c.fillRect(0,0,box.width,box.height);
  if(appearance.grid)for(let i=1;i<4;i++){c.strokeStyle=C.grid;c.beginPath();c.moveTo(0,box.height*i/4);c.lineTo(box.width,box.height*i/4);c.stroke();}
  const values=[[30,48,22,43],[43,58,36,52],[52,56,32,37],[37,47,28,33],[33,53,29,49],[49,66,44,60],[60,64,40,46],[46,59,42,55]];
  const step=box.width/(values.length+1),y=value=>box.height-12-value/75*(box.height-24);
  if(appearance.style==="line") {c.strokeStyle=C.up;c.lineWidth=2;c.beginPath();values.forEach((q,i)=>i?c.lineTo((i+1)*step,y(q[3])):c.moveTo(step,y(q[3])));c.stroke();}
  else values.forEach(([open,high,low,close],i)=>paintCandle(c,{open,high,low,close},(i+1)*step,y,candleBodyWidth(Math.min(22,step))));
}
function resetChartView() {
  if(!S.data?.candles.length)return;
  const end=S.data.candles.length-1;S.range=[Math.max(0,end-179),end];S.followLive=true;S.hover=null;draw();
}
function syncLiveButton() {
  if(!$("followLive"))return;
  $("followLive").setAttribute("aria-pressed",String(S.followLive));
  $("followLive").textContent=S.followLive?"Following latest":"Jump to latest";
}
function last24h(candles,nowMs){const cutoff=(nowMs||Date.now())-86400000;return (candles||[]).filter(k=>+k.timestamp>=cutoff);}
function draw24h(){const canvas=$("dayChart");if(!canvas||$("overviewPanel").hidden)return;const box=canvas.getBoundingClientRect(),dpr=Math.min(2,devicePixelRatio||1);canvas.width=Math.floor(box.width*dpr);canvas.height=Math.floor(box.height*dpr);const c=canvas.getContext("2d");c.setTransform(dpr,0,0,dpr,0,0);c.fillStyle=C.bg;c.fillRect(0,0,box.width,box.height);const now=+S.data?.updated_at||Date.now();const win=last24h(S.data?.candles,now);if(win.length<2){c.fillStyle=C.axis;c.font="10px -apple-system, Segoe UI, sans-serif";c.textAlign="center";c.fillText("WAITING FOR 24H DATA",box.width/2,box.height/2);return;}
const pad={l:10,r:62,t:12,b:20},w=box.width-pad.l-pad.r,h=box.height-pad.t-pad.b;const closes=win.map(k=>+k.close);let lo=Math.min(...closes),hi=Math.max(...closes);const span=(hi-lo)||hi*.001;lo-=span*.06;hi+=span*.06;const t0=+win[0].timestamp,t1=+win[win.length-1].timestamp;const x=ts=>pad.l+(ts-t0)/((t1-t0)||1)*w,y=v=>pad.t+(hi-v)/(hi-lo)*h;const first=closes[0],last=closes[closes.length-1],up=last>=first,col=up?C.up:C.down;c.font="10px -apple-system, Segoe UI, sans-serif";
c.strokeStyle=appearance.grid?C.grid:"transparent";c.setLineDash([2,4]);c.strokeStyle=C.axis;c.beginPath();c.moveTo(pad.l,y(first));c.lineTo(pad.l+w,y(first));c.stroke();c.setLineDash([]);
c.globalAlpha=.12;c.beginPath();c.moveTo(x(t0),y(first));win.forEach((k,j)=>c.lineTo(x(+k.timestamp),y(+k.close)));c.lineTo(x(t1),pad.t+h);c.lineTo(x(t0),pad.t+h);c.closePath();c.fillStyle=col;c.fill();c.globalAlpha=1;
c.beginPath();win.forEach((k,j)=>j?c.lineTo(x(+k.timestamp),y(+k.close)):c.moveTo(x(+k.timestamp),y(+k.close)));c.strokeStyle=col;c.lineWidth=1.8;c.stroke();c.lineWidth=1;
c.beginPath();c.arc(x(t1),y(last),3,0,Math.PI*2);c.fillStyle=col;c.fill();
const dlo=Math.min(...closes),dhi=Math.max(...closes);c.fillStyle=C.axis;c.textAlign="left";c.fillText(`$${money(dhi)}`,pad.l+w+6,y(dhi)+3);c.fillText(`$${money(dlo)}`,pad.l+w+6,y(dlo)+3);
for(let hr=0;hr<24;hr+=6){const ts=t1-((t1/3600000)%1+ (23-hr))*0;const target=t1-(23-hr)*3600000-(t1%3600000);if(target<t0)continue;c.strokeStyle=appearance.grid?C.grid:"transparent";c.beginPath();c.moveTo(x(target),pad.t);c.lineTo(x(target),pad.t+h);c.stroke();c.fillStyle=C.axis;c.textAlign="center";const d=new Date(target);c.fillText(`${String(d.getUTCHours()).padStart(2,"0")}:00`,x(target),box.height-6);}
const chg=last/first-1;$("dayChange").textContent=`${pct(chg)} (24h)`;$("dayChange").className=`price-change ${valueTone(chg)}`;$("dayHigh").textContent=`$${money(dhi)}`;$("dayLow").textContent=`$${money(dlo)}`;canvas.setAttribute("aria-label",`SOL ${pct(chg)} over the last 24 hours, high $${money(dhi)}, low $${money(dlo)}`);}
function draw(){const canvas=$("chart");if(!canvas||$("overviewPanel").hidden)return;const box=canvas.getBoundingClientRect(),dpr=Math.min(2,devicePixelRatio||1);canvas.width=Math.floor(box.width*dpr);canvas.height=Math.floor(box.height*dpr);const c=canvas.getContext("2d");c.setTransform(dpr,0,0,dpr,0,0);c.fillStyle=C.bg;c.fillRect(0,0,box.width,box.height);const all=S.data?.candles||[];if(all.length<2){c.fillStyle=C.axis;c.textAlign="center";c.fillText("WAITING FOR CANDLES",box.width/2,box.height/2);return;}let [a,b]=S.range||[Math.max(0,all.length-180),all.length-1];a=Math.max(0,Math.min(a,all.length-2));b=Math.max(a+1,Math.min(b,all.length-1));S.range=[a,b];const v=all.slice(a,b+1),pad={l:12,r:76,t:20,b:28},w=box.width-pad.l-pad.r,h=box.height-pad.t-pad.b,vals=v.flatMap(x=>[+x.high,+x.low]);vals.push(+S.data.price);let lo=Math.min(...vals),hi=Math.max(...vals),span=(hi-lo)||1;lo-=span*.06;hi+=span*.06;span=hi-lo;const y=q=>pad.t+(hi-q)/span*h,x=i=>pad.l+(i-a+.5)/(b-a+1)*w;const step=w/(b-a+1),bw=candleBodyWidth(step);c.font="10px -apple-system, Segoe UI, sans-serif";
for(let i=0;i<=5;i++){const yy=pad.t+i*h/5;c.strokeStyle=appearance.grid?C.grid:"transparent";c.beginPath();c.moveTo(pad.l,yy);c.lineTo(pad.l+w,yy);c.stroke();c.fillStyle=C.axis;c.textAlign="right";c.fillText((hi-i*span/5).toFixed(2),box.width-8,yy+3);}const nt=Math.min(6,Math.max(2,Math.floor(w/140)));for(let i=0;i<=nt;i++){const ix=Math.round(a+i*(b-a)/nt),xx=x(ix);c.strokeStyle=appearance.grid?C.grid:"transparent";c.beginPath();c.moveTo(xx,pad.t);c.lineTo(xx,pad.t+h);c.stroke();const dt=new Date(all[ix].timestamp);c.fillStyle=C.axis;c.textAlign="center";c.fillText(`${String(dt.getUTCHours()).padStart(2,"0")}:${String(dt.getUTCMinutes()).padStart(2,"0")}`,xx,box.height-9);}
if(appearance.style==="line"){c.strokeStyle=C.up;c.lineWidth=1.8;c.beginPath();v.forEach((q,j)=>{if(j)c.lineTo(x(a+j),y(+q.close));else c.moveTo(x(a+j),y(+q.close));});c.stroke();c.lineWidth=1;}else v.forEach((q,j)=>paintCandle(c,q,x(a+j),y,bw));if(appearance.priceLine){const lp=+S.data.price,ly=y(lp);c.setLineDash([2,4]);c.strokeStyle=lp>=+v.at(-1).open?C.up:C.down;c.beginPath();c.moveTo(pad.l,ly);c.lineTo(pad.l+w,ly);c.stroke();c.setLineDash([]);c.fillStyle=lp>=+v.at(-1).open?C.up:C.down;c.fillRect(pad.l+w+2,ly-9,66,18);c.fillStyle=C.ink;c.textAlign="center";c.fillText(money(lp),pad.l+w+35,ly+3);
}
const tfms=({"1m":60,"5m":300,"15m":900,"1h":3600,"4h":14400}[S.tf]||60)*1000;for(const m of (appearance.signals?S.data.signals||[]:[])){if(m.time<all[a].timestamp||m.time>(all[b].timestamp+tfms))continue;const ix=Math.max(a,Math.min(b,Math.round(a+(m.time-all[a].timestamp)/(all[b].timestamp-all[a].timestamp||1)*(b-a)))),xx=x(ix),yy=y(+m.price)+(m.pred==="SHORT"?-14:14);c.beginPath();c.arc(xx,yy,8,0,Math.PI*2);c.fillStyle=m.pred==="SHORT"?C.down:C.up;c.fill();c.fillStyle=C.ink;c.textAlign="center";c.textBaseline="middle";c.fillText(m.pred==="SHORT"?"S":"B",xx,yy);c.textBaseline="alphabetic";}S.plot={pad,w,h};syncLiveButton();
let selected=all[b];
if(appearance.crosshair&&S.hover&&S.hover.x>=pad.l&&S.hover.x<=pad.l+w&&S.hover.y>=pad.t&&S.hover.y<=pad.t+h) {
  const ix=Math.max(a,Math.min(b,a+Math.floor((S.hover.x-pad.l)/step)));selected=all[ix];
  c.save();c.strokeStyle=C.axis;c.setLineDash([3,4]);c.lineWidth=1;c.beginPath();
  c.moveTo(x(ix),pad.t);c.lineTo(x(ix),pad.t+h);c.moveTo(pad.l,S.hover.y);c.lineTo(pad.l+w,S.hover.y);c.stroke();c.restore();
}
$("chartReadout").textContent=appearance.crosshair?`${new Date(selected.timestamp).toISOString().slice(5,16).replace("T"," ")} UTC  O ${money(selected.open)}  H ${money(selected.high)}  L ${money(selected.low)}  C ${money(selected.close)}`:"Drag to pan · Scroll to zoom · Double-click to reset";
}
function syncChartIntervals() {
  $("horizonSelect").disabled = !!S.tfSaving;
  document.querySelectorAll("#chartIntervals button").forEach(button => {
    button.setAttribute("aria-pressed", String(button.dataset.tf === S.tf));
    button.disabled = !!S.tfSaving;
  });
}
function init() {
  applyAppearance();
  const dialog=$("appearanceDialog"),appearanceTriggers=[$("appearanceToggle"),$("chartSettingsToggle")];
  let appearanceOpener=null;
  function openAppearance(event) {
    appearanceOpener=event.currentTarget;
    if(!dialog.open)dialog.showModal();
    dialog.querySelector(".appearance-body").scrollTop=0;
    requestAnimationFrame(drawCandlePreview);
    appearanceTriggers.forEach(button=>button.setAttribute("aria-expanded","true"));
  }
  appearanceTriggers.forEach(button=>button.onclick=openAppearance);
  for(const id of ["closeAppearance","doneAppearance"])$(id).onclick=()=>dialog.close();
  dialog.addEventListener("keydown",event=>{
    if(event.key!=="Tab")return;
    const controls=[...dialog.querySelectorAll("button, select, input")].filter(control=>!control.disabled&&control.getClientRects().length);
    const target=event.shiftKey?controls.at(-1):controls[0];
    if(document.activeElement===(event.shiftKey?controls[0]:controls.at(-1))) {event.preventDefault();target.focus();}
  });
  dialog.addEventListener("close",()=>{
    appearanceTriggers.forEach(button=>button.setAttribute("aria-expanded","false"));
    appearanceOpener?.focus({preventScroll:true});
  });
  const outsideDialog=event=>{const box=dialog.getBoundingClientRect();return event.clientX<box.left||event.clientX>box.right||event.clientY<box.top||event.clientY>box.bottom;};
  let backdropPressed=false;
  dialog.addEventListener("pointerdown",event=>{backdropPressed=event.target===dialog&&outsideDialog(event);});
  dialog.addEventListener("click",event=>{if(backdropPressed&&event.target===dialog&&outsideDialog(event))dialog.close();backdropPressed=false;});
  $("themeSelect").onchange=()=>{appearance.theme=$("themeSelect").value;saveAppearance();};
  $("paletteSelect").onchange=()=>{
    appearance.palette=$("paletteSelect").value;
    if(appearance.palette==="custom")for(const [id,key] of Object.entries(colorControls))appearance[key]=$(id).value;
    saveAppearance();
  };
  for(const [id,key] of Object.entries(colorControls))$(id).oninput=()=>{
    if(!validColor($(id).value))return;
    if(appearance.palette!=="custom")for(const [colorId,colorKey] of Object.entries(colorControls))appearance[colorKey]=$(colorId).value;
    appearance.palette="custom";appearance[key]=$(id).value;saveAppearance();
  };
  for(const [id,key] of Object.entries(wickControls))$(id).oninput=()=>{if(validColor($(id).value)){appearance[key]=$(id).value;saveAppearance();}};
  $("swapColors").onclick=()=>{
    for(const [id,key] of Object.entries(colorControls))appearance[key]=$(id).value;
    [appearance.up,appearance.down]=[appearance.down,appearance.up];appearance.palette="custom";saveAppearance();
  };
  for(const [id,key] of Object.entries(chartControls)) {
    const control=$(id);
    const change=()=>{
      if(key==="matchWicks"&&!control.checked){appearance.wickUp=$("wickUpColor").value;appearance.wickDown=$("wickDownColor").value;}
      appearance[key]=control.type==="checkbox"?control.checked:numericAppearance[key]?Number(control.value):control.value;
      saveAppearance();
    };
    if(control.type==="range")control.oninput=change;else control.onchange=change;
  }
  $("resetChartAppearance").onclick=()=>{for(const key of [...Object.values(chartControls),...Object.values(wickControls)])appearance[key]=appearanceDefaults[key];saveAppearance();};
  $("resetAppearance").onclick=()=>{appearance={...appearanceDefaults};saveAppearance();};

  $("viewEvidence").onclick = () => {$("indicatorsCard").scrollIntoView({behavior:matchMedia("(prefers-reduced-motion: reduce)").matches?"instant":"smooth",block:"start"});$("indicatorsCard").focus({preventScroll:true});};
  document.querySelector(".brand").onclick = event => {event.preventDefault(); window.scrollTo({top:0});};
  $("horizonSelect").onchange = async () => {
    const select = $("horizonSelect"), status = $("horizonStatus");
    if (S.tfSaving) {select.value=S.h; return;}
    const horizon=select.value;
    S.tfSaving=true; syncChartIntervals(); $("applySettings").disabled=true;
    status.textContent="Loading forecast...";
    try {
      const response=await fetch("/api/settings", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({hz:horizon})});
      const saved=await response.json();
      if (!response.ok) throw Error(saved.error || "Could not save forecast horizon");
      S.h=saved.hz; select.value=S.h; detailsLoaded=0;
      await update();
      status.textContent="";
    } catch(error) {select.value=S.h; status.textContent=`Forecast horizon unchanged: ${error.message}`;}
    finally {S.tfSaving=false; syncChartIntervals(); $("applySettings").disabled=false;}
  };
  $("settingsForm").onsubmit = async event => {
    event.preventDefault();
    if (S.busy || S.tfSaving) {$("settingsStatus").textContent="Please wait for the current market refresh to finish."; return;}
    const button = $("applySettings"), status = $("settingsStatus");
    const payload = {sol_wallet: $("walletInput").value.trim()};
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
        S.tf = saved.tf; S.range = null; S.drag = null; S.hover=null; S.followLive=true;
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
    canvas.focus({preventScroll:true});S.hover=null;
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", event => {
    if (!S.drag) {
      if(appearance.crosshair&&event.pointerType!=="touch") {const box=canvas.getBoundingClientRect();S.hover={x:event.clientX-box.left,y:event.clientY-box.top};requestAnimationFrame(draw);}
      return;
    }
    const shift=Math.round(-(event.clientX-S.drag.x)/Math.max(1,S.plot.w)*S.drag.count);
    const left=Math.max(0,Math.min(S.data.candles.length-S.drag.count,S.drag.start+shift));
    S.range=[left,left+S.drag.count-1];S.followLive=false;draw();
  });
  canvas.addEventListener("pointerleave",()=>{S.hover=null;requestAnimationFrame(draw);});
  for (const name of ["pointerup","pointercancel","lostpointercapture"]) canvas.addEventListener(name,()=>S.drag=null);
  canvas.addEventListener("wheel", event => {
    if (!S.range || !S.data || S.data.candles.length<2) return;
    event.preventDefault();
    const count=S.range[1]-S.range[0]+1, total=S.data.candles.length;
    const next=Math.max(Math.min(30,total),Math.min(total,Math.round(count*(event.deltaY<0?.8:1.25))));
    const rect=canvas.getBoundingClientRect(), anchor=Math.max(0,Math.min(1,(event.clientX-rect.left)/rect.width));
    const left=Math.max(0,Math.min(total-next,Math.round(S.range[0]+(count-next)*anchor)));
    S.range=[left,left+next-1];S.followLive=false;S.hover=null;draw();
  },{passive:false});
  $("resetZoom").onclick=resetChartView;
  canvas.addEventListener("dblclick",resetChartView);
  $("followLive").onclick=()=>{
    if(!S.data?.candles.length)return;
    const count=S.range?S.range[1]-S.range[0]+1:180,end=S.data.candles.length-1;
    S.range=[Math.max(0,end-count+1),end];S.followLive=true;S.hover=null;draw();
  };
  canvas.addEventListener("keydown",event=>{
    if(event.ctrlKey||event.metaKey||event.altKey)return;
    if(event.key.toLowerCase()==="r"){event.preventDefault();resetChartView();}
    if(event.key.toLowerCase()==="l"){event.preventDefault();$("followLive").click();}
  });
  new ResizeObserver(()=>requestAnimationFrame(draw)).observe(canvas);
  const dayCanvas=$("dayChart");if(dayCanvas)new ResizeObserver(()=>requestAnimationFrame(draw24h)).observe(dayCanvas);
  new ResizeObserver(()=>requestAnimationFrame(drawCandlePreview)).observe($("candlePreview"));
  fetch("/api/settings",{cache:"no-store"}).then(async response=>{
    if(!response.ok)throw Error("Preferences unavailable");return response.json();
  }).then(settings=>{
    S.h=settings.hz==="60m"?"1h":(settings.hz||"15m");
    S.tf=["1m","5m","15m","1h","4h","1d"].includes(settings.tf)?settings.tf:"1m";
    S.wallet=settings.sol_wallet||"";
    syncChartIntervals();$("horizonSelect").value=S.h;$("walletInput").value=S.wallet;
  }).catch(error=>{$("settingsStatus").textContent=error.message;}).finally(()=>{
    loadAccount(S.wallet);
    update();setInterval(()=>{if(!document.hidden){update();loadAccount(S.wallet);}},5000);
  });
  draw();
}
document.addEventListener("DOMContentLoaded",init);
