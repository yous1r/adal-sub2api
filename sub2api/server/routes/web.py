"""``--web`` management UI: account CRUD, import/export, quota, reload.

Mounted only when ``settings.web`` is set, because every route here can read
or write bearer tokens and Clerk cookies.  All of them go through
:func:`sub2api.server.auth.unauthorized` — an open admin surface would hand
out working AdaL credentials to anyone who can reach the port.

The page is one self-contained HTML document (inline CSS + ``fetch``): no
template engine, no CDN, no build step.  Listings redact ``token`` and
``cookies`` to a length + last-4 fingerprint; only the explicit export
endpoint emits secrets, and only to an authenticated caller.

The fastest way to add an account is the device-code flow: ``POST
/admin/api/device/start`` returns AdaL's verification URL plus a 9-character
user code, and ``POST /admin/api/device/claim`` polls until the human
authorizes there, then registers a fresh ``sub2api-pool-*`` session and writes
the row itself — the operator types nothing into sub2api.  Pending flows live
in ``app.state.device_flows`` (one uvicorn worker by design), and the
``device_code`` never reaches the browser: it is a bearer-granting capability.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any
from uuid import uuid4

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from ...core.errors import AuthError
from ...core.pool import DEFAULT_MAX_CONCURRENT
from ..auth import unauthorized

_PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>sub2api 控制台</title>
<style>
:root{color-scheme:dark;
  --bg:#020617;--panel:rgba(30,41,59,.5);--panel-solid:#1e293b;
  --line:rgba(51,65,85,.5);--fg:#f1f5f9;--mut:#94a3b8;--dim:#64748b;
  --acc:#14b8a6;--acc2:#2dd4bf;--acc-deep:#0d9488;
  --bad:#ef4444;--ok:#10b981;--warn:#f59e0b;--info:#8b5cf6}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;min-height:100vh;background:
  radial-gradient(at 40% 20%,rgba(20,184,166,.12) 0,transparent 50%),
  radial-gradient(at 80% 0%,rgba(6,182,212,.08) 0,transparent 50%),
  radial-gradient(at 0% 50%,rgba(20,184,166,.08) 0,transparent 50%),
  var(--bg);
  color:var(--fg);font:14px/1.5 system-ui,-apple-system,"Segoe UI","PingFang SC",
  "Microsoft YaHei",sans-serif;-webkit-font-smoothing:antialiased}
::selection{background:rgba(20,184,166,.2)}
::-webkit-scrollbar{height:6px;width:6px}::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{border-radius:9999px;background:transparent}
*:hover::-webkit-scrollbar-thumb{background:rgba(71,85,105,.5)}
header{position:sticky;top:0;z-index:10;backdrop-filter:blur(20px);
  background:rgba(2,6,23,.8);border-bottom:1px solid var(--line)}
.hwrap{max-width:1200px;margin:0 auto;padding:14px 24px;display:flex;
  gap:14px;align-items:center;flex-wrap:wrap}
.logo{width:34px;height:34px;border-radius:10px;display:flex;align-items:center;
  justify-content:center;background:linear-gradient(135deg,var(--acc) 0,var(--acc-deep) 100%);
  color:#fff;font-weight:700;font-size:15px;box-shadow:0 4px 16px rgba(20,184,166,.3)}
h1{font-size:15px;margin:0;font-weight:600;letter-spacing:.02em}
.mut{color:var(--mut)}.dim{color:var(--dim)}
.pill{border:1px solid var(--line);border-radius:9999px;padding:2px 10px;
  font-size:11px;background:rgba(30,41,59,.5)}
.pill.live{color:var(--ok);border-color:rgba(16,185,129,.4)}
.spacer{flex:1}
#key{width:220px}
main{max-width:1200px;margin:0 auto;padding:24px;display:grid;gap:20px}
.card{background:var(--panel);backdrop-filter:blur(12px);border:1px solid var(--line);
  border-radius:16px;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.card-h{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;
  align-items:center;gap:10px;flex-wrap:wrap}
.card-h h2{font-size:13px;font-weight:600;margin:0;letter-spacing:.02em}
.card-b{padding:20px}
.ic{width:40px;height:40px;border-radius:12px;display:flex;align-items:center;
  justify-content:center;font-size:17px;flex-shrink:0}
.ic.teal{background:rgba(20,184,166,.15);color:var(--acc2)}
.ic.green{background:rgba(16,185,129,.15);color:var(--ok)}
.ic.amber{background:rgba(245,158,11,.15);color:var(--warn)}
.ic.violet{background:rgba(139,92,246,.15);color:var(--info)}
.ic.red{background:rgba(239,68,68,.15);color:var(--bad)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px}
.stat{display:flex;gap:12px;align-items:flex-start;background:var(--panel);
  border:1px solid var(--line);border-radius:16px;padding:16px;
  box-shadow:0 1px 3px rgba(0,0,0,.06)}
.stat .lb{font-size:12px;color:var(--mut)}
.stat .v{font-size:22px;font-weight:700;margin-top:2px;font-variant-numeric:tabular-nums}
.stat .sub{font-size:11px;color:var(--dim);margin-top:2px}
.big{font-size:26px}
.bar{height:6px;border-radius:9999px;background:rgba(51,65,85,.6);overflow:hidden;
  margin-top:8px}
.bar i{display:block;height:100%;border-radius:9999px;
  background:linear-gradient(90deg,var(--acc),var(--acc2));transition:width .4s ease}
.bar.hot i{background:linear-gradient(90deg,#f59e0b,#ef4444)}
.bar.maxed i{background:var(--bad)}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:9px 10px;border-bottom:1px solid var(--line);vertical-align:middle}
th{color:var(--mut);font-weight:500;font-size:12px;white-space:nowrap}
tr:last-child td{border-bottom:none}
td.num{font-variant-numeric:tabular-nums}
input{background:rgba(2,6,23,.6);color:var(--fg);border:1px solid var(--line);
  border-radius:10px;padding:7px 11px;font:inherit;transition:border-color .2s}
input:focus{outline:none;border-color:var(--acc);box-shadow:0 0 0 3px rgba(20,184,166,.25)}
input::placeholder{color:var(--dim)}
button{background:rgba(30,41,59,.8);color:var(--fg);border:1px solid var(--line);
  border-radius:10px;padding:7px 14px;font:inherit;font-size:13px;font-weight:500;
  cursor:pointer;transition:all .15s}
button:hover{border-color:var(--acc);color:var(--acc2)}
button:active{transform:scale(.98)}
button.primary{background:linear-gradient(135deg,var(--acc),var(--acc-deep));
  border-color:transparent;color:#fff;box-shadow:0 4px 14px rgba(20,184,166,.3)}
button.primary:hover{filter:brightness(1.1);color:#fff}
button.danger:hover{border-color:var(--bad);color:var(--bad)}
button.sm{padding:4px 10px;font-size:12px;border-radius:8px}
button:disabled{opacity:.5;cursor:not-allowed;transform:none}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.tag{border:1px solid var(--line);border-radius:9999px;padding:1px 9px;
  font-size:11px;display:inline-block}
.tag.alive{color:var(--ok);border-color:rgba(16,185,129,.4);background:rgba(16,185,129,.1)}
.tag.unknown{color:var(--warn);border-color:rgba(245,158,11,.4);background:rgba(245,158,11,.1)}
.tag.dead{color:var(--bad);border-color:rgba(239,68,68,.4);background:rgba(239,68,68,.1)}
.tag.limited{color:var(--bad);border-color:rgba(239,68,68,.4)}
.qgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px}
.qcard{background:rgba(2,6,23,.4);border:1px solid var(--line);border-radius:14px;padding:14px}
.qcard .top{display:flex;justify-content:space-between;align-items:baseline;gap:8px}
.qcard .email{font-size:13px;font-weight:600;overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap}
.qcard .nums{font-size:12px;color:var(--mut);margin-top:2px;font-variant-numeric:tabular-nums}
.qcard .nums b{color:var(--fg);font-size:14px}
.spent{color:var(--acc2)}.cost{color:var(--warn)}
#log{white-space:pre-wrap;color:var(--mut);font-size:12px;min-height:18px;
  font-family:ui-monospace,Consolas,monospace;max-height:130px;overflow:auto}
summary{cursor:pointer;font-size:12px;color:var(--mut)}
summary:hover{color:var(--acc2)}
a{color:var(--acc2);text-decoration:none}
a:hover{text-decoration:underline}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style></head><body>
<header><div class="hwrap">
  <div class="logo">S2</div>
  <h1>sub2api 控制台</h1>
  <span class="pill mut" id="ver"></span>
  <span class="spacer"></span>
  <input id="key" type="password" placeholder="API key">
  <button class="primary" onclick="saveKey()">连接</button>
  <button onclick="loadAll()">刷新</button>
</div></header>
<main>
<section class="card">
  <div class="card-h"><h2>订阅额度</h2><span class="dim" style="font-size:12px">
    试用账号按周限额计，付费账号按月度额度</span>
    <span class="spacer"></span><span class="pill mut" id="pool_note"></span></div>
  <div class="card-b"><div class="stats" id="quota_stats"></div></div>
</section>
<section class="card">
  <div class="card-h"><h2>本地计量 — 近 24 小时</h2>
    <span class="dim" style="font-size:12px">按账号单独立账，费率经实测校准</span></div>
  <div class="card-b"><div class="stats" id="local_stats"></div></div>
</section>
<section class="card">
  <div class="card-h"><h2>账号池</h2><span class="pill mut" id="acc_count"></span>
    <span class="spacer"></span>
    <button class="sm" onclick="doReload()">重载连接池</button></div>
  <div class="card-b"><div class="qgrid" id="quota_cards"></div>
  <table style="margin-top:14px"><thead><tr>
    <th>邮箱</th><th>session_id</th><th>token</th><th>cookies</th>
    <th style="width:84px">并发上限</th><th>状态</th><th>24h 计费</th><th style="width:200px"></th>
  </tr></thead><tbody id="rows"></tbody></table>
  <details style="margin-top:14px">
  <summary>手动添加 / 更新（需自备 token）</summary>
  <div class="row" style="margin-top:12px">
    <input id="n_sid" placeholder="session_id" style="width:190px">
    <input id="n_tok" placeholder="token (JWT)" style="width:250px">
    <input id="n_mail" placeholder="邮箱" style="width:170px">
    <input id="n_max" placeholder="并发" value="4" style="width:70px">
    <button onclick="addAccount()">保存</button>
  </div>
  </details></div>
</section>
<section class="card">
  <div class="card-h"><h2>添加账号</h2></div>
  <div class="card-b">
    <div class="row">
      <input id="p_text" placeholder="粘贴含 token 的内容（JWT、adal_oauth_creds.json、accounts.json 条目）"
        style="flex:1;min-width:280px">
      <button class="primary" onclick="pasteImport()">粘贴导入</button>
    </div>
    <div class="row" style="margin-top:12px">
      <button id="dev_btn" onclick="deviceStart()">设备码登录</button>
      <span class="dim" style="font-size:12px">无需 token：点击后在 AdaL 页面输入验证码，账号自动入池</span>
    </div>
    <div class="row" id="dev_box" style="display:none;margin-top:12px;padding:12px;
      border:1px dashed var(--line);border-radius:12px">
      <a id="dev_url" target="_blank" rel="noopener">打开授权页面</a>
      <b id="dev_code" style="font-size:20px;letter-spacing:.16em;color:var(--acc2)"></b>
      <button class="sm" onclick="devCopy()">复制验证码</button>
      <span class="mut" id="dev_state" style="font-size:12px"></span>
    </div>
    <div class="row" style="margin-top:14px">
      <input id="file" type="file" accept="application/json" style="width:290px">
      <button onclick="doImport()">导入 accounts.json</button>
      <button onclick="doExport()">下载导出</button>
    </div>
  </div>
</section>
<section class="card"><div class="card-h"><h2>日志</h2></div>
  <div class="card-b"><div id="log"></div></div></section>
</main>
<script>
let KEY = sessionStorage.getItem("s2a_key") || "";
document.getElementById("key").value = KEY;
function saveKey(){KEY=document.getElementById("key").value.trim();
  sessionStorage.setItem("s2a_key",KEY);loadAll();}
function log(m){const el=document.getElementById("log");
  el.textContent=m+"\\n"+el.textContent.split("\\n").slice(0,4).join("\\n");}
function hdr(extra){const h=extra||{};if(KEY)h["x-api-key"]=KEY;return h;}
async function api(path,opts){
  const o=opts||{};o.headers=hdr(o.headers);
  const r=await fetch(path,o);
  const text=await r.text();
  let body=null;try{body=text?JSON.parse(text):null}catch(e){body={raw:text}}
  if(!r.ok)throw new Error(r.status+" "+(text||"").slice(0,300));
  return body;
}
function esc(v){return String(v==null?"":v).replace(/[&<>"]/g,
  c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;"}[c]));}
function usd(v){return "$"+(Math.round(v*1e4)/1e4);}
function stat(icon,cls,label,value,sub,bar){
  return '<div class="stat"><div class="ic '+cls+'">'+icon+'</div><div style="min-width:0;flex:1">'
    +'<div class="lb">'+esc(label)+'</div><div class="v">'+esc(value)+'</div>'
    +(sub?'<div class="sub">'+sub+'</div>':"")
    +(bar?'<div class="bar '+bar.cls+'"><i style="width:'+Math.min(100,bar.pct)+'%"></i></div>':"")
    +"</div></div>";
}
function quotaCard(r){
  const spendable=r.total||0,remaining=r.remaining||0,used=r.used||0;
  const pct=spendable>0?(used/spendable*100):0;
  const cls=pct>=100?"maxed":(pct>=90?"hot":"");
  const basis=r.limit_basis==="weekly-trial"?"周限额":"月度额度";
  const monthly=(r.monthly_total!=null)
    ?'<div class="sub">月度额度 '+usd(r.monthly_total)
      +" · 剩 "+usd(r.monthly_remaining||0)+"</div>":"";
  const cost=(r.cost_usd!=null)
    ?'<div class="sub">24h 计费 <b class="cost">'+usd(r.cost_usd)+"</b> · "
      +esc(r.requests||0)+" 次请求</div>":"";
  const reset=r.period_end?'<div class="sub">'+esc(r.period_end)+" 重置</div>":"";
  return '<div class="qcard"><div class="top">'
    +'<span class="email" title="'+esc(r.email||"")+'">'+esc(r.email||"(未知邮箱)")+"</span>"
    +'<span class="tag '+(cls==="maxed"?"dead":(cls==="hot"?"unknown":"alive"))+'">'
    +esc(basis)+"</span></div>"
    +'<div class="nums">剩 <b class="spent">'+usd(remaining)+"</b> / "+usd(spendable)
    +"&nbsp;·&nbsp;已用 "+usd(used)+"</div>"
    +'<div class="bar '+cls+'"><i style="width:'+Math.min(100,pct)+'%"></i></div>'
    +monthly+cost+reset+"</div>";
}
async function loadAll(){
  try{
    const h=await api("/v1/health");
    document.getElementById("ver").textContent="v"+h.version+" · "+h.channel.channel;
  }catch(e){log(String(e.message));}
  await Promise.all([loadQuota(),loadLocal(),loadAccounts()]);
}
async function loadQuota(){
  try{
    const u=await api("/v1/usage");
    const d=u.sub2api||{};const rows=d.detail||[];
    document.getElementById("pool_note").textContent=u.extra||"";
    const trial=rows.filter(r=>r.limit_basis==="weekly-trial").length;
    document.getElementById("quota_stats").innerHTML=[
      stat("$","teal","总剩余额度",usd(u.remaining||0),"共 "+esc(rows.length)+" 个账号"),
      stat("◈","green","总额度",usd(u.total||0),
        trial?trial+" 个试用账号（按周限额）":"全部按月度额度"),
      stat("→","violet","已使用",usd(u.used||0),"占总额 "+(
        u.total>0?Math.round(u.used/(u.total||1)*100)+"%":"\\u2014")),
      stat("◉","amber","套餐",esc(u.planName||"\\u2014"),esc(u.unit||"USD")),
    ].join("");
  }catch(e){log("额度: "+String(e.message));}
}
async function loadLocal(){
  try{
    const u=await api("/v1/usage");
    const d=(u.sub2api||{});
    if(d.requests==null){document.getElementById("local_stats").innerHTML=
      '<div class="dim" style="font-size:13px">未开启计量库（启动时未配置 --db），无本地计量。</div>';
      return;}
    const t=d.tokens||{};
    const byS=d.by_session||{};
    const nAcct=Object.keys(byS).length;
    document.getElementById("local_stats").innerHTML=[
      stat("✻","teal","请求数",esc(d.requests),nAcct?nAcct+" 个账号有消耗":""),
      stat("＄","amber","累计成本",usd(d.cost_usd||0),"费率经实测校准"),
      stat("↑","green","输入 tokens",esc((t.input||0).toLocaleString()),
        "cache read "+esc((t.cache_read||0).toLocaleString())),
      stat("↓","violet","输出 tokens",esc((t.output||0).toLocaleString()),
        "含思考 "+esc((t.reasoning||0).toLocaleString())),
    ].join("");
  }catch(e){log("计量: "+String(e.message));}
}
async function loadAccounts(){
  let rows=[];
  try{rows=(await api("/admin/api/accounts")).accounts||[];}
  catch(e){log(String(e.message));return;}
  document.getElementById("acc_count").textContent=rows.length+" 个账号";
  // Per-account subscription quota (billing basis + monthly fallback) and
  // per-account 24h metering, joined onto the account list.
  let usage={},detail={};
  try{const u=await api("/v1/usage");const d=u.sub2api||{};
    usage=d.by_session||{};
    for(const r of (d.detail||[]))detail[r.session_id]=r;
  }catch(e){}
  document.getElementById("quota_cards").innerHTML=rows.map(r=>{
    const q=detail[r.session_id]||{};
    const s=usage[r.session_id]||{};
    return quotaCard({email:r.email,total:q.total,used:q.used,remaining:q.remaining,
      limit_basis:q.limit_basis,monthly_total:q.monthly_total,
      monthly_remaining:q.monthly_remaining,period_end:q.period_end,
      cost_usd:s.cost_usd,requests:s.requests});
  }).join("")||'<div class="dim" style="font-size:13px">还没有账号——用上方「粘贴导入」或「设备码登录」添加第一个。</div>';
  document.getElementById("rows").innerHTML=rows.map(r=>{
    const s=esc(r.session_id);const u=usage[r.session_id]||{};
    const statusCls={alive:"alive",unknown:"unknown",dead:"dead"}[r.status]||"unknown";
    return "<tr><td>"+esc(r.email||"\\u2014")+"</td><td class=mut style='font-size:12px'>"+s+
      "</td><td class=mut style='font-size:12px'>"+esc(r.token)+"</td><td class=mut>"+
      esc(r.cookies)+"</td>"+
      '<td><input value="'+esc(r.max_concurrent)+'" id="m_'+s+'"></td>'+
      '<td><span class="tag '+statusCls+'">'+esc(r.status)+"</span>"+(r.reason&&r.reason!=="ok"
        ?'<div class="sub">'+esc(r.reason)+"</div>":"")+"</td>"+
      '<td class="num cost">'+(u.cost_usd!=null?usd(u.cost_usd):"\\u2014")+"</td><td class=row>"+
      '<button class="sm" onclick="saveRow(\\''+s+'\\')">保存</button>'+
      '<button class="sm" onclick="quota(\\''+s+'\\')">额度</button>'+
      '<button class="sm danger" onclick="delRow(\\''+s+'\\')">删除</button></td></tr>';
  }).join("");
}
async function pasteImport(){
  const el=document.getElementById("p_text");
  const text=el.value.trim();
  if(!text){log("先粘贴内容");return;}
  try{
    const r=await api("/admin/api/accounts/paste",{method:"POST",
      headers:{"content-type":"application/json"},
      body:JSON.stringify({text:text})});
    el.value="";
    log("已导入 "+r.session_id+(r.email?" ("+r.email+")":"")+
      " · cookies:"+r.cookies+
      (r.registered?"":" [session_unregistered: "+r.detail+"]"));
    await loadAll();
  }catch(e){log(String(e.message));}
}
let DEV=null;
async function deviceStart(){
  if(DEV){log("设备码登录已在进行中");return;}
  const btn=document.getElementById("dev_btn");btn.disabled=true;DEV="starting";
  try{
    const r=await api("/admin/api/device/start",{method:"POST",
      headers:{"content-type":"application/json"},body:"{}"});
    DEV=r.flow_id;
    document.getElementById("dev_url").href=r.verification_url;
    document.getElementById("dev_code").textContent=r.user_code;
    document.getElementById("dev_box").style.display="flex";
    document.getElementById("dev_state").textContent="等待授权…";
    try{window.open(r.verification_url,"_blank","noopener");}catch(e){}
    log("输入验证码 "+r.user_code+" · "+Math.round(r.expires_in)+"s 内有效 · "
      +r.verification_url);
    await devPoll(r.flow_id,Date.now()+r.expires_in*1000);
  }catch(e){log(String(e.message));}
  finally{DEV=null;btn.disabled=false;}
}
function devCopy(){
  const c=document.getElementById("dev_code").textContent;
  if(navigator.clipboard)navigator.clipboard.writeText(c);
  log("已复制 "+c);
}
async function devPoll(id,deadline){
  const st=document.getElementById("dev_state");
  while(Date.now()<deadline){
    await new Promise(ok=>setTimeout(ok,2500));
    let out=null;
    try{out=await api("/admin/api/device/claim",{method:"POST",
      headers:{"content-type":"application/json"},
      body:JSON.stringify({flow_id:id})});}
    catch(e){const m=String(e.message);
      if(/^(404|410)/.test(m)){st.textContent=m;log(m);return;}
      st.textContent="轮询失败，重试中";continue;}
    if(out.status==="ok"){
      st.textContent="已导入";
      document.getElementById("dev_box").style.display="none";
      log("已导入 "+out.session_id+(out.email?" ("+out.email+")":"")+
        (out.registered?"":" [session_unregistered: "+out.detail+"]"));
      await loadAll();return;
    }
    if(out.status!=="pending"){st.textContent=out.status;
      log("设备码登录结束: "+out.status);return;}
  }
  st.textContent="已过期";
  log("设备码已过期，请重新开始");
}
async function addAccount(){
  const body={session_id:document.getElementById("n_sid").value.trim(),
    token:document.getElementById("n_tok").value.trim(),
    email:document.getElementById("n_mail").value.trim(),
    max_concurrent:Number(document.getElementById("n_max").value||4)};
  try{await api("/admin/api/accounts",{method:"POST",
    headers:{"content-type":"application/json"},body:JSON.stringify(body)});
    log("已保存 "+body.session_id);await loadAll();}
  catch(e){log(String(e.message));}
}
async function saveRow(sid){
  const v=Number(document.getElementById("m_"+sid).value||4);
  try{await api("/admin/api/accounts/"+encodeURIComponent(sid),{method:"PATCH",
    headers:{"content-type":"application/json"},
    body:JSON.stringify({max_concurrent:v})});
    log("已更新 "+sid);await loadAccounts();}
  catch(e){log(String(e.message));}
}
async function delRow(sid){
  if(!confirm("删除账号 "+sid+" ？"))return;
  try{await api("/admin/api/accounts/"+encodeURIComponent(sid),{method:"DELETE"});
    log("已删除 "+sid);await loadAll();}
  catch(e){log(String(e.message));}
}
async function quota(sid){
  try{const q=await api("/admin/api/accounts/"+encodeURIComponent(sid)+"/quota");
    const w=q.weekly_usage||{};
    log(sid+" → 剩余 $"+(w.remaining!=null?w.remaining:"?")+" / 周限额 $"
      +(w.limit!=null?w.limit:"?")+" · 重置 "+(w.resets_at||"\\u2014"));}
  catch(e){log(String(e.message));}
}
async function doImport(){
  const f=document.getElementById("file").files[0];
  if(!f){log("先选择文件");return;}
  try{const payload=JSON.parse(await f.text());
    const r=await api("/admin/api/accounts/import",{method:"POST",
      headers:{"content-type":"application/json"},body:JSON.stringify(payload)});
    log("导入 "+r.imported+" 个，跳过 "+r.skipped+" 个");await loadAll();}
  catch(e){log(String(e.message));}
}
async function doExport(){
  try{const data=await api("/admin/api/accounts/export");
    const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],
      {type:"application/json"}));
    const a=document.createElement("a");a.href=url;a.download="accounts.json";
    a.click();URL.revokeObjectURL(url);
    log("已导出 "+data.accounts.length+" 个账号");}
  catch(e){log(String(e.message));}
}
async function doReload(){
  try{const r=await api("/admin/api/reload",{method:"POST"});
    log("连接池已重载: "+JSON.stringify(r));await loadAll();}
  catch(e){log(String(e.message));}
}
loadAll();
</script></body></html>
"""


def fingerprint(value: Any) -> str:
    """One-way redaction: length plus the last 4 characters.

    Enough to tell two credentials apart and to spot a truncated paste;
    useless for authenticating as the account.
    """
    if isinstance(value, list):
        return f"{len(value)} cookie(s)"
    text = value if isinstance(value, str) else ""
    if not text:
        return ""
    if len(text) <= 4:
        return f"{len(text)} chars"
    return f"{len(text)} chars …{text[-4:]}"


def redact(row: dict) -> dict:
    """An account row safe to render: secrets replaced by fingerprints."""
    safe = dict(row)
    safe["token"] = fingerprint(row.get("token"))
    safe["cookies"] = fingerprint(row.get("cookies"))
    return safe


def _no_store() -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "error": {
                "code": "store_unavailable",
                "message": "no metering/credential store is open",
            }
        },
    )


# Device-code flows are held in memory between ``start`` and ``claim`` because
# ``device_code`` grants a bearer: the browser only ever sees the verification
# URL and the user code.  Measured against the live platform: ``expires_in`` is
# 600 s, ``/api/auth/device/initiate`` answered in 10.7 s once and exceeded a
# 15 s read timeout on another attempt, and ``/api/auth/device/poll`` has
# exceeded 20 s — so both calls get a deliberately generous deadline rather
# than failing a login on upstream latency.
_DEVICE_FLOW_TTL = 600.0
_DEVICE_INITIATE_TIMEOUT = 45.0
_DEVICE_POLL_TIMEOUT = 45.0
_MAX_DEVICE_FLOWS = 64


def _flows(request: Request) -> dict[str, dict[str, Any]]:
    """Pending device flows for this app, created on first use."""
    flows = getattr(request.app.state, "device_flows", None)
    if flows is None:
        flows = {}
        request.app.state.device_flows = flows
    return flows


# A Clerk JWT always starts with the base64 of ``{"`` — ``eyJ`` — and has three
# dot-separated base64url segments.  Scanning for that shape is what lets the
# UI accept *anything* the operator can copy (a bare JWT, the whole
# ``adal_oauth_creds.json``, an ``accounts.json`` entry, a pasted curl command)
# instead of demanding they know which field is which.
_JWT_RE = re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")


def _harvest_cookies(node: Any) -> list[dict]:
    """First ``[{name, value}, …]`` cookie list found anywhere in ``node``."""
    if isinstance(node, list):
        if node and all(
            isinstance(c, dict) and "name" in c and "value" in c for c in node
        ):
            return [{"name": str(c["name"]), "value": str(c["value"])} for c in node]
        for item in node:
            found = _harvest_cookies(item)
            if found:
                return found
    elif isinstance(node, dict):
        for key in ("cookies", "cookie_jar"):
            found = _harvest_cookies(node.get(key))
            if found:
                return found
        for value in node.values():
            found = _harvest_cookies(value)
            if found:
                return found
    return []


def extract_credentials(text: str) -> tuple[str, list[dict]]:
    """Pull ``(token, cookies)`` out of arbitrary pasted text.

    Accepts a bare JWT, ``~/.adal/adal_oauth_creds.json``, an
    ``accounts.json`` entry, or any blob that merely contains one — the token
    is found by shape, not by field name, and the first JWT carrying a Clerk
    ``sub`` claim wins.  Cookies are optional; when the paste happens to
    include them the row becomes re-mintable, which a device-code login never
    is.  Returns ``("" , [])`` when nothing usable is present.
    """
    from ...channels import adal_cloud as adal

    token = ""
    for candidate in _JWT_RE.findall(text or ""):
        if adal.clerk_id_from_token(candidate):
            token = candidate
            break
    if not token:
        return "", []
    cookies: list[dict] = []
    stripped = (text or "").strip()
    if stripped.startswith(("{", "[")):
        try:
            cookies = _harvest_cookies(json.loads(stripped))
        except (ValueError, TypeError):
            cookies = []
    return token, cookies


# The user lookup is authenticated by the JWT ``sub`` alone, so even a stale
# token resolves its own address.  It MUST carry that token as the bearer: the
# endpoint answers 403 ``Cannot query other users`` when the Authorization
# header belongs to a different account, and the shared channel client carries
# the single-account bearer.  Timeout is generous on purpose: the platform has
# been measured taking >10 s on auth endpoints, and losing the email would
# leave an unlabelled row in the pool.
_EMAIL_TIMEOUT = 20.0


async def _resolve_email(request: Request, token: str) -> str:
    """Best-effort account email from the JWT's ``sub``; ``""`` on any failure."""
    from ...channels import adal_cloud as adal

    clerk_id = adal.clerk_id_from_token(token)
    if not clerk_id:
        return ""
    url = f"{adal.ADAL_APP_URL}/api/user/by-clerk-id/{clerk_id}"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
    channel = request.app.state.channel
    client = getattr(channel, "_client", None)
    try:
        if client is not None:
            resp = await client.get(url, headers=headers, timeout=_EMAIL_TIMEOUT)
        else:
            settings = request.app.state.settings
            async with httpx.AsyncClient(
                proxy=getattr(settings, "proxy", None) or None
            ) as temp:
                resp = await temp.get(url, headers=headers, timeout=_EMAIL_TIMEOUT)
        if resp.status_code != 200:
            return ""
        user = resp.json()
    except (httpx.HTTPError, ValueError):
        return ""
    return str(user.get("email") or "") if isinstance(user, dict) else ""


async def _adopt_token(
    request: Request,
    store: Any,
    token: str,
    cookies: list[dict] | None = None,
) -> dict[str, Any]:
    """Turn a bare AdaL bearer into a live pool row.

    Re-importing the same account updates its existing row instead of adding a
    second one: two rows for one AdaL user would double its apparent
    concurrency and split its cache affinity.  Identity is the JWT ``sub``
    (Clerk user id), the only stable field across re-mints.

    A brand-new row's session id is generated here (``sub2api-pool-<12 hex>``,
    adal-registrar's convention) because the platform upserts on
    ``(user, session_id)`` — so nobody has to invent one.  A registration
    failure still stores the token (it is the valuable half) but marks the row
    ``session_unregistered`` rather than claiming it is alive.

    Cookies are stored when the caller had them; a device-code login has none,
    so such a row can never be re-minted through Clerk.  That is acceptable:
    the AdaL edge validates the Clerk session (``sid``), not the JWT ``exp`` —
    an aged device-flow token keeps proxying.
    """
    from ...channels import adal_cloud as adal

    clerk_id = adal.clerk_id_from_token(token)
    existing = None
    if clerk_id:
        for row in await store.accounts():
            if adal.clerk_id_from_token(row["token"]) == clerk_id:
                existing = row
                break
    session_id = (
        existing["session_id"] if existing else f"sub2api-pool-{uuid4().hex[:12]}"
    )
    registered = True
    detail = ""
    try:
        await asyncio.to_thread(
            adal.register_session, token=token, session_id=session_id
        )
    except AuthError as exc:
        registered = False
        detail = str(exc)
    # A failed lookup must not blank an address the row already carries.
    email = await _resolve_email(request, token) or (
        existing["email"] if existing else ""
    )
    await store.upsert_account(
        session_id=session_id,
        token=token,
        cookies=cookies or (existing["cookies"] if existing else []),
        email=email,
        max_concurrent=int(
            (existing["max_concurrent"] if existing else 0) or DEFAULT_MAX_CONCURRENT
        ),
        status="alive" if registered else "unknown",
        reason="ok" if registered else "session_unregistered",
        detail=detail,
    )
    # Adopt it immediately — an import that needs a second "reload pool" click
    # is not an import.  A refresh failure must not discard a stored account.
    reloaded = True
    try:
        await request.app.state.channel.refresh()
    except Exception:  # noqa: BLE001 - the row is already persisted
        reloaded = False
    return {
        "status": "ok",
        "session_id": session_id,
        "email": email,
        "cookies": len(cookies or []),
        "updated": existing is not None,
        "registered": registered,
        "pool_reloaded": reloaded,
        "detail": detail,
    }


def router() -> APIRouter:
    """The ``/admin`` router. Mounted only when ``--web``/``SUB2API_WEB=1``."""
    api = APIRouter(prefix="/admin", tags=["admin"])

    def store_of(request: Request) -> Any:
        return getattr(request.app.state, "usage_store", None)

    @api.get("", response_class=HTMLResponse)
    @api.get("/", response_class=HTMLResponse)
    async def page(request: Request):
        # The page itself carries no data — the key is entered in the browser
        # and every /admin/api call is authenticated — so it is served
        # unauthenticated on purpose. Nothing here leaks without the key.
        return HTMLResponse(_PAGE)

    @api.get("/api/accounts")
    async def list_accounts(request: Request):
        denial = unauthorized(request)
        if denial is not None:
            return denial
        store = store_of(request)
        if store is None:
            return _no_store()
        rows = await store.accounts()
        return {"accounts": [redact(r) for r in rows]}

    @api.post("/api/accounts")
    async def upsert_account(request: Request):
        denial = unauthorized(request)
        if denial is not None:
            return denial
        store = store_of(request)
        if store is None:
            return _no_store()
        body = await request.json()
        session_id = str(body.get("session_id") or "").strip()
        token = str(body.get("token") or "").strip()
        if not session_id or not token:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "code": "bad_request",
                        "message": "session_id and token are required",
                    }
                },
            )
        await store.upsert_account(
            session_id=session_id,
            token=token,
            cookies=body.get("cookies") or [],
            email=str(body.get("email") or ""),
            max_concurrent=int(body.get("max_concurrent") or 4),
            status=body.get("status"),
        )
        return {"session_id": session_id, "ok": True}

    @api.patch("/api/accounts/{session_id}")
    async def patch_account(session_id: str, request: Request):
        denial = unauthorized(request)
        if denial is not None:
            return denial
        store = store_of(request)
        if store is None:
            return _no_store()
        rows = {r["session_id"]: r for r in await store.accounts()}
        current = rows.get(session_id)
        if current is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error": {"code": "not_found", "message": "unknown session_id"}
                },
            )
        body = await request.json()
        await store.upsert_account(
            session_id=session_id,
            token=str(body.get("token") or current["token"]),
            cookies=body.get("cookies") or current["cookies"],
            email=str(body.get("email", current["email"]) or ""),
            max_concurrent=int(
                body.get("max_concurrent") or current["max_concurrent"] or 4
            ),
            status=body.get("status"),
        )
        return {"session_id": session_id, "ok": True}

    @api.delete("/api/accounts/{session_id}")
    async def delete_account(session_id: str, request: Request):
        denial = unauthorized(request)
        if denial is not None:
            return denial
        store = store_of(request)
        if store is None:
            return _no_store()
        removed = await store.delete_account(session_id)
        if not removed:
            return JSONResponse(
                status_code=404,
                content={
                    "error": {"code": "not_found", "message": "unknown session_id"}
                },
            )
        return {"session_id": session_id, "deleted": True}

    @api.post("/api/accounts/import")
    async def import_accounts(request: Request):
        denial = unauthorized(request)
        if denial is not None:
            return denial
        store = store_of(request)
        if store is None:
            return _no_store()
        try:
            payload = await request.json()
        except (ValueError, json.JSONDecodeError):
            payload = None
        if not isinstance(payload, dict):
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "code": "bad_request",
                        "message": "expected an accounts.json object",
                    }
                },
            )
        return await store.import_accounts(payload)

    @api.get("/api/accounts/export")
    async def export_accounts(request: Request):
        denial = unauthorized(request)
        if denial is not None:
            return denial
        store = store_of(request)
        if store is None:
            return _no_store()
        payload = await store.export_accounts()
        # Unredacted by design: this is the accounts.json round-trip shape.
        return Response(
            content=json.dumps(payload, indent=2, ensure_ascii=False),
            media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="accounts.json"'},
        )

    @api.get("/api/accounts/{session_id}/quota")
    async def account_quota(session_id: str, request: Request):
        denial = unauthorized(request)
        if denial is not None:
            return denial
        store = store_of(request)
        if store is None:
            return _no_store()
        rows = {r["session_id"]: r for r in await store.accounts()}
        row = rows.get(session_id)
        if row is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error": {"code": "not_found", "message": "unknown session_id"}
                },
            )
        from ...channels.adal_cloud import fetch_credits_raw

        channel = request.app.state.channel
        client = getattr(channel, "_client", None)
        if client is not None:
            payload = await fetch_credits_raw(client, row["token"])
        else:
            settings = request.app.state.settings
            async with httpx.AsyncClient(
                proxy=getattr(settings, "proxy", None) or None
            ) as temp:
                payload = await fetch_credits_raw(temp, row["token"])
        if payload is None:
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "code": "quota_unavailable",
                        "message": "credit balance lookup failed",
                    }
                },
            )
        return payload

    @api.post("/api/accounts/paste")
    async def paste_account(request: Request):
        """Import one account from arbitrary pasted text.

        The counterpart to the device flow for an operator who already *has*
        credentials somewhere: paste the JWT, the whole
        ``adal_oauth_creds.json``, or any blob containing them, and the token
        is located by shape.  ``session_id`` is generated, so the form has
        exactly one field.
        """
        denial = unauthorized(request)
        if denial is not None:
            return denial
        store = store_of(request)
        if store is None:
            return _no_store()
        try:
            body = await request.json()
        except (ValueError, json.JSONDecodeError):
            body = None
        raw = ""
        if isinstance(body, dict):
            raw = str(body.get("text") or "")
        elif isinstance(body, str):
            raw = body
        token, cookies = extract_credentials(raw)
        if not token:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "code": "no_credentials",
                        "message": "no AdaL access token found in the pasted text",
                    }
                },
            )
        return await _adopt_token(request, store, token, cookies)

    @api.post("/api/device/start")
    async def device_start(request: Request):
        """Begin a device-code login; returns only what the human must see."""
        denial = unauthorized(request)
        if denial is not None:
            return denial
        store = store_of(request)
        if store is None:
            return _no_store()
        from ...channels import adal_cloud as adal

        try:
            init = await asyncio.to_thread(
                adal.initiate_device_flow, timeout=_DEVICE_INITIATE_TIMEOUT
            )
        except AuthError as exc:
            return JSONResponse(
                status_code=502,
                content={"error": {"code": "device_flow_failed", "message": str(exc)}},
            )
        flows = _flows(request)
        now = time.monotonic()
        for stale in [k for k, v in flows.items() if v["expires_at"] <= now]:
            del flows[stale]
        while len(flows) >= _MAX_DEVICE_FLOWS:
            del flows[min(flows, key=lambda k: flows[k]["started_at"])]
        expires_in = float(init.get("expires_in") or _DEVICE_FLOW_TTL)
        flow_id = uuid4().hex
        flows[flow_id] = {
            "device_code": str(init.get("device_code") or ""),
            "started_at": now,
            "expires_at": now + expires_in,
        }
        return {
            "flow_id": flow_id,
            "verification_url": str(init.get("verification_url") or ""),
            "user_code": str(init.get("user_code") or ""),
            "expires_in": expires_in,
        }

    @api.post("/api/device/claim")
    async def device_claim(request: Request):
        """Poll one pending flow; on authorization, import the account."""
        denial = unauthorized(request)
        if denial is not None:
            return denial
        store = store_of(request)
        if store is None:
            return _no_store()
        try:
            body = await request.json()
        except (ValueError, json.JSONDecodeError):
            body = None
        if not isinstance(body, dict):
            body = {}
        flow_id = str(body.get("flow_id") or "").strip()
        flows = _flows(request)
        entry = flows.get(flow_id)
        if entry is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error": {
                        "code": "unknown_flow",
                        "message": "unknown or already completed device flow",
                    }
                },
            )
        if entry["expires_at"] <= time.monotonic():
            del flows[flow_id]
            return JSONResponse(
                status_code=410,
                content={
                    "error": {
                        "code": "flow_expired",
                        "message": "the device code expired; start a new login",
                    }
                },
            )
        from ...channels import adal_cloud as adal

        try:
            result = await asyncio.to_thread(
                adal.poll_device_flow,
                entry["device_code"],
                timeout=_DEVICE_POLL_TIMEOUT,
            )
        except AuthError as exc:
            # The flow survives a transport hiccup: the browser polls again.
            return JSONResponse(
                status_code=502,
                content={"error": {"code": "device_poll_failed", "message": str(exc)}},
            )
        token = str(result.get("token") or "")
        if not token:
            status = str(result.get("status") or "pending")
            if status in ("expired", "denied"):
                del flows[flow_id]
            return {"status": status}
        del flows[flow_id]
        return await _adopt_token(request, store, token)

    @api.post("/api/reload")
    async def reload_pool(request: Request):
        denial = unauthorized(request)
        if denial is not None:
            return denial
        channel = request.app.state.channel
        await channel.refresh()
        health = await channel.health()
        return {"reloaded": True, "pool": health.get("pool", {})}

    return api
