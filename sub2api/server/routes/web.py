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
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>sub2api admin</title>
<style>
:root{color-scheme:dark;--bg:#0f1115;--panel:#171a21;--line:#272c37;--fg:#e6e8ee;
--mut:#8b93a7;--acc:#6ea8fe;--bad:#f0616d;--ok:#4ec9a0}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace}
header{padding:16px 20px;border-bottom:1px solid var(--line);display:flex;
gap:16px;align-items:baseline;flex-wrap:wrap}
h1{font-size:16px;margin:0;letter-spacing:.04em}
.mut{color:var(--mut)}
main{padding:20px;display:grid;gap:20px;max-width:1200px}
section{background:var(--panel);border:1px solid var(--line);border-radius:8px;
padding:16px}
h2{font-size:13px;margin:0 0 12px;text-transform:uppercase;letter-spacing:.08em;
color:var(--mut)}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line);
vertical-align:middle}
th{color:var(--mut);font-weight:500}
input{background:#0c0e12;color:var(--fg);border:1px solid var(--line);
border-radius:4px;padding:5px 7px;font:inherit;width:100%}
button{background:#222735;color:var(--fg);border:1px solid var(--line);
border-radius:4px;padding:5px 11px;font:inherit;cursor:pointer}
button:hover{border-color:var(--acc)}
button.bad:hover{border-color:var(--bad);color:var(--bad)}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}
.stat{border:1px solid var(--line);border-radius:6px;padding:10px 12px}
.stat b{display:block;font-size:18px;font-weight:600}
.tag{border:1px solid var(--line);border-radius:10px;padding:1px 8px;font-size:11px}
.limited{color:var(--bad);border-color:var(--bad)}
.live{color:var(--ok);border-color:var(--ok)}
#log{white-space:pre-wrap;color:var(--mut);font-size:12px;min-height:18px}
summary{cursor:pointer;font-size:12px}
a{color:var(--acc)}
</style></head><body>
<header>
  <h1>sub2api admin</h1>
  <span class="mut" id="ver"></span>
  <span class="row" style="margin-left:auto">
    <input id="key" type="password" placeholder="API key" style="width:230px">
    <button onclick="saveKey()">use key</button>
    <button onclick="loadAll()">refresh</button>
  </span>
</header>
<main>
<section>
  <h2>usage &mdash; last 24h</h2>
  <div class="grid" id="totals"></div>
</section>
<section>
  <h2>&#19968;&#38190;&#23548;&#20837;&#36134;&#21495;</h2>
  <div class="row">
    <input id="p_text" placeholder="&#31896;&#36148; token / adal_oauth_creds.json &#20869;&#23481;" style="flex:1;min-width:280px">
    <button onclick="pasteImport()">&#31896;&#36148;&#23548;&#20837;</button>
  </div>
  <div class="row mut" style="margin-top:6px;font-size:12px">
    &#31896;&#36148;&#20219;&#24847;&#21547; token &#30340;&#20869;&#23481;&#65288;JWT&#12289;adal_oauth_creds.json&#12289;accounts.json &#26465;&#30446;&#65289;&#65292;session_id &#33258;&#21160;&#29983;&#25104;
  </div>
  <div class="row" style="margin-top:14px">
    <button id="dev_btn" onclick="deviceStart()">&#24320;&#22987;&#35774;&#22791;&#30721;&#30331;&#24405;</button>
    <span class="mut">&#25110;&#32773;&#23436;&#20840;&#19981;&#38656;&#35201; token&#65306;&#28857;&#20987;&#21518;&#22312; AdaL &#39029;&#38754;&#36755;&#20837;&#39564;&#35777;&#30721;&#65292;&#36134;&#21495;&#33258;&#21160;&#20837;&#27744;</span>
  </div>
  <div class="row" id="dev_box" style="display:none;margin-top:12px">
    <a id="dev_url" target="_blank" rel="noopener">&#25171;&#24320;&#25480;&#26435;&#39029;&#38754;</a>
    <b id="dev_code" style="font-size:20px;letter-spacing:.16em"></b>
    <button onclick="devCopy()">&#22797;&#21046;&#39564;&#35777;&#30721;</button>
    <span class="mut" id="dev_state"></span>
  </div>
</section>
<section>
  <h2>accounts</h2>
  <table><thead><tr>
    <th>session_id</th><th>email</th><th>token</th><th>cookies</th>
    <th style="width:90px">max</th><th>status</th><th style="width:210px"></th>
  </tr></thead><tbody id="rows"></tbody></table>
  <details style="margin-top:12px">
  <summary class="mut">&#25163;&#21160;&#28155;&#21152; / &#26356;&#26032;&#65288;&#38656;&#33258;&#22791; token&#65289;</summary>
  <div class="row" style="margin-top:10px">
    <input id="n_sid" placeholder="session_id" style="width:190px">
    <input id="n_tok" placeholder="token (JWT)" style="width:230px">
    <input id="n_mail" placeholder="email" style="width:160px">
    <input id="n_max" placeholder="max" value="4" style="width:70px">
    <button onclick="addAccount()">add / update</button>
  </div>
  </details>
</section>
<section>
  <h2>import / export</h2>
  <div class="row">
    <input id="file" type="file" accept="application/json" style="width:290px">
    <button onclick="doImport()">import accounts.json</button>
    <button onclick="doExport()">download export</button>
    <button onclick="doReload()">reload pool</button>
  </div>
</section>
<section><h2>log</h2><div id="log"></div></section>
</main>
<script>
let KEY = sessionStorage.getItem("s2a_key") || "";
document.getElementById("key").value = KEY;
function saveKey(){KEY=document.getElementById("key").value.trim();
  sessionStorage.setItem("s2a_key",KEY);loadAll();}
function log(m){document.getElementById("log").textContent=m;}
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
async function loadAll(){
  try{
    const h=await api("/v1/health");
    document.getElementById("ver").textContent="v"+h.version+" \\u00b7 "+h.channel.channel;
  }catch(e){log(String(e.message));}
  try{
    const u=await api("/v1/usage");
    const d=u.sub2api||{};const t=d.tokens||{};
    document.getElementById("totals").innerHTML=[
      ["requests",d.requests==null?"\\u2014":d.requests],
      ["local cost",d.cost_usd==null?"\\u2014":"$"+d.cost_usd],
      ["in tokens",t.input==null?"\\u2014":t.input],
      ["out tokens",t.output==null?"\\u2014":t.output],
      ["cache read",t.cache_read==null?"\\u2014":t.cache_read],
      ["subscription left",u.remaining+" "+(u.unit||"")],
    ].map(([k,v])=>'<div class="stat"><span class="mut">'+esc(k)+
      '</span><b>'+esc(v)+"</b></div>").join("");
  }catch(e){log(String(e.message));}
  await loadAccounts();
}
async function loadAccounts(){
  let rows=[];
  try{rows=(await api("/admin/api/accounts")).accounts||[];}
  catch(e){log(String(e.message));return;}
  document.getElementById("rows").innerHTML=rows.map(r=>{
    const s=esc(r.session_id);
    return "<tr><td>"+s+"</td><td>"+esc(r.email)+"</td><td class=mut>"+
      esc(r.token)+"</td><td class=mut>"+esc(r.cookies)+"</td>"+
      '<td><input value="'+esc(r.max_concurrent)+'" id="m_'+s+'"></td>'+
      "<td>"+esc(r.status)+"</td><td class=row>"+
      '<button onclick="saveRow(\\''+s+'\\')">save</button>'+
      '<button onclick="quota(\\''+s+'\\')">quota</button>'+
      '<button class=bad onclick="delRow(\\''+s+'\\')">delete</button></td></tr>';
  }).join("");
}
async function pasteImport(){
  const el=document.getElementById("p_text");
  const text=el.value.trim();
  if(!text){log("\\u5148\\u7c98\\u8d34\\u5185\\u5bb9");return;}
  try{
    const r=await api("/admin/api/accounts/paste",{method:"POST",
      headers:{"content-type":"application/json"},
      body:JSON.stringify({text:text})});
    el.value="";
    log("\\u5df2\\u5bfc\\u5165 "+r.session_id+(r.email?" ("+r.email+")":"")+
      " \\u00b7 cookies:"+r.cookies+
      (r.registered?"":" [session_unregistered: "+r.detail+"]"));
    await loadAll();
  }catch(e){log(String(e.message));}
}
let DEV=null;
async function deviceStart(){
  if(DEV){log("\\u8bbe\\u5907\\u7801\\u767b\\u5f55\\u5df2\\u5728\\u8fdb\\u884c\\u4e2d");return;}
  const btn=document.getElementById("dev_btn");btn.disabled=true;DEV="starting";
  try{
    const r=await api("/admin/api/device/start",{method:"POST",
      headers:{"content-type":"application/json"},body:"{}"});
    DEV=r.flow_id;
    document.getElementById("dev_url").href=r.verification_url;
    document.getElementById("dev_code").textContent=r.user_code;
    document.getElementById("dev_box").style.display="flex";
    document.getElementById("dev_state").textContent="\\u7b49\\u5f85\\u6388\\u6743\\u2026";
    try{window.open(r.verification_url,"_blank","noopener");}catch(e){}
    log("\\u8f93\\u5165\\u9a8c\\u8bc1\\u7801 "+r.user_code+" \\u00b7 "+
      Math.round(r.expires_in)+"s \\u5185\\u6709\\u6548 \\u00b7 "+r.verification_url);
    await devPoll(r.flow_id,Date.now()+r.expires_in*1000);
  }catch(e){log(String(e.message));}
  finally{DEV=null;btn.disabled=false;}
}
function devCopy(){
  const c=document.getElementById("dev_code").textContent;
  if(navigator.clipboard)navigator.clipboard.writeText(c);
  log("\\u5df2\\u590d\\u5236 "+c);
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
      st.textContent="\\u8f6e\\u8be2\\u5931\\u8d25\\uff0c\\u91cd\\u8bd5\\u4e2d";continue;}
    if(out.status==="ok"){
      st.textContent="\\u5df2\\u5bfc\\u5165";
      document.getElementById("dev_box").style.display="none";
      log("\\u5df2\\u5bfc\\u5165 "+out.session_id+(out.email?" ("+out.email+")":"")+
        (out.registered?"":" [session_unregistered: "+out.detail+"]"));
      await loadAll();return;
    }
    if(out.status!=="pending"){st.textContent=out.status;
      log("\\u8bbe\\u5907\\u7801\\u767b\\u5f55\\u7ed3\\u675f: "+out.status);return;}
  }
  st.textContent="\\u5df2\\u8fc7\\u671f";
  log("\\u8bbe\\u5907\\u7801\\u5df2\\u8fc7\\u671f\\uff0c\\u8bf7\\u91cd\\u65b0\\u5f00\\u59cb");
}
async function addAccount(){
  const body={session_id:document.getElementById("n_sid").value.trim(),
    token:document.getElementById("n_tok").value.trim(),
    email:document.getElementById("n_mail").value.trim(),
    max_concurrent:Number(document.getElementById("n_max").value||4)};
  try{await api("/admin/api/accounts",{method:"POST",
    headers:{"content-type":"application/json"},body:JSON.stringify(body)});
    log("saved "+body.session_id);await loadAccounts();}
  catch(e){log(String(e.message));}
}
async function saveRow(sid){
  const v=Number(document.getElementById("m_"+sid).value||4);
  try{await api("/admin/api/accounts/"+encodeURIComponent(sid),{method:"PATCH",
    headers:{"content-type":"application/json"},
    body:JSON.stringify({max_concurrent:v})});
    log("updated "+sid);await loadAccounts();}
  catch(e){log(String(e.message));}
}
async function delRow(sid){
  try{await api("/admin/api/accounts/"+encodeURIComponent(sid),{method:"DELETE"});
    log("deleted "+sid);await loadAccounts();}
  catch(e){log(String(e.message));}
}
async function quota(sid){
  try{const q=await api("/admin/api/accounts/"+encodeURIComponent(sid)+"/quota");
    log(sid+" \\u2192 "+JSON.stringify(q));}
  catch(e){log(String(e.message));}
}
async function doImport(){
  const f=document.getElementById("file").files[0];
  if(!f){log("pick a file first");return;}
  try{const payload=JSON.parse(await f.text());
    const r=await api("/admin/api/accounts/import",{method:"POST",
      headers:{"content-type":"application/json"},body:JSON.stringify(payload)});
    log("imported "+r.imported+", skipped "+r.skipped);await loadAccounts();}
  catch(e){log(String(e.message));}
}
async function doExport(){
  try{const data=await api("/admin/api/accounts/export");
    const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],
      {type:"application/json"}));
    const a=document.createElement("a");a.href=url;a.download="accounts.json";
    a.click();URL.revokeObjectURL(url);log("exported "+data.accounts.length+" account(s)");}
  catch(e){log(String(e.message));}
}
async function doReload(){
  try{const r=await api("/admin/api/reload",{method:"POST"});
    log("pool reloaded: "+JSON.stringify(r));await loadAll();}
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
