# Created by Oliver Meihls

# Argus Local Web Dashboard
#
# Lightweight localhost-only dashboard for debugging and command access.
# Runs alongside the headless main process, reading status via callbacks.

import asyncio
import errno
import json
import logging
import os
import time
import secrets as secrets_lib
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from aiohttp import web
from yarl import URL

logger = logging.getLogger('argus.dashboard')

# Inline HTML template (single page, no external deps)
_HTML = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Argus Dashboard</title>
<meta http-equiv="refresh" content="30">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Consolas','Courier New',monospace;background:#1a1a2e;color:#e0e0e0;padding:16px}
h1{color:#00d4ff;margin-bottom:12px;font-size:1.4em}
h2{color:#00d4ff;margin:16px 0 8px;font-size:1.1em;border-bottom:1px solid #333;padding-bottom:4px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.card{background:#16213e;border:1px solid #0f3460;border-radius:8px;padding:12px}
.ok{color:#4caf50}.warn{color:#ff9800}.err{color:#f44336}
table{width:100%;border-collapse:collapse;font-size:0.85em}
td,th{text-align:left;padding:4px 8px;border-bottom:1px solid #222}
th{color:#888}
pre{background:#0d1117;padding:8px;border-radius:4px;max-height:300px;overflow-y:auto;font-size:0.8em;white-space:pre-wrap}
input[type=text]{background:#0d1117;color:#e0e0e0;border:1px solid #0f3460;padding:6px 10px;border-radius:4px;width:70%;font-family:monospace}
button{background:#0f3460;color:#e0e0e0;border:none;padding:6px 16px;border-radius:4px;cursor:pointer;margin-left:8px}
button:hover{background:#1a4a8a}
#cmd-output{margin-top:8px}
.stat{font-size:1.4em;font-weight:bold}
.label{color:#888;font-size:0.85em}
</style></head><body>
<h1>Argus Dashboard</h1>
<div class="grid">
<div class="card">
<h2>System</h2>
<div id="system">Loading...</div>
</div>
<div class="card">
<h2>Providers</h2>
<div id="providers">Loading...</div>
</div>
<div class="card">
<h2>Detectors</h2>
<div id="detectors">Loading...</div>
</div>
<div class="card">
<h2>Internal</h2>
<div id="internal">Loading...</div>
</div>
<div class="card">
<h2>P&amp;L Summary</h2>
<div id="pnl">Loading...</div>
</div>
<div class="card">
<h2>Farm</h2>
<div id="farm">Loading...</div>
</div>
</div>
<div class="card" style="margin-top:16px">
<h2>Command</h2>
<div>
<input type="text" id="cmd-input" placeholder="/pnl, /status, /positions, /zombies, /dashboard ..." onkeydown="if(event.key==='Enter')runCmd()">
<button onclick="runCmd()">Run</button>
</div>
<pre id="cmd-output"></pre>
</div>
<div class="card" style="margin-top:16px">
<h2>Recent Logs</h2>
<pre id="logs">Loading...</pre>
</div>
<script>
function esc(s){let d=document.createElement('div');d.textContent=String(s);return d.innerHTML;}
async function load(){
  try{
    let r=await fetch('/api/status');let d=await r.json();
    // System
    let s=d.system||{};
    document.getElementById('system').innerHTML=
      '<div><span class="label">Uptime:</span> '+esc(s.uptime)+'</div>'+
      '<div><span class="label">DB Size:</span> '+esc(s.db_size_mb||'?')+' MB</div>'+
      '<div><span class="label">Boot phases:</span></div>'+
      '<pre>'+esc(s.boot_phases||'N/A')+'</pre>';
    // Providers
    let p=d.providers||{};let ph='<table><tr><th>Provider</th><th>Health</th><th>Age(s)</th><th>Counters</th></tr>';
    for(let k in p){let v=p[k]||{};let status=v.health||'unknown';
      let cls=(status==='ok')?'ok':(status==='warn'?'warn':(status==='alert'?'err':'warn'));
      let age=(v.last_msg_age_s!=null)?v.last_msg_age_s:'-';
      let counters=v.counters||{};
      let counts='msgs '+(counters.messages_total||0)+' / q '+(counters.quotes_total||0)+' / b '+(counters.bars_total||0)+' / m '+(counters.metrics_total||0);
      ph+='<tr><td>'+esc(k)+'</td><td class="'+cls+'">'+esc(status).toUpperCase()+'</td><td>'+esc(age)+'</td><td>'+esc(counts)+'</td></tr>';}
    ph+='</table>';document.getElementById('providers').innerHTML=ph;
    // Detectors
    let det=d.detectors||{};let dh='<table><tr><th>Detector</th><th>Health</th><th>Age(s)</th><th>Counters</th></tr>';
    for(let k in det){let v=det[k]||{};let status=v.health||'unknown';
      let cls=(status==='ok')?'ok':(status==='warn'?'warn':(status==='alert'?'err':'warn'));
      let age=(v.last_event_age_s!=null)?v.last_event_age_s:'-';
      let counters=v.counters||{};
      let counts='events '+(counters.events_total||0)+' / bars '+(counters.bars_total||0)+' / metrics '+(counters.metrics_total||0)+' / signals '+(counters.signals_total||0);
      dh+='<tr><td>'+esc(k)+'</td><td class="'+cls+'">'+esc(status).toUpperCase()+'</td><td>'+esc(age)+'</td><td>'+esc(counts)+'</td></tr>';}
    dh+='</table>';document.getElementById('detectors').innerHTML=dh;
    // Internal
    let i=d.internal||{};let ih='<table><tr><th>Component</th><th>Status</th><th>Age</th><th>Details</th></tr>';
    for(let k in i){let v=i[k]||{};let status=v.status||'unknown';
      let cls=(status==='ok')?'ok':(status==='degraded'?'warn':'err');
      let age=(v.staleness&&v.staleness.age_seconds!=null)?v.staleness.age_seconds:'-';
      let detail='-';
      if(k==='bar_builder'){detail='bars '+(v.extras&&v.extras.bars_emitted_total||0);}
      if(k==='persistence'){detail='buffer '+(v.extras&&v.extras.bar_buffer_size||0);}
      ih+='<tr><td>'+esc(k)+'</td><td class="'+cls+'">'+esc(status)+'</td><td>'+esc(age)+'</td><td>'+esc(detail)+'</td></tr>';}
    ih+='</table>';document.getElementById('internal').innerHTML=ih;
    // PnL
    let pnl=d.pnl||{};
    document.getElementById('pnl').innerHTML=
      '<div><span class="label">Today:</span> <span class="stat '+(pnl.today_pnl>=0?'ok':'err')+'">$'+(pnl.today_pnl||0).toFixed(2)+'</span></div>'+
      '<div><span class="label">MTD:</span> $'+(pnl.month_pnl||0).toFixed(2)+'</div>'+
      '<div><span class="label">YTD:</span> $'+(pnl.year_pnl||0).toFixed(2)+'</div>'+
      '<div><span class="label">Open positions:</span> '+(pnl.open_positions||0)+'</div>'+
      '<div><span class="label">Win rate MTD:</span> '+(pnl.win_rate_mtd||0).toFixed(0)+'%</div>';
    // Farm
    let f=d.farm||{};
    document.getElementById('farm').innerHTML=
      '<div><span class="label">Total configs:</span> '+(f.total_configs||0).toLocaleString()+'</div>'+
      '<div><span class="label">Active traders:</span> '+(f.active_traders||0).toLocaleString()+'</div>'+
      '<div><span class="label">Last eval:</span> '+esc(f.last_evaluation_symbol||'N/A')+'</div>';
    // Logs
    let logs=d.recent_logs||'No logs available';
    document.getElementById('logs').textContent=logs;
  }catch(e){console.error(e);}
}
async function runCmd(){
  let inp=document.getElementById('cmd-input');
  let cmd=inp.value.trim();if(!cmd)return;
  document.getElementById('cmd-output').textContent='Running...';
  try{
    let r=await fetch('/api/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({command:cmd})});
    let d=await r.json();
    document.getElementById('cmd-output').textContent=d.result||d.error||'No output';
  }catch(e){document.getElementById('cmd-output').textContent='Error: '+e;}
}
load();setInterval(load,30000);
</script>
</body></html>"""


class ArgusWebDashboard:
    # Lightweight local web dashboard for Argus.

    def __init__(self, host: str = '127.0.0.1', port: int = 8777, port_scan_range: int = 5):
        self.host = host
        self.port = port
        self._port_scan_range = max(1, port_scan_range)
        self._app = web.Application()
        self._runner = None

        # Callbacks set by orchestrator
        self._get_status: Optional[Callable] = None
        self._get_pnl: Optional[Callable] = None
        self._get_farm_status: Optional[Callable] = None
        self._get_providers: Optional[Callable] = None
        self._run_command: Optional[Callable] = None
        self._get_recent_logs: Optional[Callable] = None
        self._get_soak_summary: Optional[Callable] = None
        self._boot_phases: str = ''
        self._oauth_states: Dict[str, float] = {}

        self._app.router.add_get('/', self._handle_index)
        self._app.router.add_get('/api/status', self._handle_status)
        self._app.router.add_get('/debug/soak', self._handle_soak)
        self._app.router.add_get('/debug/tape/export', self._handle_tape_export)
        self._app.router.add_post('/api/command', self._handle_command)
        self._app.router.add_get('/oauth/tastytrade/start', self._handle_tastytrade_oauth_start)
        self._app.router.add_get('/oauth/tastytrade/callback', self._handle_tastytrade_oauth_callback)
        self._app.router.add_get('/callback', self._handle_tastytrade_oauth_callback)

    def set_callbacks(
        self,
        get_status=None,
        get_pnl=None,
        get_farm_status=None,
        get_providers=None,
        run_command=None,
        get_recent_logs=None,
        get_soak_summary=None,
        export_tape=None,
    ):
        if get_status: self._get_status = get_status
        if get_pnl: self._get_pnl = get_pnl
        if get_farm_status: self._get_farm_status = get_farm_status
        if get_providers: self._get_providers = get_providers
        if run_command: self._run_command = run_command
        if get_recent_logs: self._get_recent_logs = get_recent_logs
        if get_soak_summary: self._get_soak_summary = get_soak_summary
        if export_tape: self._export_tape = export_tape

    def set_boot_phases(self, text: str):
        self._boot_phases = text

    async def start(self):
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        base_port = self.port
        last_error = None
        for offset in range(self._port_scan_range):
            candidate_port = base_port + offset
            try:
                site = web.TCPSite(self._runner, self.host, candidate_port)
                await site.start()
                if candidate_port != base_port:
                    logger.warning(
                        "Dashboard port %s in use; using %s instead",
                        base_port,
                        candidate_port,
                    )
                self.port = candidate_port
                logger.info(f"Dashboard running at http://{self.host}:{self.port}")
                return
            except OSError as exc:
                last_error = exc
                if exc.errno == errno.EADDRINUSE:
                    continue
                raise
        await self._runner.cleanup()
        message = "Dashboard port already in use. Stop previous Argus instance or change port."
        logger.error(message)
        raise OSError(message) from last_error

    async def stop(self):
        if self._runner:
            await self._runner.cleanup()

    async def _handle_index(self, request):
        return web.Response(text=_HTML, content_type='text/html')

    async def _handle_status(self, request):
        start = time.perf_counter()
        status_code = 200

        def _ensure_json_safe(payload: Dict[str, Any]) -> Dict[str, Any]:
            try:
                json.dumps(payload)
                return payload
            except TypeError:
                return json.loads(json.dumps(payload, default=str))

        result: Dict[str, Any] = {}
        try:
            # System info
            from ..core.logger import _uptime
            system = {'uptime': _uptime(), 'boot_phases': self._boot_phases}
            if self._get_status:
                try:
                    s = await self._get_status()
                    internal = s.pop("internal", None) if isinstance(s, dict) else None
                    bus = s.pop("bus", None) if isinstance(s, dict) else None
                    db = s.pop("db", None) if isinstance(s, dict) else None
                    if isinstance(s, dict):
                        system.update(s)
                    if internal is not None:
                        result["internal"] = internal
                    if bus is not None:
                        result["bus"] = bus
                    if db is not None:
                        result["db"] = db
                except Exception as e:
                    system['error'] = str(e)
            result['system'] = system

            # Providers (prefer canonical snapshot)
            if "providers" not in result and self._get_providers:
                try:
                    result['providers'] = await self._get_providers()
                except Exception as e:
                    result['providers'] = {'error': str(e)}

            # PnL
            if self._get_pnl:
                try:
                    result['pnl'] = await self._get_pnl()
                except Exception as e:
                    result['pnl'] = {'error': str(e)}

            # Farm
            if self._get_farm_status:
                try:
                    result['farm'] = await self._get_farm_status()
                except Exception as e:
                    result['farm'] = {'error': str(e)}

            # Recent logs
            if self._get_recent_logs:
                try:
                    result['recent_logs'] = await self._get_recent_logs()
                except Exception as e:
                    result['recent_logs'] = str(e)
        except Exception as e:
            logger.error(f"/api/status error: {e}")
            result = {'error': str(e)}
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.info(f"/api/status {status_code} in {duration_ms:.1f}ms")

        return web.json_response(_ensure_json_safe(result), status=status_code)

    async def _handle_command(self, request):
        start = time.perf_counter()
        status_code = 200
        try:
            data = await request.json()
            cmd = data.get('command', '').strip()
            _ALLOWED_CMD_PREFIXES = {
                '/pnl', '/status', '/positions', '/zombies', '/dashboard',
                '/signals', '/help', '/regime', '/health', '/farm',
            }
            if not cmd:
                response = {'error': 'No command provided'}
            elif not any(cmd.startswith(prefix) for prefix in _ALLOWED_CMD_PREFIXES):
                response = {'error': f'Unknown command. Allowed prefixes: {sorted(_ALLOWED_CMD_PREFIXES)}'}
            elif self._run_command:
                result = await self._run_command(cmd)
                response = {'result': result}
            else:
                response = {'error': 'Command handler not configured'}
        except Exception as e:
            logger.error(f"/api/command error: {e}")
            response = {'error': str(e)}
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.info(f"/api/command {status_code} in {duration_ms:.1f}ms")

        return web.json_response(response, status=status_code)

    async def _handle_soak(self, request):
        # GET /debug/soak — single-payload soak summary.
        start = time.perf_counter()
        try:
            if self._get_soak_summary:
                result = await self._get_soak_summary()
            else:
                result = {"error": "Soak summary not configured"}
        except Exception as e:
            logger.error(f"/debug/soak error: {e}")
            result = {"error": str(e)}
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.info(f"/debug/soak 200 in {duration_ms:.1f}ms")

        def _ensure_json_safe(payload):
            try:
                json.dumps(payload)
                return payload
            except TypeError:
                return json.loads(json.dumps(payload, default=str))

        return web.json_response(_ensure_json_safe(result))

    async def _handle_tape_export(self, request):
        # GET /debug/tape/export?minutes=N — export tape to JSONL.
        start = time.perf_counter()
        try:
            minutes = request.query.get('minutes')
            minutes = int(minutes) if minutes else None
            
            # Use the orchestrator callback to get the tape data
            if hasattr(self, '_export_tape') and self._export_tape:
                result = await self._export_tape(minutes)
                return web.json_response(result)
            else:
                return web.json_response({"error": "Tape export not configured"}, status=501)
        except Exception as e:
            logger.error(f"/debug/tape/export error: {e}")
            return web.json_response({"error": str(e)}, status=500)
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.info(f"/debug/tape/export in {duration_ms:.1f}ms")

    async def _handle_tastytrade_oauth_start(self, request):
        from ..core.config import load_secrets
        from ..connectors.tastytrade_oauth import (
            TastytradeOAuthConfig,
            TASTYTRADE_AUTH_URL,
            TASTYTRADE_TOKEN_URL,
            build_authorize_url,
            parse_scopes,
            prune_states,
        )

        prune_states(self._oauth_states)
        secrets = load_secrets()
        oauth_cfg = secrets.get("tastytrade_oauth2", {}) or {}
        client_id = oauth_cfg.get("client_id", "")
        client_secret = oauth_cfg.get("client_secret", "")
        refresh_token = oauth_cfg.get("refresh_token")
        raw_scopes = oauth_cfg.get("scopes")
        if refresh_token:
            html = (
                "<html><body><h3>Tastytrade OAuth already configured.</h3>"
                "<p>Refresh token is already saved. You can close this tab.</p></body></html>"
            )
            return web.Response(text=html, content_type="text/html")
        if not client_id or not client_secret:
            html = (
                "<html><body><h3>Missing Tastytrade OAuth credentials.</h3>"
                "<p>Please set tastytrade_oauth2.client_id/client_secret in secrets.yaml.</p>"
                "</body></html>"
            )
            return web.Response(text=html, content_type="text/html", status=500)

        callback_url = str(
            URL.build(
                scheme=request.url.scheme,
                host=request.url.host,
                port=request.url.port,
                path="/oauth/tastytrade/callback",
            )
        )
        state = secrets_lib.token_urlsafe(24)
        self._oauth_states[state] = time.time()
        config = TastytradeOAuthConfig(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=callback_url,
            scopes=parse_scopes(raw_scopes),
            auth_url=oauth_cfg.get("auth_url")
            or os.getenv("TASTYTRADE_AUTH_URL")
            or TASTYTRADE_AUTH_URL,
            token_url=oauth_cfg.get("token_url")
            or os.getenv("TASTYTRADE_TOKEN_URL")
            or TASTYTRADE_TOKEN_URL,
        )
        auth_url = build_authorize_url(config, state)
        raise web.HTTPFound(auth_url)

    async def _handle_tastytrade_oauth_callback(self, request):
        from ..core.config import load_secrets, save_secrets
        from ..connectors.tastytrade_oauth import (
            TastytradeOAuthConfig,
            TASTYTRADE_AUTH_URL,
            TASTYTRADE_TOKEN_URL,
            exchange_code_for_tokens,
            parse_scopes,
            prune_states,
        )

        prune_states(self._oauth_states)
        params = request.query
        if "error" in params:
            html = (
                "<html><body><h3>OAuth error returned.</h3>"
                f"<p>{params.get('error_description') or params.get('error')}</p>"
                "</body></html>"
            )
            return web.Response(text=html, content_type="text/html", status=400)

        code = params.get("code")
        state = params.get("state")
        if not code:
            html = "<html><body><h3>Missing authorization code.</h3></body></html>"
            return web.Response(text=html, content_type="text/html", status=400)
        if not state or state not in self._oauth_states:
            html = (
                "<html><body><h3>Invalid or expired OAuth state.</h3>"
                "<p>Restart the flow from /oauth/tastytrade/start.</p></body></html>"
            )
            return web.Response(text=html, content_type="text/html", status=400)
        state_ts = self._oauth_states.get(state, 0)
        if time.time() - state_ts > 600:  # 10-minute expiry
            self._oauth_states.pop(state, None)
            html = "<html><body><h3>OAuth state expired. Please try again.</h3></body></html>"
            return web.Response(text=html, content_type="text/html", status=400)
        self._oauth_states.pop(state, None)

        secrets = load_secrets()
        oauth_cfg = secrets.get("tastytrade_oauth2", {}) or {}
        client_id = oauth_cfg.get("client_id", "")
        client_secret = oauth_cfg.get("client_secret", "")
        raw_scopes = oauth_cfg.get("scopes")
        callback_url = str(request.url.with_path("/oauth/tastytrade/callback").with_query({}))
        if not client_id or not client_secret:
            html = (
                "<html><body><h3>Missing Tastytrade OAuth credentials.</h3>"
                "<p>Set tastytrade_oauth2.client_id/client_secret in secrets.yaml.</p>"
                "</body></html>"
            )
            return web.Response(text=html, content_type="text/html", status=500)

        config = TastytradeOAuthConfig(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=callback_url,
            scopes=parse_scopes(raw_scopes),
            auth_url=oauth_cfg.get("auth_url")
            or os.getenv("TASTYTRADE_AUTH_URL")
            or TASTYTRADE_AUTH_URL,
            token_url=oauth_cfg.get("token_url")
            or os.getenv("TASTYTRADE_TOKEN_URL")
            or TASTYTRADE_TOKEN_URL,
        )
        try:
            token_data = await exchange_code_for_tokens(config, code)
        except Exception as exc:
            logger.error("Tastytrade token exchange failed: %s", exc)
            html = (
                "<html><body><h3>Token exchange failed.</h3>"
                f"<p>{exc}</p></body></html>"
            )
            return web.Response(text=html, content_type="text/html", status=500)

        refresh_token = token_data.get("refresh_token")
        if not refresh_token and isinstance(token_data.get("data"), dict):
            refresh_token = token_data["data"].get("refresh_token")
        if not refresh_token:
            html = (
                "<html><body><h3>No refresh token returned.</h3>"
                "<p>Check your OAuth client scopes and permissions.</p></body></html>"
            )
            return web.Response(text=html, content_type="text/html", status=500)

        oauth_section = secrets.setdefault("tastytrade_oauth2", {})
        oauth_section["refresh_token"] = refresh_token
        access_token = token_data.get("access_token")
        if not access_token and isinstance(token_data.get("data"), dict):
            access_token = token_data["data"].get("access_token")
        if access_token:
            oauth_section["access_token"] = access_token
        save_secrets(secrets)
        html = (
            "<html><body><h3>Refresh token saved.</h3>"
            "<p>You can close this tab.</p></body></html>"
        )
        return web.Response(text=html, content_type="text/html")
