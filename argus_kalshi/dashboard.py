# Created by Oliver Meihls

# Web dashboard for Argus Kalshi — matches terminal UI layout, fully decoupled from backend.
#
# When visualizer_process == "separate", the backend runs one snapshot producer
# and pushes to (1) IPC for terminal UI, (2) this dashboard via a callback.
# Open http://localhost:<dashboard_port> in a browser.

from __future__ import annotations

import json
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, Optional

log = logging.getLogger("argus_kalshi.dashboard")

_lock = threading.Lock()
_snapshot: Optional[Dict[str, Any]] = None


def set_snapshot(snapshot: Dict[str, Any]) -> None:
    # Called from IPC broadcast loop (async main thread).
    with _lock:
        global _snapshot
        _snapshot = snapshot


def get_snapshot() -> Optional[Dict[str, Any]]:
    # Called from HTTP handler (server thread).
    with _lock:
        return _snapshot


# Replicate terminal layout: header, MARKETS (Strike/Fair/History/Ask/Edge/Expires/Signal), STATS, ORDERS, HISTORY, DWARF RANKING
_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Argus Kalshi</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: ui-monospace, "Cascadia Code", monospace; font-size: 13px; margin: 12px; background: #0d1117; color: #c9d1d9; }
    .header-row { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 16px; margin-bottom: 8px; }
    .brand { color: #f0883e; font-weight: bold; }
    .sim { color: #58a6ff; font-weight: bold; }
    .section { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 10px 12px; margin-bottom: 10px; }
    .section h2 { font-size: 0.8em; color: #f0883e; margin: 0 0 8px 0; font-weight: bold; }
    table { border-collapse: collapse; width: 100%; font-size: 12px; }
    th, td { padding: 3px 8px; text-align: left; border-bottom: 1px solid #21262d; }
    th { color: #8b949e; font-weight: normal; }
    .val { font-weight: bold; }
    .green { color: #7ee787; }
    .red { color: #f85149; }
    .amber { color: #d29922; }
    .gray { color: #8b949e; }
    .health-ok { color: #7ee787; }
    .health-err { color: #f85149; }
    .spark { font-family: ui-monospace; letter-spacing: 0.5px; display:inline-block; width: 12ch; min-width: 12ch; white-space: pre; }
    .edge-pos { color: #7ee787; font-weight: bold; }
    .edge-neg { color: #f85149; font-weight: bold; }
    .edge-flat { color: #8b949e; }
    .sig-ready { color: #7ee787; font-weight: bold; }
    .sig-stale { color: #d29922; font-weight: bold; }
    .sig-wait { color: #8b949e; }
    .exp-expired { color: #f85149; font-weight: bold; }
    .exp-live { color: #c9d1d9; }
    .history-box { height: 170px; min-height: 170px; overflow-y: auto; overflow-x: hidden; }
    .fills-box { height: 140px; min-height: 140px; overflow-y: auto; overflow-x: hidden; }
    .activity-cell { width: 18ch; min-width: 18ch; }
    .meta { color: #8b949e; font-size: 11px; margin-top: 8px; }
  </style>
</head>
<body>
  <div class="header-row">
    <span class="brand">◇ ARGUS·VISION</span>
    <span class="sim">◈ SIM</span>
    <span>Up <span id="uptime">—</span></span>
    <span>Bal <span id="balance" class="val green">—</span></span>
    <span>Health <span id="health" class="val">—</span></span>
    <span class="gray">|</span>
    <span>BTC <span id="btc" class="val">—</span> <span id="spark-btc" class="spark"></span></span>
    <span>ETH <span id="eth" class="val">—</span> <span id="spark-eth" class="spark"></span></span>
    <span>SOL <span id="sol" class="val">—</span> <span id="spark-sol" class="spark"></span></span>
    <span class="gray">|</span>
    <span>Mkts <span id="mkt">—</span> Probs <span id="prob">—</span> OB <span id="ob">—</span> Kalshi <span id="rtt">—</span></span>
  </div>

  <div class="section">
    <h2>══ MARKETS</h2>
    <p class="gray">Bot: <span id="promoted-bot" class="amber">default</span></p>
    <table><thead><tr><th>Asset</th><th>Window</th><th>Strike (Ref)</th><th>Fair</th><th>History</th><th>Ask</th><th>Edge</th><th>Expires</th><th>Signal</th></tr></thead><tbody id="markets"></tbody></table>
  </div>

  <div class="section">
    <h2>══ STATS</h2>
    <div id="stats" class="gray">No bot chosen for promoted slot — set promoted_bot_id in config</div>
  </div>

  <div class="section">
    <h2>══ ORDERS</h2>
    <div id="orders">Win Rate — | All-Time — | Fills 0 | Open Orders 0</div>
    <div id="fills-area" class="gray fills-box" style="margin-top:6px;"></div>
  </div>

  <div class="section">
    <h2>══ HISTORY</h2>
    <div id="history" class="gray history-box">No bot chosen for promoted slot</div>
  </div>

  <div class="section">
    <h2>══ DWARF RANKING (TOP 20)</h2>
    <table><thead><tr><th>Bot Name</th><th>Robust</th><th>Net PnL</th><th>E/S PnL</th><th>WR</th><th>E/S Fills</th><th>Trd</th><th class="activity-cell">Activity</th></tr></thead><tbody id="leaderboard"></tbody></table>
  </div>

  <p class="meta">Polling /snapshot every 0.2s · Backend decoupled</p>

  <script>
    var startTs = Date.now() / 1000;
    function uptimeStr() {
      var s = Math.floor((Date.now()/1000) - startTs);
      var m = Math.floor(s/60), h = Math.floor(m/60);
      if (h) return h + "h" + (m%60).toString().padStart(2,"0") + "m";
      return m + "m" + (s%60).toString().padStart(2,"0") + "s";
    }
    function fmtNum(n) { return n == null ? "—" : Number(n).toLocaleString(undefined, { maximumFractionDigits: 0 }); }
    function sparkline(hist, w) {
      if (!hist || hist.length === 0) return "[      ]";
      var chars = "▁▂▃▄▅▆▇█";
      var out = "";
      var slice = hist.slice(-w);
      var lo = Math.min.apply(null, slice), hi = Math.max.apply(null, slice), r = (hi - lo) || 1;
      for (var i = 0; i < slice.length; i++) {
        var idx = Math.min(chars.length - 1, Math.floor((slice[i] - lo) / r * (chars.length - 1)));
        out += chars[idx];
      }
      return "[" + out + "]";
    }
    function bestAskAndEdge(s) {
      var p = s.p_yes != null ? Number(s.p_yes) : 0.5;
      var ya = s.yes_ask != null ? parseInt(s.yes_ask, 10) : 0, na = s.no_ask != null ? parseInt(s.no_ask, 10) : 0;
      var edgeYes = Math.round(p * 100) - ya, edgeNo = Math.round((1 - p) * 100) - na;
      var bestEdge = Math.max(edgeYes, edgeNo);
      var bestAsk = edgeYes >= edgeNo ? ya : na;
      return { ask: bestAsk, edge: bestEdge, side: edgeYes >= edgeNo ? "YES" : "NO" };
    }
    function timeRemaining(expTs) {
      if (!expTs || expTs <= 0) return "EXPRD";
      var rem = Math.floor(expTs - (Date.now()/1000));
      if (rem <= 0) return "EXPRD";
      var h = Math.floor(rem/3600), m = Math.floor((rem%3600)/60), s = rem%60;
      if (h > 0) return h + "h" + m.toString().padStart(2,"0") + "m";
      return m.toString().padStart(2,"0") + ":" + s.toString().padStart(2,"0");
    }
    function strikeDisplay(m) {
      if (!m) return "—";
      if (m.is_range && m.strike_floor != null && m.strike_cap != null) return "$" + fmtNum(m.strike_floor) + "-$" + fmtNum(m.strike_cap);
      if (m.strike_price != null) return "$" + fmtNum(m.strike_price);
      return "—";
    }
    function signalStr(s, prices) {
      var be = bestAskAndEdge(s);
      if (!s.ob_valid && !s.ob_had_valid) return "<span class=\"sig-wait\">WAIT</span>";
      if (!s.ob_valid && s.ob_had_valid) return "<span class=\"sig-stale\">STALE</span>";
      if (be.edge >= 3) return "<span class=\"sig-ready\">READY " + be.side + "</span>";
      if (be.edge > 0) return "<span class=\"sig-wait\">PASS</span>";
      return "<span class=\"sig-wait\">PASS</span>";
    }
    function bestPerType(snap) {
      var states = snap.states || {}, meta = snap.metadata || {}, prices = snap.prices || {};
      var buckets = { BTC: { "15min": [], "60min": [], "Range": [] }, ETH: { "15min": [], "60min": [], "Range": [] }, SOL: { "15min": [], "60min": [] } };
      var nowTs = Date.now() / 1000;
      var horizonS = 3600;
      for (var ticker in states) {
        var s = states[ticker], m = meta[ticker] || {};
        var asset = (s.asset || m.asset || "BTC").toString().toUpperCase();
        var isRange = s.is_range === true || m.is_range === true;
        var wMin = s.window_min != null ? s.window_min : (m.window_minutes != null ? m.window_minutes : 15);
        var wl = isRange ? "Range" : (wMin === 15 ? "15min" : "60min");
        if (!buckets[asset] || !buckets[asset][wl]) continue;
        var expTs = s.exp_ts || 0;
        if (expTs > 0 && expTs <= nowTs) continue;
        var remS = expTs > 0 ? (expTs - nowTs) : Number.POSITIVE_INFINITY;
        if (remS > horizonS) continue;
        var be = bestAskAndEdge(s);
        var validScore = s.ob_valid ? 1 : 0;
        var hadValidScore = s.ob_had_valid ? 1 : 0;
        var freshnessScore = s._last_update_ts || 0;
        var expiryScore = Number.isFinite(remS) ? (horizonS - remS) : 0;
        var score = validScore * 1e12 + hadValidScore * 1e9 + expiryScore * 1e5 + freshnessScore + be.edge;
        buckets[asset][wl].push({ s: s, m: m, score: score });
      }
      var result = { BTC: {}, ETH: {}, SOL: {} };
      for (var asset in buckets) {
        for (var w in buckets[asset]) {
          var arr = buckets[asset][w];
          if (arr.length === 0) { result[asset][w] = null; continue; }
          arr.sort(function(a,b) { return b.score - a.score; });
          result[asset][w] = arr[0].s;
          result[asset][w]._meta = arr[0].m;
        }
      }
      return result;
    }
    function robustScore(stats) {
      if (stats.robust_score != null) return Number(stats.robust_score).toFixed(2);
      return Number(stats.pnl || 0).toFixed(2);
    }
    function render(snap) {
      if (!snap) return;
      if (!startTs || startTs <= 0) startTs = snap.ts || (Date.now()/1000);
      var p = snap.prices || {};
      document.getElementById("btc").textContent = p.BTC != null ? "$" + fmtNum(p.BTC) : "—";
      document.getElementById("eth").textContent = p.ETH != null ? "$" + fmtNum(p.ETH) : "—";
      document.getElementById("sol").textContent = p.SOL != null ? "$" + fmtNum(p.SOL) : "—";
      document.getElementById("spark-btc").textContent = sparkline((snap._spark && snap._spark.BTC) || [], 8);
      document.getElementById("spark-eth").textContent = sparkline((snap._spark && snap._spark.ETH) || [], 8);
      document.getElementById("spark-sol").textContent = sparkline((snap._spark && snap._spark.SOL) || [], 8);
      document.getElementById("uptime").textContent = uptimeStr();
      document.getElementById("health").textContent = snap.ws_connected ? "OK" : "ERR";
      document.getElementById("health").className = snap.ws_connected ? "val health-ok" : "val health-err";
      document.getElementById("balance").textContent = snap.balance_usd != null ? "$" + Number(snap.balance_usd).toFixed(2) : "—";
      document.getElementById("promoted-bot").textContent = snap.primary_bot_id || "default";
      var st = snap.states || {};
      var nOb = 0, nProb = 0;
      for (var k in st) { if (st[k].ob_valid) nOb++; if (st[k].p_yes != null && st[k].p_yes !== 0.5) nProb++; }
      var shownStates = Object.keys(st).length;
      var totalStates = snap.states_total != null ? snap.states_total : shownStates;
      document.getElementById("mkt").textContent = totalStates.toLocaleString() + (shownStates < totalStates ? " (" + shownStates.toLocaleString() + " shown)" : "");
      document.getElementById("prob").textContent = (snap.prob_set_total != null ? snap.prob_set_total : nProb).toLocaleString();
      document.getElementById("ob").textContent = (snap.ob_valid_total != null ? snap.ob_valid_total : nOb).toLocaleString();
      document.getElementById("rtt").textContent = snap.kalshi_rtt_ms != null ? Math.round(snap.kalshi_rtt_ms) + "ms (" + (snap.kalshi_rtt_source || "rest").toUpperCase() + ")" : "—";

      var best = bestPerType(snap);
      var order = [["BTC","15min"],["BTC","60min"],["BTC","Range"],["ETH","15min"],["ETH","60min"],["ETH","Range"],["SOL","15min"],["SOL","60min"]];
      var mktHtml = "";
      for (var i = 0; i < order.length; i++) {
        var asset = order[i][0], w = order[i][1];
        var x = best[asset] && best[asset][w];
        if (!x) {
          mktHtml += "<tr><td>"+asset+"</td><td>"+w+"</td><td class=\"gray\">—</td><td class=\"gray\">---</td><td class=\"gray\">[      ]</td><td class=\"gray\">---</td><td class=\"gray\">---</td><td>--:--</td><td class=\"gray\">EMPTY</td></tr>";
          continue;
        }
        var m = x._meta || {};
        var be = bestAskAndEdge(x);
        var fair = (x.p_yes != null ? (x.p_yes * 100).toFixed(0) + "%" : "---");
        var hist = sparkline(x.p_yes_hist || [], 8);
        var askStr = x.ob_valid ? (be.ask + "¢") : (x.ob_had_valid ? "~" + be.ask + "¢" : "---");
        var edgeRaw = x.ob_valid ? (be.edge >= 0 ? "+" : "") + be.edge + "¢" : (x.ob_had_valid ? "~" + be.edge + "¢" : "---");
        var edgeCls = !x.ob_valid ? "edge-flat" : (be.edge > 0 ? "edge-pos" : (be.edge < 0 ? "edge-neg" : "edge-flat"));
        var edgeStr = "<span class=\"" + edgeCls + "\">" + edgeRaw + "</span>";
        var expStrRaw = timeRemaining(x.exp_ts);
        var expStr = (expStrRaw === "EXPRD") ? "<span class=\"exp-expired\">EXPRD</span>" : "<span class=\"exp-live\">" + expStrRaw + "</span>";
        var sig = signalStr(x, p);
        var strikeRef = strikeDisplay(m);
        mktHtml += "<tr><td>"+asset+"</td><td>"+w+"</td><td>"+strikeRef+"</td><td>"+fair+"</td><td class=\"spark\">"+hist+"</td><td>"+askStr+"</td><td>"+edgeStr+"</td><td>"+expStr+"</td><td>"+sig+"</td></tr>";
      }
      document.getElementById("markets").innerHTML = mktHtml;

      if (snap.primary_bot_id) {
        var sess = snap.session_pnl != null ? Number(snap.session_pnl) : null;
        var alltime = snap.alltime_pnl != null ? Number(snap.alltime_pnl) : null;
        var sessCls = sess == null ? "gray" : (sess >= 0 ? "green" : "red");
        var allCls = alltime == null ? "gray" : (alltime >= 0 ? "green" : "red");
        document.getElementById("stats").innerHTML = "Balance $"+ (snap.balance_usd != null ? Number(snap.balance_usd).toFixed(2) : "—") + " | Session <span class=\"" + sessCls + "\">$"+ (sess != null ? (sess >= 0 ? "+" : "") + sess.toFixed(2) : "—") + "</span> | All-Time <span class=\"" + allCls + "\">$"+ (alltime != null ? (alltime >= 0 ? "+" : "") + alltime.toFixed(2) : "—") + "</span>";
      }
      var total = (snap.wins || 0) + (snap.losses || 0), wr = total > 0 ? Math.floor((snap.wins || 0) * 100 / total) + "%" : "---";
      document.getElementById("orders").innerHTML = "Win Rate " + (snap.wins || 0) + "/" + total + " (" + wr + ") | All-Time $" + (snap.alltime_pnl != null ? (snap.alltime_pnl >= 0 ? "+" : "") + Number(snap.alltime_pnl).toFixed(2) : "—") + " | Fills " + (snap.total_contracts != null ? snap.total_contracts : 0) + " | Open Orders " + (snap.open_orders || 0);
      var fills = snap.recent_fills || [];
      var fillsHtml = "";
      for (var f = 0; f < Math.min(8, fills.length); f++) {
        var fi = fills[f];
        fillsHtml += (fi.side || "").toUpperCase() + " " + (fi.ticker || "").slice(0,12) + " @" + (fi.price || 0) + "¢ x" + (fi.count || 0) + "<br/>";
      }
      document.getElementById("fills-area").innerHTML = fillsHtml || "No fills yet";

      var hist = snap.history || [];
      var histHtml = "";
      for (var h = 0; h < Math.min(10, hist.length); h++) {
        var he = hist[h];
        histHtml += (he.won ? "<span class=\"green\">WIN</span>" : "<span class=\"red\">LOSS</span>") + " " + (he.ticker || "").slice(0,12) + " $" + (he.pnl != null ? (he.pnl >= 0 ? "+" : "") + Number(he.pnl).toFixed(2) : "—") + "<br/>";
      }
      document.getElementById("history").innerHTML = histHtml || "No settled trades yet";

      var bots = snap.bot_stats || {};
      var arr = [];
      for (var bid in bots) {
        var st = bots[bid];
        arr.push({ id: bid, pnl: st.pnl || 0, pnl_e: st.pnl_e || 0, pnl_s: st.pnl_s || 0, wins: st.wins || 0, losses: st.losses || 0, fills: st.fills || 0, fills_e: st.fills_e || 0, fills_s: st.fills_s || 0, trade_count: st.trade_count || (st.wins + st.losses), robust_score: st.robust_score });
      }
      arr.sort(function(a,b) { return (b.robust_score || b.pnl) - (a.robust_score || a.pnl); });
      var lbHtml = "";
      for (var i = 0; i < Math.min(20, arr.length); i++) {
        var b = arr[i];
        var tot = b.wins + b.losses;
        var wr = tot > 0 ? (b.wins / tot * 100).toFixed(1) + "%" : "0.0%";
        var robust = robustScore(b);
        var trail = "················";
        var pos = (i + Math.floor(Date.now()/200) % 16) % 16;
        trail = trail.slice(0, pos) + "●" + trail.slice(pos + 1);
        lbHtml += "<tr><td>"+b.id+"</td><td>"+robust+"</td><td class=\""+(b.pnl>=0?"green":"red")+"\">"+(b.pnl>=0?"+":"")+b.pnl.toFixed(2)+"</td><td><span class=\""+(b.pnl_e>=0?"green":"red")+"\">"+(b.pnl_e>=0?"+":"")+b.pnl_e.toFixed(1)+"</span>/<span class=\""+(b.pnl_s>=0?"green":"red")+"\">"+(b.pnl_s>=0?"+":"")+b.pnl_s.toFixed(1)+"</span></td><td>"+wr+"</td><td>"+b.fills_e+"/"+b.fills_s+"</td><td>"+tot+"</td><td class=\"spark activity-cell\">["+trail+"]</td></tr>";
      }
      document.getElementById("leaderboard").innerHTML = lbHtml || "<tr><td colspan=\"8\" class=\"gray\">No bots</td></tr>";
    }
    function poll() {
      fetch("/snapshot").then(function(r) { return r.json(); }).then(render).catch(function() {});
    }
    setInterval(poll, 200);
    poll();
  </script>
</body>
</html>
"""


class _DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/snapshot":
            snap = get_snapshot()
            body = json.dumps(snap if snap is not None else {}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/" or self.path == "/index.html":
            body = _DASHBOARD_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        log.debug("%s - %s", self.address_string(), format % args)


def run_dashboard_server(host: str = "127.0.0.1", port: int = 9998) -> None:
    # Run HTTP server in the current thread (call from a dedicated thread).
    server = HTTPServer((host, port), _DashboardHandler)
    log.info("Dashboard HTTP server on http://%s:%s", host, port)
    try:
        server.serve_forever()
    except Exception as e:
        log.warning("Dashboard server stopped: %s", e)
    finally:
        server.server_close()


def start_dashboard_thread(host: str = "127.0.0.1", port: int = 9998) -> threading.Thread:
    # Start the dashboard HTTP server in a daemon thread. Returns the thread.
    t = threading.Thread(target=run_dashboard_server, args=(host, port), daemon=True)
    t.start()
    return t
