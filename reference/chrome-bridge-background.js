// Chrome Bridge service worker (MV3).
// Connects OUT to the local WebSocket server hosted by the Chrome Bridge MCP
// server (server/server.js). The extension cannot listen on a port, so the MCP
// server owns the socket and the extension dials it.
//
// Protocol: JSON messages over WS.
//   Client -> SW: { id, action, params }
//   SW -> Client: { id, ok, result } | { id, ok:false, error }
//   SW -> Client: { event: "status", connected: true, tabCount, activeUrl }
//
// Actions:
//   list_tabs          -> [{ id, title, url, active }]
//   get_active_tab     -> { id, title, url }
//   read_page          -> { url, title, text }  (visible text of active tab)
//   navigate           -> { url }  (open in active tab, or new tab if none)
//   get_text_by_selector -> { text } of matched selector(s) in active tab
//   click_selector     -> click first element matching CSS selector
//   type_into          -> set value + dispatch input/change on a selector
//   scroll             -> { direction, amount }
//   exec_js            -> run arbitrary JS in active tab (returns JSON-ish)

let ws = null;
let reconnectTimer = null;
let reconnectDelay = 1500;

const WS_URL = "ws://127.0.0.1:8765";
const RECONNECT_MAX_DELAY = 10000;

function setBadge(ok, text) {
  const color = ok ? "#0f9d58" : "#d93025";
  chrome.action.setBadgeBackgroundColor({ color });
  chrome.action.setBadgeText({ text });
}

function connect() {
  clearTimeout(reconnectTimer);
  try {
    ws = new WebSocket(WS_URL);
  } catch (e) {
    setBadge(false, "!");
    scheduleReconnect();
    return;
  }
  ws.onopen = () => {
    reconnectDelay = 1500;
    setBadge(true, "ON");
    sendStatus();
  };
  ws.onmessage = (ev) => {
    handleMessage(ev.data).catch((err) => {
      console.error("[chrome-bridge] handler error:", err);
    });
  };
  ws.onclose = () => {
    ws = null;
    setBadge(false, "OFF");
    scheduleReconnect();
  };
  ws.onerror = () => {
    try { ws.close(); } catch (_) {}
  };
}

function scheduleReconnect() {
  clearTimeout(reconnectTimer);
  reconnectTimer = setTimeout(connect, reconnectDelay);
  reconnectDelay = Math.min(reconnectDelay * 1.8, RECONNECT_MAX_DELAY);
}

function send(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(obj));
  }
}

async function sendStatus() {
  try {
    const tabs = await chrome.tabs.query({});
    const active = tabs.find((t) => t.active);
    send({
      event: "status",
      connected: true,
      tabCount: tabs.length,
      activeUrl: active && active.url ? active.url : null,
    });
  } catch (_) {
    send({ event: "status", connected: true, tabCount: 0, activeUrl: null });
  }
}

async function getActiveTab() {
  const [tab] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
  return tab || null;
}

async function injectInto(tabId) {
  // Ensures the content-script "bridge.js" is injected into the given tab,
  // then returns a handle to call it. Returns null if the tab is restricted
  // (chrome:// pages, Web Store, etc.).
  try {
    await chrome.scripting.executeScript({
      target: { tabId },
      files: ["content.js"],
    });
    return true;
  } catch (e) {
    return { error: String(e.message || e) };
  }
}

async function callContent(tabId, fnName, args) {
  const inject = await injectInto(tabId);
  if (inject && inject.error) return { error: inject.error };
  const results = await chrome.tabs.sendMessage(tabId, { fn: fnName, args: args || [] });
  return results;
}

async function handleMessage(raw) {
  let msg;
  try {
    msg = JSON.parse(raw);
  } catch (_) {
    return;
  }
  if (!msg || !msg.id || !msg.action) return;

  const respond = (payload) => send({ id: msg.id, ...payload });

  try {
    switch (msg.action) {
      case "list_tabs": {
        const tabs = await chrome.tabs.query({});
        respond({
          ok: true,
          result: tabs.map((t) => ({
            id: t.id,
            title: t.title || "",
            url: t.url || "",
            active: !!t.active,
          })),
        });
        break;
      }

      case "get_active_tab": {
        const t = await getActiveTab();
        if (!t) return respond({ ok: false, error: "No active tab found." });
        respond({ ok: true, result: { id: t.id, title: t.title || "", url: t.url || "" } });
        break;
      }

      case "read_page": {
        const tab = await getActiveTab();
        if (!tab) return respond({ ok: false, error: "No active tab found." });
        if (!tab.url || tab.url.startsWith("chrome://") || tab.url.startsWith("chrome-extension://") || tab.url.startsWith("edge://")) {
          return respond({ ok: false, error: `Cannot read restricted page: ${tab.url}` });
        }
        const r = await callContent(tab.id, "getVisibleText", []);
        if (r && r.error) return respond({ ok: false, error: r.error });
        respond({
          ok: true,
          result: { url: tab.url, title: tab.title || "", text: r || "" },
        });
        break;
      }

      case "get_text_by_selector": {
        const tab = await getActiveTab();
        if (!tab) return respond({ ok: false, error: "No active tab found." });
        const r = await callContent(tab.id, "getTextBySelector", [msg.params.selector]);
        if (r && r.error) return respond({ ok: false, error: r.error });
        respond({ ok: true, result: { text: r || "" } });
        break;
      }

      case "navigate": {
        const url = msg.params && msg.params.url;
        if (!url) return respond({ ok: false, error: "navigate requires params.url" });
        let tab = await getActiveTab();
        if (tab && tab.url && !(tab.url.startsWith("chrome://") || tab.url.startsWith("chrome-extension://"))) {
          await chrome.tabs.update(tab.id, { url });
        } else {
          tab = await chrome.tabs.create({ url });
        }
        respond({ ok: true, result: { tabId: tab.id, url } });
        break;
      }

      case "click_selector": {
        const sel = msg.params && msg.params.selector;
        if (!sel) return respond({ ok: false, error: "click_selector requires params.selector" });
        const tab = await getActiveTab();
        if (!tab) return respond({ ok: false, error: "No active tab found." });
        const r = await callContent(tab.id, "clickSelector", [sel]);
        if (r && r.error) return respond({ ok: false, error: r.error });
        respond({ ok: true, result: { clicked: !!r, selector: sel } });
        break;
      }

      case "type_into": {
        const sel = msg.params && msg.params.selector;
        const value = msg.params && msg.params.value;
        if (!sel || value === undefined) return respond({ ok: false, error: "type_into requires params.selector and params.value" });
        const tab = await getActiveTab();
        if (!tab) return respond({ ok: false, error: "No active tab found." });
        const r = await callContent(tab.id, "typeInto", [sel, value]);
        if (r && r.error) return respond({ ok: false, error: r.error });
        respond({ ok: true, result: { typed: !!r } });
        break;
      }

      case "scroll": {
        const direction = (msg.params && msg.params.direction) || "down";
        const amount = msg.params && msg.params.amount ? Number(msg.params.amount) : 600;
        const tab = await getActiveTab();
        if (!tab) return respond({ ok: false, error: "No active tab found." });
        const r = await callContent(tab.id, "scrollPage", [direction, amount]);
        if (r && r.error) return respond({ ok: false, error: r.error });
        respond({ ok: true, result: { direction, amount } });
        break;
      }

      case "exec_js": {
        const code = msg.params && msg.params.code;
        if (!code) return respond({ ok: false, error: "exec_js requires params.code" });
        const tab = await getActiveTab();
        if (!tab) return respond({ ok: false, error: "No active tab found." });
        const results = await chrome.scripting.executeScript({
          target: { tabId: tab.id },
          func: (src) => {
            // eslint-disable-next-line no-new-func
            const fn = new Function("return (" + src + ");");
            const out = fn();
            return out === undefined ? null : (typeof out === "object" ? JSON.stringify(out) : String(out));
          },
          args: [code],
        });
        respond({ ok: true, result: results[0] ? results[0].result : null });
        break;
      }

      case "ping": {
        respond({ ok: true, result: "pong" });
        break;
      }

      default:
        respond({ ok: false, error: `Unknown action: ${msg.action}` });
    }
  } catch (e) {
    respond({ ok: false, error: String((e && e.message) || e) });
  }
}

// Answer the popup's "are you connected?" ping, and broadcast status changes.
chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg && msg.ping) {
    sendResponse({ connected: !!(ws && ws.readyState === WebSocket.OPEN) });
  }
});

// Stay alive: MV3 service workers idle out after 30s. Reconnect on startup.
chrome.runtime.onStartup.addListener(connect);
chrome.runtime.onInstalled.addListener(() => {
  setBadge(false, "…");
  connect();
});

// Keepalive: an open WebSocket alone does NOT hold an MV3 service worker
// alive. Use a 30s chrome.alarms tick (minimum supported period) to wake the
// SW, so the bridge connection survives idle and redials promptly after the
// MCP server restarts. This is the standard MV3 WS-keepalive pattern.
chrome.alarms.create("bridge-keepalive", { periodInMinutes: 0.5 });
chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name !== "bridge-keepalive") return;
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    connect();
  } else {
    sendStatus();
  }
});

chrome.tabs.onActivated.addListener(() => {
  if (ws && ws.readyState === WebSocket.OPEN) sendStatus();
});
chrome.tabs.onUpdated.addListener(() => {
  if (ws && ws.readyState === WebSocket.OPEN) sendStatus();
});
connect();
setBadge(false, "…");
