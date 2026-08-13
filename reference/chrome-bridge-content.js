// Chrome Bridge content script — injected on demand via chrome.scripting.
// Exposes small DOM helpers callable through chrome.runtime messages.

function getVisibleText() {
  const b = document.body;
  if (!b) return "";
  const clone = b.cloneNode(true);
  clone.querySelectorAll("script,style,noscript,svg,canvas,video,iframe").forEach((n) => n.remove());
  const text = (clone.innerText || clone.textContent || "").replace(/\n{3,}/g, "\n\n").trim();
  return text.length > 200000 ? text.slice(0, 200000) : text;
}

function getTextBySelector(selector) {
  const nodes = document.querySelectorAll(selector);
  if (!nodes.length) return "";
  const parts = [];
  nodes.forEach((n) => {
    const t = (n.innerText || n.textContent || "").trim();
    if (t) parts.push(t);
  });
  return parts.join("\n---\n").slice(0, 100000);
}

function clickSelector(selector) {
  const el = document.querySelector(selector);
  if (!el) return false;
  el.scrollIntoView({ block: "center" });
  el.click();
  return true;
}

function typeInto(selector, value) {
  const el = document.querySelector(selector);
  if (!el) return false;
  el.focus();
  // React-controlled inputs need the NATIVE value setter from the prototype,
  // not the element's own (possibly patched) property.
  try {
    const proto =
      el instanceof HTMLTextAreaElement
        ? HTMLTextAreaElement.prototype
        : el instanceof HTMLInputElement
          ? HTMLInputElement.prototype
          : null;
    if (proto) {
      const setter = Object.getOwnPropertyDescriptor(proto, "value").set;
      setter.call(el, value);
    } else {
      el.value = value;
    }
  } catch (_) {
    try {
      el.value = value;
    } catch (__) {
      // contenteditable: fall back to textContent (last resort)
      el.textContent = value;
    }
  }
  el.dispatchEvent(new InputEvent("input", { bubbles: true, inputType: "insertText", data: value }));
  el.dispatchEvent(new Event("change", { bubbles: true }));
  return true;
}

function scrollPage(direction, amount) {
  const before = window.scrollY;
  if (direction === "up") window.scrollBy(0, -amount);
  else if (direction === "down") window.scrollBy(0, amount);
  else if (direction === "top") window.scrollTo(0, 0);
  else if (direction === "bottom") window.scrollTo(0, document.body.scrollHeight);
  else return { error: `Unknown direction: ${direction}` };
  return { before, after: window.scrollY };
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (!msg || !msg.fn) return;
  let out;
  try {
    switch (msg.fn) {
      case "getVisibleText": out = getVisibleText(); break;
      case "getTextBySelector": out = getTextBySelector(msg.args && msg.args[0]); break;
      case "clickSelector": out = clickSelector(msg.args && msg.args[0]); break;
      case "typeInto": out = typeInto(msg.args && msg.args[0], msg.args && msg.args[1]); break;
      case "scrollPage": out = scrollPage(msg.args && msg.args[0], msg.args && msg.args[1]); break;
      default: out = { error: `Unknown content fn: ${msg.fn}` };
    }
  } catch (e) {
    out = { error: String((e && e.message) || e) };
  }
  sendResponse(out);
  return false;
});
