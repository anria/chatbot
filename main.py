#!/usr/bin/env python3
"""
chatbot — standalone chatbot with Ollama, feedback, and judgements.

Run with:  uv run main.py
Serves on: http://127.0.0.1:8765
uv add python-dotenv
uv add openai
"""

import json
import os
from dotenv import load_dotenv

import threading
import urllib.request
import urllib.error
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HOST           = os.getenv("HOST")
PORT           = int(os.getenv("PORT", 8765))
DEEPSEEK_URL   = os.getenv("DEEPSEEK_HOST")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL")
BEARER_TOKEN   = os.getenv("BEARER_TOKEN")

# use Modal.com's deepseek
client = OpenAI(
    base_url=DEEPSEEK_URL,
    api_key=BEARER_TOKEN,
)
print("DEEPSEEK_URL", DEEPSEEK_URL, "BEARER_TOKEN",BEARER_TOKEN, flush=True)

BASE_DIR  = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "assistant_responses.json"
DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
if not DATA_FILE.exists():
    DATA_FILE.write_text("[]", encoding="utf-8")

_lock = threading.Lock()

SYSTEM_PROMPT = (
    "You are a friendly assistant for the AI Search site. Answer concisely and helpfully. If you don't know, say so. Do not answer anything illegal or which has questionable morals. Answer in a single short paragraph each time."
)
MAX_HISTORY = 4

# ---------------------------------------------------------------------------
# Feedback store
# ---------------------------------------------------------------------------
def _load_all() -> list:
    with _lock:
        try:
            return json.loads(DATA_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError):
            return []

def _save_all(records: list) -> None:
    with _lock:
        DATA_FILE.write_text(
            json.dumps(records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

def feedback_append(prompt: str, answer: str, judgement: int, notes: str = "") -> dict:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "judgement": int(judgement),
        "prompt": prompt,
        "answer": answer,
        "approved": None,
        "notes": notes or "",
    }
    with _lock:
        try:
            records = json.loads(DATA_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError):
            records = []
        records.append(record)
        DATA_FILE.write_text(
            json.dumps(records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return record

def feedback_update(idx: int, approved) -> dict | None:
    records = _load_all()
    if idx < 0 or idx >= len(records):
        return None
    records[idx]["approved"] = approved
    _save_all(records)
    return records[idx]

def feedback_notes(idx: int, notes: str) -> dict | None:
    records = _load_all()
    if idx < 0 or idx >= len(records):
        return None
    records[idx]["notes"] = str(notes)[:4000]
    _save_all(records)
    return records[idx]

# ---------------------------------------------------------------------------
# Ollama streaming generator
# ---------------------------------------------------------------------------

def stream_reply(messages):
    messages = messages[-MAX_HISTORY:]
    all_messages = [{"role": "system", "content": SYSTEM_PROMPT}, *messages]
    try:
        completion = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=all_messages,
            temperature=0.3,
            max_tokens=4096,
            top_p=0.9,
            stream=False,
            extra_body={"reasoning_effort": "high"},
        )
        chunk = completion.choices[0].message.content
        if chunk:
            yield chunk
        if obj.get("done"):
            return
    except Exception as e:
        yield f"\n[Chatbot error: {e}]"


def stream_reply_requests(messages):
    messages = messages[-MAX_HISTORY:]
    ollama_messages = [{"role": "system", "content": SYSTEM_PROMPT}, *messages]
    payload = {
        "model": OLLAMA_MODEL,
        "messages": ollama_messages,
        "stream": True,
        "options": {"temperature": 0.7},
    }
    url = f"{OLLAMA_HOST}/api/chat"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" in obj:
                    yield f"\n[error: {obj['error']}]"
                    return
                chunk = obj.get("message", {}).get("content", "")
                if chunk:
                    yield chunk
                if obj.get("done"):
                    return
    except urllib.error.HTTPError as e:
        if e.code == 404:
            yield (f"\n[Ollama says model '{OLLAMA_MODEL}' is not available. "
                   f"Run: ollama pull {OLLAMA_MODEL}]")
        else:
            yield f"\n[Ollama returned HTTP {e.code}: {e.reason}]"
    except urllib.error.URLError:
        yield (f"\n[Could not reach Ollama at {OLLAMA_HOST}. "
               "Is it running? Start it with: ollama serve]")
    except Exception as e:
        yield f"\n[Chatbot error: {e}]"

# ---------------------------------------------------------------------------
# HTML pages (single-page chatbot + judgements page)
# ---------------------------------------------------------------------------
BASE_CSS = """
:root {
    --bg: #1e293b; --bg-elevated: #334155; --bg-surface: #0f172a;
    --border: #475569; --border-accent: #00d4ff;
    --text: #e2e8f0; --text-muted: #94a3b8;
    --accent: #00d4ff; --accent-hover: #38bdf8;
    --accent-glow: rgba(0, 212, 255, 0.35);
    --up: #00d4ff; --down: #ff6b6b;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
    font-size: 14px; line-height: 1.55;
    background: var(--bg); color: var(--text); min-height: 100vh;
}
a { color: var(--accent); text-decoration: none; }
a:hover { color: var(--accent-hover); }
header.banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border-bottom: 1px solid var(--border-accent);
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    padding: 12px 24px; display: flex; align-items: center;
    justify-content: space-between; flex-wrap: wrap; gap: 12px;
    position: sticky; top: 0; z-index: 100;
}
.banner-title { color: var(--accent); font-size: 20px; font-weight: 700;
    letter-spacing: 0.4px; text-shadow: 0 0 12px var(--accent-glow); }
.banner-link { color: var(--accent); border: 1px solid var(--accent);
    border-radius: 6px; padding: 7px 14px; font-size: 13px; font-weight: 600;
    transition: all 0.15s ease; }
.banner-link:hover { background: var(--accent); color: #0f172a;
    box-shadow: 0 0 14px var(--accent-glow); }
"""

CHAT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AI Search Assistant</title>
<style>
__BASE_CSS__

main { max-width: 900px; margin: 0 auto; padding: 24px; display: flex;
       flex-direction: column; height: calc(100vh - 70px); }

.chat-messages { flex: 1; overflow-y: auto; background: var(--bg-surface);
    border: 1px solid var(--border); border-radius: 12px; padding: 20px;
    display: flex; flex-direction: column; gap: 10px; margin-bottom: 12px; }
.chat-msg { padding: 10px 14px; border-radius: 14px; max-width: 85%;
    word-wrap: break-word; white-space: pre-wrap; line-height: 1.45; }
.chat-bot { background: var(--bg-elevated); color: var(--text);
    border: 1px solid var(--border); border-left: 3px solid var(--accent);
    align-self: flex-start; border-bottom-left-radius: 4px; }
.chat-user { background: var(--accent); color: #0f172a; font-weight: 500;
    align-self: flex-end; border-bottom-right-radius: 4px;
    box-shadow: 0 0 12px rgba(0,212,255,0.25); }
.chat-typing { font-style: italic; color: var(--text-muted); }

.chat-feedback { display: flex; gap: 6px; margin-top: 4px;
    align-self: flex-start; }
.chat-feedback button { background: transparent; border: 1px solid var(--border);
    color: var(--text-muted); border-radius: 6px; padding: 3px 9px;
    font-size: 13px; cursor: pointer; transition: all 0.15s ease;
    font-family: inherit; }
.chat-feedback button:hover { border-color: var(--accent); color: var(--accent); }
.chat-feedback button.selected-up { background: rgba(0,212,255,0.15);
    border-color: var(--accent); color: var(--accent); }
.chat-feedback button.selected-down { background: rgba(255,107,107,0.15);
    border-color: var(--down); color: var(--down); }
.chat-feedback button:disabled { opacity: 0.5; cursor: default; }

.chat-form { display: flex; gap: 8px; background: var(--bg-elevated);
    border: 1px solid var(--border); border-radius: 12px; padding: 10px; }
#chat-input { flex: 1; background: var(--bg-surface);
    border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px;
    color: var(--text); font-size: 14px; font-family: inherit; outline: none;
    transition: border-color 0.15s ease, box-shadow 0.15s ease; }
#chat-input:focus { border-color: var(--accent);
    box-shadow: 0 0 0 3px rgba(0,212,255,0.15); }
#chat-send { background: var(--accent); color: #0f172a; border: none;
    border-radius: 8px; padding: 10px 20px; cursor: pointer; font-size: 14px;
    font-weight: 600; transition: background 0.15s ease, box-shadow 0.15s ease; }
#chat-send:hover { background: var(--accent-hover);
    box-shadow: 0 0 14px var(--accent-glow); }
#chat-send:disabled { background: var(--border); color: var(--text-muted);
    cursor: not-allowed; box-shadow: none; }

.banner-left {
    display: flex;
    align-items: center;
    gap: 12px;
}
.banner-left svg { 
    width: 38px;
    height: 38px;
    filter: drop-shadow(0 0 6px var(--accent-glow));
}

.banner-right {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}
</style>
</head>
<body>
<header class="banner">
    <div class="banner-left">
        <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
            <circle cx="50" cy="50" r="45" fill="#0f172a" stroke="#00d4ff" stroke-width="3"/>
            <text x="50" y="68" font-size="46" text-anchor="middle"
                  fill="#00d4ff" font-family="-apple-system, Arial, sans-serif" font-weight="700">AI</text>
            <circle cx="30" cy="40" r="7" fill="#00d4ff"/>
            <circle cx="70" cy="40" r="7" fill="#00d4ff"/>
            <path d="M35 65 Q50 74 65 65" stroke="#00d4ff" stroke-width="3" fill="none"/>
        </svg>    
        <span class="banner-title">AI Search Assistant</span>
    </div>
    <div class="banner-right">
        <a href="/judgements" class="banner-link" target="blank">View judgements</a>
    </div>
</header>
<main>
    <div class="chat-messages" id="chat-messages">
        <div class="chat-msg chat-bot">Hi! Ask me anything.</div>
    </div>
    <form class="chat-form" id="chat-form" autocomplete="off">
        <input id="chat-input" type="text" placeholder="Type a message…" maxlength="2000" required>
        <button type="submit" id="chat-send">Send</button>
    </form>
</main>
<script>
(function () {
    "use strict";
    const CFG = { chatEndpoint: "/api/chat", feedbackEndpoint: "/api/feedback",
                  timeoutMs: 180000 };
    const form     = document.getElementById("chat-form");
    const input    = document.getElementById("chat-input");
    const sendBtn  = document.getElementById("chat-send");
    const messages = document.getElementById("chat-messages");
    const history  = [];

    function addMessage(text, role) {
        const div = document.createElement("div");
        div.className = "chat-msg " + (role === "user" ? "chat-user" : "chat-bot");
        div.textContent = text;
        messages.appendChild(div);
        messages.scrollTop = messages.scrollHeight;
        return div;
    }
    function addStreamingBotBubble() {
        const div = document.createElement("div");
        div.className = "chat-msg chat-bot chat-typing";
        div.textContent = "…";
        messages.appendChild(div);
        messages.scrollTop = messages.scrollHeight;
        return div;
    }

    function finalizeBotMessage(bubble, reply, userPrompt) {
        bubble.classList.remove("chat-typing");
        if (!reply) return;
        history.push({ role: "assistant", content: reply });

        const row = document.createElement("div");
        row.className = "chat-feedback";
        const upBtn = document.createElement("button");
        upBtn.textContent = "👍";
        upBtn.title = "Helpful";
        const downBtn = document.createElement("button");
        downBtn.textContent = "👎";
        downBtn.title = "Not helpful";

        async function sendJudgement(judgement, chosenBtn, otherBtn) {
            upBtn.disabled = true; downBtn.disabled = true;
            chosenBtn.classList.add(judgement === 1 ? "selected-up" : "selected-down");
            otherBtn.classList.remove("selected-up", "selected-down");
            const note = window.prompt("Optional note (leave blank to skip):", "") || "";
            try {
                await fetch(CFG.feedbackEndpoint, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ prompt: userPrompt, answer: reply,
                                           judgement: judgement, notes: note }),
                });
            } catch (err) { console.warn("feedback failed:", err); }
        }
        upBtn.addEventListener("click",   () => sendJudgement(1, upBtn, downBtn));
        downBtn.addEventListener("click", () => sendJudgement(0, downBtn, upBtn));
        row.appendChild(upBtn); row.appendChild(downBtn);
        messages.appendChild(row);
        messages.scrollTop = messages.scrollHeight;
    }

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        const text = input.value.trim();
        if (!text) return;
        addMessage(text, "user");
        history.push({ role: "user", content: text });
        input.value = ""; input.focus();
        sendBtn.disabled = true; input.disabled = true;

        const bubble = addStreamingBotBubble();
        let replySoFar = "", gotAnyChunk = false, finished = false;

        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), CFG.timeoutMs);

        function handleFrame(rawEvent) {
            const dataLines = rawEvent.split(/\\r?\\n/)
                .filter(l => l.startsWith("data:"))
                .map(l => l.slice(5).trimStart());
            if (dataLines.length === 0) return;
            let evt;
            try { evt = JSON.parse(dataLines.join("\\n")); } catch (_) { return; }
            if (evt.done) {
                if (!finished) { finished = true; finalizeBotMessage(bubble, replySoFar, text); }
                return;
            }
            if (evt.error) {
                bubble.classList.remove("chat-typing");
                bubble.textContent = "Error: " + evt.error;
                finished = true; return;
            }
            if (typeof evt.delta === "string" && evt.delta.length) {
                if (!gotAnyChunk) { bubble.classList.remove("chat-typing");
                    bubble.textContent = ""; gotAnyChunk = true; }
                replySoFar += evt.delta;
                bubble.textContent = replySoFar;
                messages.scrollTop = messages.scrollHeight;
            }
        }

        try {
            const resp = await fetch(CFG.chatEndpoint, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ messages: history }),
                signal: controller.signal,
            });
            if (!resp.ok || !resp.body) {
                let errMsg = resp.statusText;
                try { const j = await resp.json(); errMsg = j.error || errMsg; } catch (_) {}
                bubble.classList.remove("chat-typing");
                bubble.textContent = "Sorry, something went wrong: " + errMsg;
                return;
            }
            const reader  = resp.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let buffer = "";
            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true }).replace(/\\r\\n/g, "\\n");
                let idx;
                while ((idx = buffer.indexOf("\\n\\n")) !== -1) {
                    const rawEvent = buffer.slice(0, idx);
                    buffer = buffer.slice(idx + 2);
                    handleFrame(rawEvent);
                }
            }
            if (buffer.trim()) handleFrame(buffer);
            if (!finished) { finished = true; finalizeBotMessage(bubble, replySoFar, text); }
        } catch (err) {
            bubble.classList.remove("chat-typing");
            if (err.name === "AbortError") {
                const secs = Math.round(CFG.timeoutMs / 1000);
                bubble.textContent = replySoFar
                    ? replySoFar + `\\n\\n[stopped after ${secs}s]`
                    : `The model took longer than ${secs}s. Try again or use a shorter question.`;
            } else {
                bubble.textContent = "Network error — please try again.";
            }
        } finally {
            clearTimeout(timer);
            sendBtn.disabled = false; input.disabled = false; input.focus();
        }
    });
})();
</script>
</body>
</html>
"""

JUDGEMENTS_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Assistant Judgements — AI Search</title>
<style>
__BASE_CSS__

main { max-width: 1100px; margin: 0 auto; padding: 28px 24px; }
.stats { color: var(--text-muted); font-size: 13px; }
.stats strong { color: var(--accent); }
.filters { display: flex; gap: 8px; margin-bottom: 20px; flex-wrap: wrap; }
.filters button { background: transparent; color: var(--text-muted);
    border: 1px solid var(--border); border-radius: 20px; padding: 6px 16px;
    font-size: 13px; cursor: pointer; transition: all 0.15s ease;
    font-family: inherit; }
.filters button:hover { border-color: var(--accent); color: var(--accent); }
.filters button.active { background: var(--accent); color: #0f172a;
    border-color: var(--accent); font-weight: 600; }
.record { background: var(--bg-elevated); border: 1px solid var(--border);
    border-left: 3px solid var(--accent); border-radius: 10px;
    padding: 18px 22px; margin-bottom: 16px; transition: all 0.18s ease; }
.record.rejected { opacity: 0.55; filter: grayscale(0.5); border-left-color: var(--border); }
.record.approved { border-left-color: var(--accent);
    box-shadow: 0 0 18px rgba(0,212,255,0.12); }
.record-header { display: flex; justify-content: space-between;
    align-items: center; gap: 12px; margin-bottom: 12px; flex-wrap: wrap; }
.badges { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.badge { font-size: 11px; padding: 3px 10px; border-radius: 12px;
    font-weight: 600; letter-spacing: 0.3px; text-transform: uppercase; }
.badge-up { background: rgba(0,212,255,0.15); color: var(--up);
    border: 1px solid var(--up); }
.badge-down { background: rgba(255,107,107,0.15); color: var(--down);
    border: 1px solid var(--down); }
.badge-pending { background: transparent; color: var(--text-muted);
    border: 1px solid var(--border); }
.badge-approved { background: rgba(0,212,255,0.2); color: var(--accent);
    border: 1px solid var(--accent); }
.badge-rejected { background: rgba(148,163,184,0.15); color: var(--text-muted);
    border: 1px solid var(--border); }
.timestamp { color: var(--text-muted); font-size: 12px; }
.index-tag { color: var(--text-muted); font-size: 12px; font-family: ui-monospace, monospace; }
.field { margin-bottom: 12px; }
.field-label { color: var(--accent); font-size: 11px; text-transform: uppercase;
    letter-spacing: 0.6px; font-weight: 700; margin-bottom: 4px; }
.field-value { color: var(--text); white-space: pre-wrap;
    word-break: break-word; font-size: 14px; }
.notes-input { width: 100%; background: var(--bg-surface);
    border: 1px solid var(--border); border-radius: 6px; padding: 8px 10px;
    color: var(--text); font-family: inherit; font-size: 13px; outline: none;
    resize: vertical; min-height: 52px; transition: border-color 0.15s ease,
    box-shadow 0.15s ease; }
.notes-input:focus { border-color: var(--accent);
    box-shadow: 0 0 0 3px rgba(0,212,255,0.15); }
.notes-actions { display: flex; justify-content: flex-end; margin-top: 6px; }
.notes-actions button { background: transparent; color: var(--accent);
    border: 1px solid var(--accent); border-radius: 6px; padding: 4px 14px;
    font-size: 12px; font-weight: 600; cursor: pointer; font-family: inherit;
    transition: all 0.15s ease; }
.notes-actions button:hover { background: var(--accent); color: #0f172a; }
.notes-actions button:disabled { opacity: 0.5; cursor: default; }
.notes-saved { color: var(--accent); font-size: 12px; margin-right: auto; }
.actions { display: flex; gap: 8px; margin-top: 14px; }
.actions button { padding: 7px 18px; border-radius: 22px; font-size: 13px;
    font-weight: 600; cursor: pointer; font-family: inherit;
    transition: all 0.15s ease; border: 1.5px solid; background: transparent; }
.btn-approve { border-color: var(--accent); color: var(--accent); }
.btn-approve:hover { background: var(--accent); color: #0f172a;
    box-shadow: 0 0 14px var(--accent-glow); }
.btn-reject { border-color: var(--text-muted); color: var(--text-muted); }
.btn-reject:hover { background: var(--text-muted); color: var(--bg); }
.btn-approve.current, .btn-reject.current { background: var(--accent); color: #0f172a; }
.btn-reject.current { background: var(--text-muted); color: var(--bg); }
.empty { text-align: center; padding: 60px 20px; color: var(--text-muted); }
.banner-left {
    display: flex;
    align-items: center;
    gap: 12px;
}
.banner-left svg { 
    width: 38px;
    height: 38px;
    filter: drop-shadow(0 0 6px var(--accent-glow));
}

.banner-right {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}
</style>
</head>
<body>
<header class="banner">
    <div class="banner-left">
        <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
            <circle cx="50" cy="50" r="45" fill="#0f172a" stroke="#00d4ff" stroke-width="3"/>
            <text x="50" y="68" font-size="46" text-anchor="middle"
                  fill="#00d4ff" font-family="-apple-system, Arial, sans-serif" font-weight="700">AI</text>
            <circle cx="30" cy="40" r="7" fill="#00d4ff"/>
            <circle cx="70" cy="40" r="7" fill="#00d4ff"/>
            <path d="M35 65 Q50 74 65 65" stroke="#00d4ff" stroke-width="3" fill="none"/>
        </svg>    
        <span class="banner-title">Assistant Judgements</span>
    </div>
    <span class="stats" id="stats">Loading…</span>
</header>
<main>
    <div class="filters" id="filters">
        <button data-filter="all" class="active">All</button>
        <button data-filter="pending">Pending</button>
        <button data-filter="approved">Approved</button>
        <button data-filter="rejected">Rejected</button>
        <button data-filter="up">👍 Only</button>
        <button data-filter="down">👎 Only</button>
    </div>
    <div id="records"></div>
</main>
<script>
(function () {
    "use strict";
    const recordsEl = document.getElementById("records");
    const statsEl   = document.getElementById("stats");
    const filtersEl = document.getElementById("filters");
    let allItems = [], activeFilter = "all";

    function fmtTime(iso) { try { return new Date(iso).toLocaleString(); }
                            catch { return iso; } }
    function statusOf(item) {
        if (item.approved === true) return "approved";
        if (item.approved === false) return "rejected";
        return "pending";
    }
    function passesFilter(item, filter) {
        const status = statusOf(item);
        if (filter === "all") return true;
        if (filter === "pending") return status === "pending";
        if (filter === "approved") return status === "approved";
        if (filter === "rejected") return status === "rejected";
        if (filter === "up") return item.judgement === 1;
        if (filter === "down") return item.judgement === 0;
        return true;
    }
    function escapeHtml(s) {
        const div = document.createElement("div");
        div.textContent = s == null ? "" : String(s);
        return div.innerHTML;
    }

    function render() {
        recordsEl.innerHTML = "";
        const visible = allItems.map((item, idx) => ({ item, idx }))
            .filter(({ item }) => passesFilter(item, activeFilter));
        if (visible.length === 0) {
            recordsEl.innerHTML = '<div class="empty">No records match this filter.</div>';
        }
        for (const { item, idx } of visible) recordsEl.appendChild(renderRecord(item, idx));
        const total = allItems.length;
        const approved = allItems.filter(i => i.approved === true).length;
        const rejected = allItems.filter(i => i.approved === false).length;
        statsEl.innerHTML = `${total} total · <strong>${approved}</strong> approved · ` +
                            `${rejected} rejected · ${total - approved - rejected} pending`;
    }

    function renderRecord(item, idx) {
        const status = statusOf(item);
        const div = document.createElement("div");
        div.className = "record " + status;
        const judgementBadge = item.judgement === 1
            ? '<span class="badge badge-up">👍 Helpful</span>'
            : '<span class="badge badge-down">👎 Not helpful</span>';
        const statusBadge = status === "approved"
            ? '<span class="badge badge-approved">Approved</span>'
            : status === "rejected"
            ? '<span class="badge badge-rejected">Rejected</span>'
            : '<span class="badge badge-pending">Pending</span>';

        div.innerHTML = `
            <div class="record-header">
                <div class="badges">${judgementBadge}${statusBadge}
                    <span class="index-tag">#${idx}</span></div>
                <span class="timestamp">${fmtTime(item.timestamp)}</span>
            </div>
            <div class="field"><div class="field-label">Prompt</div>
                <div class="field-value">${escapeHtml(item.prompt)}</div></div>
            <div class="field"><div class="field-label">Answer</div>
                <div class="field-value">${escapeHtml(item.answer)}</div></div>
            <div class="field"><div class="field-label">Notes</div>
                <textarea class="notes-input" data-idx="${idx}"
                    placeholder="Add a note…" maxlength="4000">${escapeHtml(item.notes || "")}</textarea>
                <div class="notes-actions"><span class="notes-saved" style="display:none;">Saved ✓</span>
                    <button type="button" data-idx="${idx}" class="btn-save-notes" disabled>Save notes</button></div>
            </div>
            <div class="actions">
                <button class="btn-approve ${item.approved === true ? "current" : ""}"
                    data-idx="${idx}" data-approved="true">Approve</button>
                <button class="btn-reject ${item.approved === false ? "current" : ""}"
                    data-idx="${idx}" data-approved="false">Reject</button>
            </div>`;

        div.querySelectorAll(".actions button").forEach(btn => {
            btn.addEventListener("click", () => updateJudgement(
                parseInt(btn.dataset.idx, 10), btn.dataset.approved === "true"));
        });

        const textarea = div.querySelector(".notes-input");
        const saveBtn  = div.querySelector(".btn-save-notes");
        const savedMsg = div.querySelector(".notes-saved");
        textarea.addEventListener("input", () => {
            saveBtn.disabled = textarea.value === (item.notes || "");
        });
        saveBtn.addEventListener("click", async () => {
            saveBtn.disabled = true;
            try {
                const resp = await fetch(`/api/judgements/${idx}/notes`, {
                    method: "POST", headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ notes: textarea.value }),
                });
                const data = await resp.json();
                allItems[idx] = data.record; item.notes = data.record.notes;
                savedMsg.style.display = "inline";
                setTimeout(() => { savedMsg.style.display = "none"; }, 1500);
            } catch (err) { saveBtn.disabled = false; }
        });
        return div;
    }

    async function load() {
        try {
            const resp = await fetch("/api/judgements");
            const data = await resp.json();
            allItems = data.items || []; render();
        } catch { recordsEl.innerHTML = '<div class="empty">Could not load judgements.</div>'; }
    }
    async function updateJudgement(idx, approved) {
        try {
            const resp = await fetch(`/api/judgements/${idx}`, {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ approved }),
            });
            const data = await resp.json();
            allItems[idx] = data.record; render();
        } catch (err) { console.warn("update failed", err); }
    }

    filtersEl.addEventListener("click", (e) => {
        const btn = e.target.closest("button[data-filter]");
        if (!btn) return;
        filtersEl.querySelectorAll("button").forEach(b => b.classList.remove("active"));
        btn.classList.add("active"); activeFilter = btn.dataset.filter; render();
    });
    load();
})();
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[chatbot] {self.address_string()} - {fmt % args}")

    # ---------- helpers ----------
    def _send_html(self, html: str, status: int = 200):
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj, status: int = 200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    def _path(self):
        return urlparse(self.path).path.rstrip("/") or "/"

    # ---------- GET ----------
    def do_GET(self):
        path = self._path()
        if path == "/":
            self._send_html(CHAT_HTML.replace("__BASE_CSS__", BASE_CSS))
        elif path == "/judgements":
            self._send_html(JUDGEMENTS_HTML.replace("__BASE_CSS__", BASE_CSS))
        elif path == "/api/judgements":
            self._send_json({"items": _load_all()})
        else:
            self._send_html("<h1>404 Not Found</h1>", 404)

    # ---------- POST ----------
    def do_POST(self):
        path = self._path()
        body = self._read_json_body()
        if body is None:
            self._send_json({"error": "Invalid JSON"}, 400)
            return

        if path == "/api/chat":
            self._handle_chat(body)
        elif path == "/api/feedback":
            self._handle_feedback(body)
        elif path.startswith("/api/judgements/") and path.endswith("/notes"):
            idx_str = path[len("/api/judgements/"):-len("/notes")]
            self._handle_notes(idx_str, body)
        elif path.startswith("/api/judgements/"):
            idx_str = path[len("/api/judgements/"):]
            self._handle_judgement(idx_str, body)
        else:
            self._send_json({"error": "Not found"}, 404)

    # ---------- chat streaming ----------
    def _handle_chat(self, body):
        messages = body.get("messages", [])
        if not isinstance(messages, list) or not messages:
            self._send_json({"error": "messages must be a non-empty list"}, 400)
            return

        clean = []
        for m in messages[-MAX_HISTORY:]:
            if (isinstance(m, dict)
                    and m.get("role") in ("user", "assistant")
                    and isinstance(m.get("content"), str)):
                clean.append({"role": m["role"], "content": m["content"][:4000]})
        if not clean:
            self._send_json({"error": "No valid messages"}, 400)
            return

        # SSE headers — no Content-Length, connection stays open
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        try:
            for chunk in stream_reply(clean):
                payload = json.dumps({"delta": chunk})
                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # Client disconnected mid-stream — stop cleanly
            return
        except Exception as e:
            try:
                err = json.dumps({"error": str(e)})
                self.wfile.write(f"data: {err}\n\n".encode("utf-8"))
                self.wfile.flush()
            except Exception:
                pass
        finally:
            try:
                self.wfile.write(b'data: {"done": true}\n\n')
                self.wfile.flush()
            except Exception:
                pass

    # ---------- feedback ----------
    def _handle_feedback(self, body):
        prompt    = str(body.get("prompt", ""))[:4000]
        answer    = str(body.get("answer", ""))[:20000]
        judgement = body.get("judgement")
        notes     = str(body.get("notes", ""))[:4000]

        if judgement not in (0, 1):
            self._send_json({"error": "judgement must be 0 or 1"}, 400)
            return
        if not prompt or not answer:
            self._send_json({"error": "prompt and answer are required"}, 400)
            return

        try:
            record = feedback_append(prompt, answer, judgement, notes)
        except Exception as e:
            self._send_json({"error": f"Could not save feedback: {e}"}, 500)
            return
        self._send_json({"ok": True, "record": record})

    # ---------- judgements: approve/reject ----------
    def _handle_judgement(self, idx_str: str, body):
        try:
            idx = int(idx_str)
        except ValueError:
            self._send_json({"error": "invalid index"}, 400)
            return
        approved = body.get("approved")
        if approved not in (True, False):
            self._send_json({"error": "approved must be true or false"}, 400)
            return
        record = feedback_update(idx, approved)
        if record is None:
            self._send_json({"error": "index out of range"}, 404)
            return
        self._send_json({"ok": True, "record": record})

    # ---------- judgements: notes ----------
    def _handle_notes(self, idx_str: str, body):
        try:
            idx = int(idx_str)
        except ValueError:
            self._send_json({"error": "invalid index"}, 400)
            return
        notes = str(body.get("notes", ""))[:4000]
        record = feedback_notes(idx, notes)
        if record is None:
            self._send_json({"error": "index out of range"}, 404)
            return
        self._send_json({"ok": True, "record": record})


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"[chatbot] Serving on http://{HOST}:{PORT}")
    print(f"[chatbot] Data file: {DATA_FILE}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[chatbot] Shutting down…")
        server.shutdown()


if __name__ == "__main__":
    main()
