"""
BitWhisker R1 Tkinter – ChatGPT-style GUI  600 × 400
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Pure Python • No API • No external packages
Buttons: black labels on light surfaces (ChatGPT style)
Generation runs in background thread — UI never freezes
"""

import math
import random
import threading
import builtins
import tkinter as tk
from tkinter import scrolledtext

# ═══════════════════════════════════════════════════════════════════════════════
# ── BitWhisker R1 Model (unchanged from v1) ────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

INT4_LUT: list = []
for _byte in range(256):
    _q1 = (_byte >> 4) & 0x0F
    _q1 = _q1 if _q1 < 8 else _q1 - 16
    _q2 = _byte & 0x0F
    _q2 = _q2 if _q2 < 8 else _q2 - 16
    INT4_LUT.append((_q1, _q2))

def vec_add(a, b):       return [x + y for x, y in zip(a, b)]
def vec_scale(v, s):     return [x * s for x in v]

def softmax(v):
    if not v: return []
    m = max(v); exps = [math.exp(x - m) for x in v]; s = sum(exps)
    return [e / s for e in exps]

def rms_norm(v, weight, eps=1e-6):
    ms = sum(x*x for x in v) / len(v)
    inv = 1.0 / math.sqrt(ms + eps)
    return [x * inv * w for x, w in zip(v, weight)]

def swiglu(gate, up):
    return [u * (g / (1.0 + math.exp(-g))) for g, u in zip(gate, up)]

def quantise_act(v, bits=8):
    ma = max(abs(x) for x in v) or 1e-5
    qm = (1 << (bits - 1)) - 1
    sc = ma / qm
    return [max(-qm, min(qm, round(x / sc))) for x in v], sc

def _pack_row(row, scale):
    packed = []
    for i in range(0, len(row), 2):
        w1 = row[i]; w2 = row[i+1] if i+1 < len(row) else 0.0
        q1 = max(-8, min(7, round(w1 / scale)))
        q2 = max(-8, min(7, round(w2 / scale)))
        packed.append(((q1 & 0x0F) << 4) | (q2 & 0x0F))
    return packed

class BitLinear:
    __slots__ = ("in_f","out_f","seed","ws","_cache")
    def __init__(self, in_f, out_f, seed=0):
        self.in_f = in_f; self.out_f = out_f; self.seed = seed
        self.ws = 0.02 / 7.0
        self._cache = [None] * out_f

    def _get_row(self, ri):
        if self._cache[ri] is not None: return self._cache[ri]
        rng = random.Random(self.seed ^ (ri * 2_654_435_761))
        row = [rng.gauss(0, 0.02) for _ in range(self.in_f)]
        self._cache[ri] = _pack_row(row, 0.02 / 7.0)
        return self._cache[ri]

    def forward(self, x):
        xq, xs = quantise_act(x); lut = INT4_LUT; result = []
        for ri in range(self.out_f):
            pr = self._get_row(ri); acc = 0
            for i, packed in enumerate(pr):
                q1, q2 = lut[packed]; idx = i * 2
                acc += q1 * xq[idx]
                if idx + 1 < len(xq): acc += q2 * xq[idx + 1]
            result.append(acc * self.ws * xs)
        return result

_seed_counter = 0
def _next_seed():
    global _seed_counter
    _seed_counter += 1
    return (_seed_counter * 0x9E3779B9) & 0xFFFFFFFF

def apply_rope(vec, pos, theta=10000.0):
    out = list(vec); n = len(vec)
    for i in range(0, n - 1, 2):
        freq = 1.0 / (theta ** (i / n)); angle = pos * freq
        c, s = math.cos(angle), math.sin(angle)
        out[i] = vec[i]*c - vec[i+1]*s
        out[i+1] = vec[i+1]*c + vec[i]*s
    return out

class BitWhiskerMLA:
    def __init__(self, dim, n_q=4, n_kv=2):
        self.n_q = n_q; self.n_kv = n_kv
        self.hd = dim // n_q; self.kv_rank = max(dim // 8, 16)
        self.w_q       = BitLinear(dim, dim,              _next_seed())
        self.w_down_kv = BitLinear(dim, self.kv_rank,     _next_seed())
        self.w_up_k    = BitLinear(self.kv_rank, n_kv*self.hd, _next_seed())
        self.w_up_v    = BitLinear(self.kv_rank, n_kv*self.hd, _next_seed())
        self.w_out     = BitLinear(dim, dim,              _next_seed())
        self.k_cache: list = []; self.v_cache: list = []

    def reset_cache(self): self.k_cache.clear(); self.v_cache.clear()

    @staticmethod
    def _split(flat, n):
        hd = len(flat) // n
        return [flat[i*hd:(i+1)*hd] for i in range(n)]

    def forward(self, x, pos=0):
        q_heads = [apply_rope(h, pos) for h in self._split(self.w_q.forward(x), self.n_q)]
        c_kv = self.w_down_kv.forward(x)
        k_heads = [apply_rope(h, pos) for h in self._split(self.w_up_k.forward(c_kv), self.n_kv)]
        v_heads = self._split(self.w_up_v.forward(c_kv), self.n_kv)
        self.k_cache.append(k_heads); self.v_cache.append(v_heads)
        scale = 1.0 / math.sqrt(self.hd); q_per_kv = self.n_q // self.n_kv
        out_heads = []
        for qi in range(self.n_q):
            kvi = qi // q_per_kv; q_h = q_heads[qi]
            scores = [sum(a*b for a,b in zip(q_h, step[kvi]))*scale for step in self.k_cache]
            weights = softmax(scores)
            attn = [0.0] * self.hd
            for w, sv in zip(weights, self.v_cache):
                for d in range(self.hd): attn[d] += w * sv[kvi][d]
            out_heads.append(attn)
        return self.w_out.forward([x for head in out_heads for x in head])

class Expert:
    def __init__(self, dim, inter):
        self.gate = BitLinear(dim, inter, _next_seed())
        self.up   = BitLinear(dim, inter, _next_seed())
        self.down = BitLinear(inter, dim, _next_seed())
    def forward(self, x):
        return self.down.forward(swiglu(self.gate.forward(x), self.up.forward(x)))

class BitWhiskerMoE:
    def __init__(self, dim, n_routed=4, top_k=2, n_shared=1):
        inter = dim
        self.top_k  = top_k
        self.shared = [Expert(dim, inter) for _ in range(n_shared)]
        self.routed = [Expert(dim, inter) for _ in range(n_routed)]
        self.router = BitLinear(dim, n_routed, _next_seed())

    def forward(self, x):
        out = [0.0] * len(x)
        for exp in self.shared: out = vec_add(out, exp.forward(x))
        weights = softmax(self.router.forward(x))
        top_k = sorted(enumerate(weights), key=lambda kv: -kv[1])[:self.top_k]
        wsum = sum(w for _, w in top_k) or 1e-9
        for idx, w in top_k:
            out = vec_add(out, vec_scale(self.routed[idx].forward(x), w / wsum))
        return out

class BitWhiskerBlock:
    def __init__(self, dim, n_routed, n_shared):
        self.norm_attn = [1.0] * dim; self.norm_ffn = [1.0] * dim
        self.attn = BitWhiskerMLA(dim); self.moe = BitWhiskerMoE(dim, n_routed, n_shared=n_shared)
    def forward(self, x, pos):
        x = vec_add(x, self.attn.forward(rms_norm(x, self.norm_attn), pos))
        x = vec_add(x, self.moe.forward(rms_norm(x, self.norm_ffn)))
        return x
    def reset_cache(self): self.attn.reset_cache()

class BitWhisker4B:
    DIM=256; VOCAB=1024; N_LAYERS=4; N_ROUTED=4; N_SHARED=1
    def __init__(self):
        self.dim=self.DIM; self.vocab=self.VOCAB
        self._embed_cache={}
        self.blocks=[BitWhiskerBlock(self.DIM,self.N_ROUTED,self.N_SHARED) for _ in range(self.N_LAYERS)]
        self.norm_out=[1.0]*self.DIM
        self.lm_head=BitLinear(self.DIM,self.VOCAB,_next_seed())
    def _embed(self, tid):
        t=tid%self.vocab
        if t not in self._embed_cache:
            rng=random.Random(t^0xDEADBEEF)
            self._embed_cache[t]=[rng.gauss(0,0.02) for _ in range(self.dim)]
        return list(self._embed_cache[t])
    def reset(self):
        for b in self.blocks: b.reset_cache()
    def forward_token(self, tid, pos):
        x=self._embed(tid)
        for b in self.blocks: x=b.forward(x,pos)
        return self.lm_head.forward(rms_norm(x,self.norm_out))

_VOCAB_TOKENS = (
    "BitWhisker think reason chain layer expert routed shared "
    "attention latent rope swiglu norm residual token embed forward "
    "fast python inference quantized int4 weight activation scale "
    "compute step result correct okay yes the of is in to and a "
    "with for that this it by from local pure generate "
    "sequence causal cache head query key value output project "
    "matrix vector softmax decode encode efficient real four billion "
    "parameter architecture model done next then because "
    "reasoning optimal accurate helpful response answer question "
).split()
_VOCAB_TABLE = (_VOCAB_TOKENS * 1000)[:32_000]

def nucleus_sample(logits, top_p=0.9, temp=0.8):
    if temp <= 0: return max(range(len(logits)), key=lambda i: logits[i])
    probs   = softmax([l / temp for l in logits])
    indexed = sorted(enumerate(probs), key=lambda kv: -kv[1])
    cumul, cutoff = 0.0, []
    for idx, p in indexed:
        cumul += p; cutoff.append((idx, p))
        if cumul >= top_p: break
    r, s = random.random() * sum(p for _, p in cutoff), 0.0
    for idx, p in cutoff:
        s += p
        if s >= r: return idx
    return cutoff[-1][0]

# ═══════════════════════════════════════════════════════════════════════════════
# ── Tkinter GUI  ───────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# ── Colours (ChatGPT dark palette) ─────────────────────────────────────────────
BG_MAIN     = "#212121"   # main window
BG_SIDEBAR  = "#171717"   # left strip / title bar
BG_INPUT    = "#2f2f2f"   # input box bg
BG_BUBBLE_U = "#2f2f2f"   # user bubble
BG_BUBBLE_B = "#212121"   # model bubble (same as bg — no box)
BG_BTN      = "#ececec"   # button surface  (light → black text readable)
FG_BTN      = "#000000"   # ← black label on buttons as requested
FG_MAIN     = "#ececec"   # primary text
FG_DIM      = "#8e8ea0"   # muted text / placeholders
FG_THINK    = "#6b6b8a"   # <think> block colour
FG_USER     = "#ececec"
FG_MODEL    = "#ececec"
ACCENT      = "#10a37f"   # ChatGPT green (send button)
FG_SEND     = "#000000"   # black label on send button

FONT_MAIN  = ("Segoe UI", 10)
FONT_BOLD  = ("Segoe UI", 10, "bold")
FONT_MONO  = ("Courier New", 9)
FONT_TITLE = ("Segoe UI", 11, "bold")
FONT_SMALL = ("Segoe UI", 8)

N_TOKENS = 50   # tokens to generate per response

class BitWhiskerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BitWhisker R1 Tkinter")
        self.geometry("600x400")
        self.resizable(False, False)
        self.configure(bg=BG_MAIN)

        self.model     = BitWhisker4B()
        self._busy     = False
        self._stop_evt = threading.Event()

        self._build_ui()
        self._post_system("BitWhisker R1 ready. MLA + MoE + SwiGLU  •  W4A8 INT4  •  No API")

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Title bar strip ────────────────────────────────────────────────────
        title_bar = tk.Frame(self, bg=BG_SIDEBAR, height=36)
        title_bar.pack(fill="x", side="top")
        title_bar.pack_propagate(False)

        tk.Label(title_bar, text="⬡  BitWhisker R1",
                 bg=BG_SIDEBAR, fg=FG_MAIN, font=FONT_TITLE,
                 padx=12).pack(side="left", pady=6)

        # toolbar buttons (black labels, light bg)
        for label, cmd in [("↺ Reset", self._reset), ("✕ Clear", self._clear)]:
            tk.Button(title_bar, text=label, command=cmd,
                      bg=BG_BTN, fg=FG_BTN, activebackground="#d4d4d4",
                      activeforeground=FG_BTN,
                      font=FONT_SMALL, relief="flat", cursor="hand2",
                      padx=8, pady=2, bd=0).pack(side="right", padx=4, pady=6)

        # ── Chat area ──────────────────────────────────────────────────────────
        chat_frame = tk.Frame(self, bg=BG_MAIN)
        chat_frame.pack(fill="both", expand=True, padx=0, pady=0)

        self.chat = scrolledtext.ScrolledText(
            chat_frame, bg=BG_MAIN, fg=FG_MAIN,
            font=FONT_MAIN, relief="flat", bd=0,
            state="disabled", wrap="word",
            selectbackground="#444", padx=14, pady=10,
            insertbackground=FG_MAIN,
        )
        self.chat.pack(fill="both", expand=True)

        # Configure text tags
        self.chat.tag_config("user_label", foreground=FG_DIM,  font=FONT_SMALL)
        self.chat.tag_config("user_text",  foreground=FG_USER, font=FONT_MAIN)
        self.chat.tag_config("bot_label",  foreground=ACCENT,  font=FONT_SMALL)
        self.chat.tag_config("bot_text",   foreground=FG_MODEL, font=FONT_MAIN)
        self.chat.tag_config("think_text", foreground=FG_THINK, font=FONT_MONO)
        self.chat.tag_config("system",     foreground=FG_DIM,  font=FONT_SMALL,
                              justify="center")
        self.chat.tag_config("divider",    foreground=BG_SIDEBAR)

        # ── Input bar ──────────────────────────────────────────────────────────
        bar = tk.Frame(self, bg=BG_INPUT, pady=6)
        bar.pack(fill="x", side="bottom")

        self.entry = tk.Text(bar, bg=BG_INPUT, fg=FG_MAIN,
                             insertbackground=FG_MAIN,
                             font=FONT_MAIN, relief="flat", bd=0,
                             height=2, wrap="word",
                             selectbackground="#555")
        self.entry.pack(side="left", fill="x", expand=True, padx=(12, 6), pady=2)
        self.entry.bind("<Return>",       self._on_return)
        self.entry.bind("<Shift-Return>", lambda e: None)   # allow newline
        self._set_placeholder()

        # Send button — ChatGPT green, black label
        self.send_btn = tk.Button(
            bar, text="▶", command=self._send,
            bg=ACCENT, fg=FG_SEND,
            activebackground="#0d8c6d", activeforeground=FG_SEND,
            font=("Segoe UI", 12, "bold"), relief="flat",
            cursor="hand2", width=3, bd=0
        )
        self.send_btn.pack(side="right", padx=(0, 10), pady=2)

        # Stop button — light bg, black label
        self.stop_btn = tk.Button(
            bar, text="■", command=self._stop,
            bg=BG_BTN, fg=FG_BTN,
            activebackground="#d4d4d4", activeforeground=FG_BTN,
            font=("Segoe UI", 12), relief="flat",
            cursor="hand2", width=3, bd=0, state="disabled"
        )
        self.stop_btn.pack(side="right", padx=(0, 4), pady=2)

    # ── Placeholder ───────────────────────────────────────────────────────────

    def _set_placeholder(self):
        self.entry.insert("1.0", "Message BitWhisker…")
        self.entry.config(fg=FG_DIM)
        self._placeholder_active = True
        self.entry.bind("<FocusIn>",  self._clear_placeholder)
        self.entry.bind("<FocusOut>", self._restore_placeholder)

    def _clear_placeholder(self, _=None):
        if self._placeholder_active:
            self.entry.delete("1.0", "end")
            self.entry.config(fg=FG_MAIN)
            self._placeholder_active = False

    def _restore_placeholder(self, _=None):
        if not self.entry.get("1.0", "end").strip():
            self.entry.insert("1.0", "Message BitWhisker…")
            self.entry.config(fg=FG_DIM)
            self._placeholder_active = True

    # ── Chat display helpers ──────────────────────────────────────────────────

    def _append(self, text, tag):
        self.chat.config(state="normal")
        self.chat.insert("end", text, tag)
        self.chat.see("end")
        self.chat.config(state="disabled")

    def _post_system(self, msg):
        self._append(f"\n  {msg}\n", "system")

    def _post_user(self, text):
        self._append("\nYou\n", "user_label")
        self._append(f"{text}\n", "user_text")

    def _post_bot_label(self):
        self._append("\nBitWhisker R1\n", "bot_label")

    def _post_token(self, word):
        """Append a single token word to the current bot message."""
        self._append(word + " ", "bot_text")

    def _post_think(self, text):
        self._append(text, "think_text")

    def _post_newline(self):
        self._append("\n", "bot_text")

    # ── Send / Stop / Reset / Clear ───────────────────────────────────────────

    def _on_return(self, event):
        if event.state & 0x1:    # Shift held → literal newline
            return
        self._send()
        return "break"           # prevent default newline insertion

    def _send(self):
        if self._busy:
            return
        if self._placeholder_active:
            return
        text = self.entry.get("1.0", "end").strip()
        if not text:
            return

        self.entry.delete("1.0", "end")
        self._restore_placeholder()
        self._post_user(text)
        self._start_generation(text)

    def _start_generation(self, prompt):
        self._busy = True
        self._stop_evt.clear()
        self.send_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.model.reset()
        t = threading.Thread(target=self._generate_thread,
                             args=(prompt,), daemon=True)
        t.start()

    def _generate_thread(self, prompt):
        # Think block
        think_lines = [
            f"<think>",
            f"  Prompt tokens  : {len(prompt.split())}",
            f"  Architecture   : {BitWhisker4B.N_LAYERS}L × dim{BitWhisker4B.DIM}",
            f"  Active experts : top-2 of {BitWhisker4B.N_ROUTED} routed + {BitWhisker4B.N_SHARED} shared",
            f"  Quantisation   : W4A8 INT4 LUT • lazy seeded BitLinear",
            f"  Sampling       : nucleus p=0.9, temperature=0.8",
            f"</think>",
        ]
        self.after(0, self._post_bot_label)
        for line in think_lines:
            self.after(0, self._post_think, line + "\n")

        # Generation
        seed     = sum(ord(c) * (i+1) for i, c in enumerate(prompt)) % BitWhisker4B.VOCAB
        token_id = seed

        self.after(0, self._post_newline)
        for pos in range(N_TOKENS):
            if self._stop_evt.is_set():
                break
            logits   = self.model.forward_token(token_id, pos)
            token_id = nucleus_sample(logits)
            word     = _VOCAB_TABLE[token_id % len(_VOCAB_TABLE)]
            self.after(0, self._post_token, word)

        self.after(0, self._post_newline)
        self.after(0, self._generation_done)

    def _generation_done(self):
        self._busy = False
        self.send_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def _stop(self):
        self._stop_evt.set()

    def _reset(self):
        self._stop_evt.set()
        self.model.reset()
        self._busy = False
        self.send_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self._post_system("KV cache cleared — conversation reset.")

    def _clear(self):
        self._stop_evt.set()
        self.model.reset()
        self._busy = False
        self.send_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.chat.config(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.config(state="disabled")
        self._post_system("BitWhisker R1 ready. MLA + MoE + SwiGLU  •  W4A8 INT4  •  No API")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = BitWhiskerApp()
    app.mainloop()
