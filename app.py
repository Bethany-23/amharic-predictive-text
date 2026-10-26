import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import streamlit as st

st.set_page_config(page_title="Amharic Next-Word Suggestion", page_icon="🇪🇹", layout="centered")
MODEL_DIR = Path(__file__).parent / "model"

# Match the notebook's preprocessing rules.
TERMINATORS = "።፧?!"
RE_SENT = re.compile(r"[።፧?!]+")
ETH = r"\u1200-\u1358"
RE_WORD = re.compile(rf"[{ETH}]+")
SEP = r"[./\u00b7\u2022\u2024\u2027]"
RE_JOIN = re.compile(rf"(?<![{ETH}])([{ETH}])[ \t]?{SEP}[ \t]?(?=[{ETH}])")
JOIN_SPACED_ABBR = True
SPACED_ABBR = ["ዓ ም", "እ ኤ አ", "እ አ አ", "ዶ ር"]
WS = r"[ \t\u00a0\u2009\u202f]+"
RE_ABBR = re.compile("|".join(
    rf"(?<![{ETH}])" + WS.join(re.escape(p) for p in a.split()) + rf"(?![{ETH}])"
    for a in SPACED_ABBR
))
RE_WS = re.compile(WS)


def clean_to_sentences(text):
    text = unicodedata.normalize("NFC", text).replace("፡፡", "።")
    text = RE_JOIN.sub(r"\1", text)
    if JOIN_SPACED_ABBR:
        text = RE_ABBR.sub(lambda m: RE_WS.sub("", m.group(0)), text)
    result = []
    for segment in RE_SENT.split(text):
        tokens = RE_WORD.findall(segment)
        if tokens:
            result.append(tokens)
    return result


@st.cache_resource(show_spinner="Loading the Amharic language model…")
def load_model():
    required = ["vocab.json", "ngram_totals.json"]
    required += [f"ngram_{kind}_{n}.npy" for kind in ("keys", "counts") for n in range(1, 5)]
    missing = [name for name in required if not (MODEL_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Model files are missing from the model/ folder: " + ", ".join(missing)
            + ". See README.md for the required files."
        )

    with (MODEL_DIR / "vocab.json").open(encoding="utf-8") as f:
        words = json.load(f)
    w2i = {word: i for i, word in enumerate(words)}
    V = len(words)
    UNK, BOS, EOS = 0, 1, 2

    # Memory-mapped arrays reduce the initial RAM spike, though the full model is still large.
    keys = {n: np.load(MODEL_DIR / f"ngram_keys_{n}.npy", mmap_mode="r") for n in range(1, 5)}
    cnt = {n: np.load(MODEL_DIR / f"ngram_counts_{n}.npy", mmap_mode="r") for n in range(1, 5)}
    with (MODEL_DIR / "ngram_totals.json").open(encoding="utf-8") as f:
        total = {int(k): int(v) for k, v in json.load(f).items()}

    uni = np.zeros(V, dtype=np.int64)
    uni[keys[1]] = cnt[1]
    n_pred = int(uni.sum() - uni[BOS])

    # Build compact context statistics needed by interpolated absolute discounting.
    ck, ct, cy, discounts = {}, {}, {}, {}
    for n in (2, 3, 4):
        context_ids = keys[n] // V
        starts = np.flatnonzero(np.r_[True, context_ids[1:] != context_ids[:-1]])
        ck[n] = context_ids[starts]
        ct[n] = np.add.reduceat(cnt[n], starts, dtype=np.int64).astype(np.int32)
        cy[n] = np.diff(np.r_[starts, len(context_ids)]).astype(np.int32)
        del context_ids, starts
        n1 = int(np.count_nonzero(cnt[n] == 1))
        n2 = int(np.count_nonzero(cnt[n] == 2))
        discounts[n] = n1 / (n1 + 2 * n2) if n1 + 2 * n2 else 0.9

    special_set = {UNK, BOS, EOS}
    top_order = np.argsort(-uni)
    top_uni = np.array([w for w in top_order[:100] if w not in special_set][:50], dtype=np.int64)
    del top_order
    return {
        "words": words, "w2i": w2i, "V": V, "UNK": UNK, "BOS": BOS, "EOS": EOS,
        "keys": keys, "cnt": cnt, "total": total, "uni": uni, "N_PRED": n_pred,
        "ck": ck, "ct": ct, "cy": cy, "D": discounts, "TOP_UNI": top_uni,
    }


def key_of(t, m):
    V, keys = m["V"], m["keys"]
    if len(t) == 4:
        prefix = key_of(t[:3], m)
        pos = int(np.searchsorted(keys[3], prefix))
        if pos >= len(keys[3]) or keys[3][pos] != prefix:
            return None
        return pos * V + t[3]
    k = 0
    for w in t:
        k = k * V + w
    return k


def continuations(history, m):
    n = len(history) + 1
    if n == 1:
        return np.arange(m["V"]), m["uni"]
    if n == 4:
        k3 = key_of(history, m)
        pos = int(np.searchsorted(m["keys"][3], k3))
        if pos >= len(m["keys"][3]) or m["keys"][3][pos] != k3:
            return None
        base, arr = pos * m["V"], m["keys"][4]
    else:
        base, arr = key_of(history, m) * m["V"], m["keys"][n]
    lo, hi = int(np.searchsorted(arr, base)), int(np.searchsorted(arr, base + m["V"]))
    if hi == lo:
        return None
    return (arr[lo:hi] - base).astype(np.int64), m["cnt"][n][lo:hi]


def context_lookup(n, history, m):
    if n == 4:
        k3 = key_of(history, m)
        pos = int(np.searchsorted(m["keys"][3], k3))
        if pos >= len(m["keys"][3]) or m["keys"][3][pos] != k3:
            return None
        ident = pos
    else:
        ident = key_of(history, m)
    i = int(np.searchsorted(m["ck"][n], ident))
    if i < len(m["ck"][n]) and m["ck"][n][i] == ident:
        return int(m["ct"][n][i]), int(m["cy"][n][i])
    return None


def candidate_suggestions(history, order, k, m):
    V, keys, cnt = m["V"], m["keys"], m["cnt"]
    history = tuple(history[-3:])
    parts = [m["TOP_UNI"]]
    for n in range(2, min(order, len(history) + 1) + 1):
        found = continuations(tuple(history[-(n - 1):]), m)
        if found is not None:
            parts.append(found[0].astype(np.int64))
    candidates = np.unique(np.concatenate(parts))
    candidates = candidates[~np.isin(candidates, np.array([m["UNK"], m["BOS"], m["EOS"]]))]
    probs = m["uni"][candidates].astype(np.float64) / m["N_PRED"]

    for n in range(2, min(order, len(history) + 1) + 1):
        h = tuple(history[-(n - 1):])
        context = context_lookup(n, h, m)
        if context is None:
            continue
        context_total, distinct_next = context
        if n == 4:
            base = int(np.searchsorted(keys[3], key_of(h, m))) * V
        else:
            base = key_of(h, m) * V
        packed = base + candidates
        arr = keys[n]
        positions = np.minimum(np.searchsorted(arr, packed), len(arr) - 1)
        counts = np.where(arr[positions] == packed, cnt[n][positions], 0).astype(np.float64)
        discount = m["D"][n]
        probs = np.maximum(counts - discount, 0.0) / context_total + discount * distinct_next / context_total * probs

    top = np.argsort(-probs)[:k]
    return [(m["words"][int(candidates[i])], float(probs[i])) for i in top]


def history_from_text(text, m):
    normalized = unicodedata.normalize("NFC", text or "").replace("፡፡", "።")
    ended = normalized.rstrip()[-1:] in tuple(TERMINATORS)
    sentences = clean_to_sentences(normalized)
    tokens = [] if ended or not sentences else sentences[-1]
    ids = [m["w2i"].get(token, m["UNK"]) for token in tokens]
    history = ids[-3:] if len(ids) >= 3 else [m["BOS"]] + ids
    return history, tokens


st.markdown("# አማርኛ የቃል ትንበያ")
st.markdown("**Amharic Next-Word Suggestion**  ·  GPAC corpus · interpolated n-gram language model")
st.write("Type Amharic text, then choose a suggested word to continue. The model predicts from the current sentence.")

try:
    model = load_model()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:
    st.error(f"The model could not be loaded ({type(exc).__name__}: {exc}). The full model is large; see README.md for deployment limitations.")
    st.stop()

if "draft" not in st.session_state:
    st.session_state.draft = ""


def append_suggestion(word):
    current = st.session_state.get("draft", "").rstrip()
    st.session_state.draft = f"{current} {word} " if current else f"{word} "


def delete_last_word():
    st.session_state.draft = " ".join((st.session_state.get("draft", "") or "").split()[:-1])


def clear_text():
    st.session_state.draft = ""

order = st.select_slider("N-gram order", options=[1, 2, 3, 4], value=4, help="4-gram uses up to the last three words as context.")
text = st.text_area("Your Amharic text", key="draft", height=130, placeholder="ኢትዮጵያ ውስጥ የቡና ")
history, tokens = history_from_text(text, model)
suggestions = candidate_suggestions(history, order, 5, model)
context_words = [model["words"][i] for i in history if i != model["BOS"]]
st.caption("Context used: " + (" ".join(context_words) if context_words else "start of sentence"))
if tokens and model["w2i"].get(tokens[-1], model["UNK"]) == model["UNK"]:
    st.info(f"The last token ‘{tokens[-1]}’ is outside the model vocabulary and is treated as <unk>.")
if order == 1:
    st.caption("The 1-gram model ignores the context and suggests frequent words.")

st.subheader("Top 5 next-word suggestions")
cols = st.columns(5)
for i, (word, probability) in enumerate(suggestions):
    with cols[i]:
        st.button(
            f"{word}\n{probability * 100:.1f}%",
            key=f"suggest_{i}", use_container_width=True,
            on_click=append_suggestion, args=(word,),
        )

left, right = st.columns(2)
with left:
    st.button("Delete last word", use_container_width=True, on_click=delete_last_word)
with right:
    st.button("Clear text", use_container_width=True, on_click=clear_text)

with st.expander("Compare all four models"):
    for model_order in range(1, 5):
        results = candidate_suggestions(history, model_order, 5, model)
        st.markdown(f"**{model_order}-gram:** " + " · ".join(f"{word} ({p * 100:.1f}%)" for word, p in results))

st.caption("Research prototype for an academic NLP project. Suggestions are ranked by model probability, not guaranteed to be grammatically correct.")
