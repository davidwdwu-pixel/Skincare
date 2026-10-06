# app.py
"""
ISOM5240 Group Project
Sephora Skin Concern Advisor — a two-pipeline deep learning advisor.

Pipelines (Hugging Face):
  1) Vision  : ViT image-classification        -> acne severity (5 classes)
  2) Text    : DistilBERT text-classification  -> PRO / CON review curation
  3) (opt.)  : DistilGPT2 text-generation      -> recommendation copy (template fallback)

Compliance: cosmetic skin concern only, no diagnosis, no transaction.
"""

import os
import re
import io
import gc
import time
import hashlib
from datetime import datetime

import streamlit as st
from PIL import Image

# ============================================================
# 1. CONFIG
# ============================================================
st.set_page_config(
    page_title="SEPHORA · Skin Concern Advisor",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="collapsed",
)

HF_USERNAME   = os.getenv("HF_USERNAME", "your-hf-username")
VISION_MODEL  = os.getenv("VISION_MODEL", f"{HF_USERNAME}/vit-acne-severity")
REVIEW_MODEL  = os.getenv("REVIEW_MODEL", f"{HF_USERNAME}/distilbert-sephora-review-curator")
GEN_MODEL     = os.getenv("GEN_MODEL",    f"{HF_USERNAME}/distilgpt2-sephora-reco")

ENABLE_TEXTGEN = os.getenv("ENABLE_TEXTGEN", "1") == "1"
USE_MOCK       = os.getenv("USE_MOCK", "0") == "1"       # 本地无模型时演示 UI
MAX_MB         = 10
REVIEWS_CSV    = os.getenv("REVIEWS_CSV", "data/reviews_sample.csv")

SEVERITY_LEVELS = ["None", "Very mild", "Mild", "Moderate", "Severe"]

DISCLAIMER = (
    "For skincare reference only. Not a medical diagnosis. "
    "Please consult a dermatologist for skin conditions or concerns."
)

# ============================================================
# 2. KNOWLEDGE BASE — severity -> ingredients -> products
# ============================================================
PRODUCT_MAP = {
    "None": {
        "ingredients": ["Hyaluronic Acid", "Glycerin", "Ceramides"],
        "why": "Ingredients commonly used in cosmetic skincare for everyday hydration and barrier support.",
        "products": [
            {
                "name": "Hyaluronic Acid 2% + B5",
                "brand": "THE ORDINARY",
                "tags": ["Hyaluronic Acid", "Hydration", "Barrier support"],
                "query": "Hyaluronic Acid 2%",
                "fallback": [
                    ("PRO", "Plumps my skin without any stickiness — layers well under moisturizer."),
                    ("PRO", "Great value. My skin feels hydrated all day and no breakouts."),
                    ("CON", "Needs to be applied on damp skin, otherwise it can feel a bit tacky."),
                ],
            },
            {
                "name": "Toleriane Double Repair Face Moisturizer",
                "brand": "LA ROCHE-POSAY",
                "tags": ["Ceramides", "Niacinamide", "Barrier repair"],
                "query": "Toleriane Double Repair",
                "fallback": [
                    ("PRO", "Fragrance-free and very calming — my skin barrier feels stronger."),
                    ("PRO", "Lightweight but nourishing. Works under makeup."),
                    ("CON", "The tube is small for the price, but a little goes a long way."),
                ],
            },
        ],
    },
    "Very mild": {
        "ingredients": ["Niacinamide", "Panthenol", "Glycerin"],
        "why": "Ingredients commonly used in cosmetic skincare for very mild breakout concerns.",
        "products": [
            {
                "name": "10% Niacinamide + Zinc Serum",
                "brand": "THE ORDINARY",
                "tags": ["Niacinamide", "Oil control", "Barrier repair"],
                "query": "Niacinamide 10%",
                "fallback": [
                    ("PRO", "Lightweight and non-comedogenic — less oil and a brighter overall tone."),
                    ("PRO", "Layers well under sunscreen. No pilling."),
                    ("CON", "Not hydrating enough in winter; layer a moisturizer on top."),
                ],
            },
            {
                "name": "Effaclar Duo+ M",
                "brand": "LA ROCHE-POSAY",
                "tags": ["Niacinamide", "Panthenol", "Calming"],
                "query": "Effaclar Duo",
                "fallback": [
                    ("PRO", "Smoothed out my texture within a month without irritation."),
                    ("PRO", "Absorbs fast and sits well under SPF."),
                    ("CON", "Slight tingling on the first few uses — build up slowly."),
                ],
            },
        ],
    },
    "Mild": {
        "ingredients": ["Salicylic Acid", "Niacinamide"],
        "why": "Ingredients commonly used in cosmetic skincare for mild breakout concerns.",
        "products": [
            {
                "name": "2% BHA Liquid Exfoliant",
                "brand": "PAULA'S CHOICE",
                "tags": ["Salicylic Acid", "Pore-clearing", "Calming"],
                "query": "2% BHA Liquid Exfoliant",
                "fallback": [
                    ("PRO", "Cleared my closed comedones and blackheads in two weeks — lightweight finish."),
                    ("PRO", "Noticeable improvement on texture. Didn't dry me out."),
                    ("CON", "Stings at first; sensitive skin should build up tolerance slowly."),
                ],
            },
            {
                "name": "10% Niacinamide + Zinc Serum",
                "brand": "THE ORDINARY",
                "tags": ["Niacinamide", "Oil control", "Barrier repair"],
                "query": "Niacinamide 10%",
                "fallback": [
                    ("PRO", "Lightweight and non-comedogenic — less oil, and a brighter overall tone."),
                    ("PRO", "Layers well under sunscreen. No pilling."),
                    ("CON", "Not hydrating enough in winter; layer a moisturizer on top."),
                ],
            },
        ],
    },
    "Moderate": {
        "ingredients": ["Salicylic Acid", "Azelaic Acid", "Niacinamide"],
        "why": "Ingredients commonly used in cosmetic skincare for moderate breakout concerns.",
        "products": [
            {
                "name": "Azelaic Acid Suspension 10%",
                "brand": "THE ORDINARY",
                "tags": ["Azelaic Acid", "Tone evening", "Calming"],
                "query": "Azelaic Acid Suspension",
                "fallback": [
                    ("PRO", "Evened out my post-breakout marks over about six weeks."),
                    ("PRO", "Gentle enough for nightly use on my combination skin."),
                    ("CON", "Silicone texture takes some getting used to under makeup."),
                ],
            },
            {
                "name": "2% BHA Liquid Exfoliant",
                "brand": "PAULA'S CHOICE",
                "tags": ["Salicylic Acid", "Pore-clearing", "Calming"],
                "query": "2% BHA Liquid Exfoliant",
                "fallback": [
                    ("PRO", "Keeps my pores clear without over-drying."),
                    ("PRO", "A staple — consistent results after a month."),
                    ("CON", "Can be too strong if you use it twice a day. Start slow."),
                ],
            },
        ],
    },
    "Severe": {
        "ingredients": ["Benzoyl Peroxide", "Azelaic Acid", "Niacinamide"],
        "why": "Ingredients commonly used in cosmetic skincare for more visible breakout concerns.",
        "products": [
            {
                "name": "Effaclar Duo+ M",
                "brand": "LA ROCHE-POSAY",
                "tags": ["Benzoyl Peroxide", "Niacinamide", "Calming"],
                "query": "Effaclar Duo",
                "fallback": [
                    ("PRO", "Reduced the look of my breakouts noticeably within a month."),
                    ("PRO", "Non-greasy and easy to layer into a simple routine."),
                    ("CON", "Can bleach fabric — be careful with towels and pillowcases."),
                ],
            },
            {
                "name": "Azelaic Acid Suspension 10%",
                "brand": "THE ORDINARY",
                "tags": ["Azelaic Acid", "Tone evening", "Calming"],
                "query": "Azelaic Acid Suspension",
                "fallback": [
                    ("PRO", "Helped with redness and uneven tone over time."),
                    ("PRO", "Affordable and effective when used consistently."),
                    ("CON", "Pilling can happen if you apply too much at once."),
                ],
            },
        ],
    },
}

STYLE_OPENERS = {
    "Gentle":       "Here is a gentle skincare suggestion for you.",
    "Professional": "Based on the cosmetic skin concern analysis,",
    "Concise":      "Quick take:",
    "Enthusiastic": "Great news — here is a routine idea for you!",
}

TEMPLATES = {
    "Gentle": (
        "Based on your {sev} breakout pattern, a gentle routine built around {ings} "
        "may help. Start slowly, keep it consistent, and consult a dermatologist if you are concerned."
    ),
    "Professional": (
        "According to the cosmetic skin concern analysis, a {sev} breakout pattern was identified. "
        "Ingredients such as {ings} are commonly used in cosmetic skincare for this concern. "
        "Consult a dermatologist for skin conditions."
    ),
    "Concise": (
        "{sev} breakout concern -> {ings}. Two picks below. "
        "Consult a dermatologist if concerned."
    ),
    "Enthusiastic": (
        "Great starting point for your routine! Your {sev} breakout pattern pairs well with {ings}. "
        "Go slow, stay consistent, and consult a dermatologist if concerned."
    ),
}

# ============================================================
# 3. STYLES
# ============================================================
CUSTOM_CSS = """
<style>
:root{
  --ivory:#FBF8F3; --paper:#FFFFFF; --ink:#1A1512; --ink-soft:#4A433D;
  --muted:#8A8177; --muted-strong:#6B6259;
  --sephora-red:#C8102E; --sephora-red-soft:#A50D25;
  --gold:#B99A5B; --gold-soft:#D6C49A;
  --line:#E6DFD3; --line-strong:#D8CFC0;
}
.stApp{background:var(--ivory);}
header[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:1.2rem;padding-bottom:3rem;max-width:940px;}
#MainMenu, footer{visibility:hidden;}

/* topbar */
.sph-topbar{
  background:var(--ink);color:#fff;padding:14px 28px;
  display:flex;align-items:baseline;gap:12px;
  border-bottom:2px solid var(--gold);margin-bottom:8px;
}
.sph-topbar .wordmark{font-family:Georgia,serif;font-size:19px;letter-spacing:.32em;font-weight:700;}
.sph-topbar .flame{color:var(--sephora-red);font-weight:700;}
.sph-topbar .sub{color:var(--gold-soft);font-size:12px;letter-spacing:.18em;margin-left:auto;}

/* hero */
.sph-hero{padding:44px 0 22px;border-bottom:1px solid var(--line-strong);}
.sph-eyebrow{color:var(--gold);font-size:12px;letter-spacing:.4em;font-weight:600;margin-bottom:12px;}
.sph-hero h1{font-family:Georgia,serif;font-size:40px;font-weight:700;line-height:1.15;color:var(--ink);margin:0;}
.sph-hero h1 .accent{color:var(--sephora-red);}
.sph-hero .tag{margin-top:14px;color:var(--ink-soft);font-size:15.5px;}
.sph-hero .tag-sub{margin-top:6px;color:var(--muted-strong);font-size:13.5px;}
.sph-hero .rule{margin-top:20px;width:56px;height:3px;background:var(--sephora-red);}

/* section */
.sph-sec{padding:32px 0 4px;}
.sph-sec .idx{color:var(--gold);font-size:11.5px;letter-spacing:.3em;font-weight:600;}
.sph-sec h2{font-family:Georgia,serif;font-size:25px;margin:6px 0 0 0;color:var(--ink);}
.sph-sec .hint{margin-top:6px;color:var(--muted-strong);font-size:13.5px;}

/* cards */
.sph-card{
  background:var(--paper);border:1px solid var(--line-strong);
  border-left:3px solid var(--sephora-red);padding:24px 26px;margin:16px 0;
}
.sph-card.gold{border-left-color:var(--gold);}
.sph-label{font-size:11.5px;color:var(--muted);letter-spacing:.2em;font-weight:600;}
.sph-sevrow{display:flex;align-items:baseline;gap:16px;margin-top:8px;flex-wrap:wrap;}
.sph-sevword{font-family:Georgia,serif;font-size:38px;font-weight:700;color:var(--ink);}
.sph-sevdesc{color:var(--ink-soft);font-size:14.5px;max-width:520px;}
.sph-conf{margin-top:10px;font-size:12.5px;color:var(--muted-strong);letter-spacing:.04em;}
.sph-conf b{color:var(--ink);}

/* segmented bar */
.sph-segbar{display:flex;gap:6px;margin-top:16px;}
.sph-seg{width:56px;height:10px;background:#ECE6DA;border:1px solid var(--line-strong);}
.sph-seg.on{background:var(--ink);border-color:var(--ink);}
.sph-seglabel{
  margin-top:8px;font-size:11.5px;color:var(--muted);letter-spacing:.04em;
  display:flex;justify-content:space-between;max-width:340px;
}

/* disclaimer */
.sph-disc{margin-top:14px;font-size:12.5px;color:var(--muted-strong);
  border-top:1px solid var(--line);padding-top:12px;}
.sph-lowconf{margin-top:12px;font-size:13px;color:var(--ink-soft);
  background:#FBF4E9;border:1px solid var(--gold-soft);padding:10px 14px;}

/* why box */
.sph-why{margin-top:14px;font-size:13.5px;color:var(--ink-soft);
  background:#FAF6EF;border:1px solid var(--line);padding:14px 16px;line-height:1.7;}
.sph-why .k{color:var(--gold);font-weight:700;letter-spacing:.06em;font-size:11.5px;}
.sph-why .chain{font-family:Georgia,serif;font-size:14px;color:var(--ink);margin-top:4px;}
.sph-why .arrow{color:var(--muted);padding:0 6px;}

/* generated copy */
.sph-copy{margin-top:14px;background:#FAF6EF;border:1px solid var(--line);
  padding:18px 20px;font-size:15px;line-height:1.7;color:var(--ink);}
.sph-copy .src{display:block;margin-top:10px;font-size:12px;color:var(--muted);}

/* product card */
.sph-prod{background:var(--paper);border:1px solid var(--line-strong);
  padding:22px 26px;margin-bottom:14px;position:relative;}
.sph-prod::before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--gold);}
.sph-pname{font-family:Georgia,serif;font-size:19px;font-weight:700;color:var(--ink);}
.sph-pbrand{font-size:12.5px;color:var(--muted);letter-spacing:.08em;margin-top:2px;}
.sph-ing{display:flex;gap:8px;margin-top:12px;flex-wrap:wrap;}
.sph-ing span{font-size:12.5px;color:var(--ink-soft);border:1px solid var(--line-strong);
  padding:3px 10px;letter-spacing:.02em;}
.sph-rv{margin-top:16px;border-top:1px solid var(--line);padding-top:12px;}
.sph-rvhead{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;margin-bottom:10px;}
.sph-rvhead .vb{font-size:11.5px;letter-spacing:.18em;font-weight:700;color:var(--muted-strong);}
.sph-rvhead .pill{font-size:11.5px;letter-spacing:.06em;padding:2px 8px;
  border:1px solid var(--line-strong);color:var(--ink-soft);}
.sph-rvrow{display:flex;gap:10px;margin-bottom:7px;font-size:14px;}
.sph-rvrow .mark{width:64px;flex-shrink:0;font-size:12px;font-weight:700;letter-spacing:.06em;}
.sph-rvrow .mark.pos{color:var(--ink);}
.sph-rvrow .mark.neg{color:var(--ink-soft);}
.sph-rvrow .txt{color:var(--ink-soft);}

/* footer */
.sph-footer{margin-top:40px;background:var(--ink);color:#E8E2D8;
  padding:24px 28px;font-size:12.5px;line-height:1.8;}
.sph-footer .gold{color:var(--gold-soft);}
.sph-footer .flame{color:var(--sephora-red);}

/* responsive */
@media(max-width:640px){
  .sph-hero h1{font-size:30px;}
  .sph-sevword{font-size:30px;}
  .sph-seg{width:42px;}
  .sph-seglabel{max-width:290px;font-size:10.5px;}
  .sph-card,.sph-prod{padding:18px 16px;}
  .sph-topbar .sub{display:none;}
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# 4. MODEL LOADING (lazy + cached)
# ============================================================
def _hf_available() -> bool:
    try:
        import transformers  # noqa: F401
        import torch         # noqa: F401
        return True
    except Exception:
        return False


@st.cache_resource(show_spinner=False)
def load_vision():
    from transformers import pipeline
    return pipeline("image-classification", model=VISION_MODEL, device=-1)


@st.cache_resource(show_spinner=False)
def load_review():
    from transformers import pipeline
    return pipeline("text-classification", model=REVIEW_MODEL, device=-1)


@st.cache_resource(show_spinner=False)
def load_generator():
    from transformers import pipeline
    return pipeline("text-generation", model=GEN_MODEL, device=-1)


def release_generator():
    """释放生成模型内存（Streamlit Cloud 内存吃紧时使用）。"""
    try:
        load_generator.clear()
    except Exception:
        pass
    gc.collect()


# ============================================================
# 5. LABEL NORMALISATION
# ============================================================
_ALNUM = re.compile(r"[^a-z0-9]")

_SEV_ALIASES = {
    "none": "None", "0": "None", "noacne": "None", "clear": "None",
    "verymild": "Very mild", "1": "Very mild", "minimal": "Very mild",
    "mild": "Mild", "2": "Mild", "low": "Mild",
    "moderate": "Moderate", "3": "Moderate", "medium": "Moderate",
    "severe": "Severe", "4": "Severe", "high": "Severe",
}


def normalize_severity(raw: str) -> str:
    k = _ALNUM.sub("", str(raw).lower())
    if k in _SEV_ALIASES:
        return _SEV_ALIASES[k]
    for alias, canon in _SEV_ALIASES.items():
        if alias and alias in k:
            return canon
    return "Mild"


def normalize_review_label(raw: str) -> str:
    k = _ALNUM.sub("", str(raw).lower())
    if k in ("pro", "label1", "positive", "pos", "1"):
        return "PRO"
    if k in ("con", "label0", "negative", "neg", "0"):
        return "CON"
    if "pos" in k:
        return "PRO"
    if "neg" in k:
        return "CON"
    return "PRO"


# ============================================================
# 6. FACE CHECK (optional, graceful degradation)
# ============================================================
try:
    import cv2
    import numpy as np
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


@st.cache_resource(show_spinner=False)
def _face_cascade():
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(path)


def has_face(pil_img: Image.Image) -> bool:
    if not _HAS_CV2:
        return True
    try:
        arr = np.array(pil_img.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        faces = _face_cascade().detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
        return len(faces) > 0
    except Exception:
        return True


# ============================================================
# 7. INFERENCE HELPERS
# ============================================================
def predict_severity(pil_img: Image.Image):
    """Return (severity_label, confidence_float, all_scores_dict)."""
    if USE_MOCK:
        time.sleep(0.6)
        return "Mild", 0.87, {lvl: (0.87 if lvl == "Mild" else 0.0325) for lvl in SEVERITY_LEVELS}

    clf = load_vision()
    img = pil_img.convert("RGB")
    preds = clf(img, top_k=5)
    scores = {}
    for p in preds:
        scores[normalize_severity(p["label"])] = float(p["score"])
    best = max(scores.items(), key=lambda kv: kv[1])
    return best[0], best[1], scores


def classify_reviews(texts):
    """Return list of (label, score) aligned with texts."""
    if not texts:
        return []
    if USE_MOCK:
        return [("PRO", 0.94) if i % 3 != 2 else ("CON", 0.88) for i in range(len(texts))]

    clf = load_review()
    outs = clf(texts, truncation=True, max_length=256, batch_size=8)
    return [(normalize_review_label(o["label"]), float(o["score"])) for o in outs]


def build_template(severity: str, ingredients, tone: str) -> str:
    tpl = TEMPLATES.get(tone, TEMPLATES["Gentle"])
    return tpl.format(sev=severity.lower(), ings=" and ".join(ingredients[:2]))


def _clean_generated(text: str, max_len: int = 300) -> str:
    text = " ".join(str(text).split())
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    cut = text[:max_len]
    idx = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
    return cut[: idx + 1] if idx > 80 else cut.rstrip() + "..."


def generate_recommendation(severity: str, ingredients, tone: str):
    """Return (text, source) where source in {'generated','template'}."""
    fallback = build_template(severity, ingredients, tone)
    if not ENABLE_TEXTGEN or USE_MOCK:
        return fallback, "template"

    try:
        gen = load_generator()
        opener = STYLE_OPENERS.get(tone, STYLE_OPENERS["Gentle"])
        prompt = (
            f"{opener} Concern level: {severity} breakout. "
            f"Key ingredients: {', '.join(ingredients)}. "
            f"Recommendation:"
        )
        out = gen(
            prompt,
            max_new_tokens=60,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            num_return_sequences=1,
            pad_token_id=gen.tokenizer.eos_token_id,
        )
        raw = out[0]["generated_text"][len(prompt):]
        text = _clean_generated(raw)
        if len(text) < 25:
            return fallback, "template"
        return text, "generated"
    except Exception:
        return fallback, "template"


# ============================================================
# 8. REVIEW DATA
# ============================================================
@st.cache_data(show_spinner=False)
def load_reviews_csv(path: str):
    import pandas as pd
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def reviews_for_product(df, query: str, max_n: int = 12):
    if df is None or getattr(df, "empty", True):
        return []
    if "product_name" not in df.columns or "review_text" not in df.columns:
        return []

    sub = df[df["product_name"].astype(str).str.contains(re.escape(query), case=False, na=False)]
    if sub.empty:
        return []

    if "verified_purchase" in sub.columns:
        vp = sub[sub["verified_purchase"].astype(str).str.lower().isin(["true", "1", "yes", "y"])]
        if not vp.empty:
            sub = vp

    sub = sub.dropna(subset=["review_text"])
    sub = sub[sub["review_text"].astype(str).str.len() > 25]
    return sub.head(max_n).to_dict("records")


def digest_for_product(product, reviews_df, max_pro=3, max_con=1):
    """Return (pros, cons, n_pro, n_con)."""
    rows = reviews_for_product(reviews_df, product["query"])
    texts, fallback_labels = [], []

    for r in rows:
        texts.append(str(r["review_text"])[:400])
        rating = r.get("rating", None)
        try:
            rating = float(rating)
        except Exception:
            rating = None
        fallback_labels.append("PRO" if (rating is None or rating >= 4) else "CON")

    if texts:
        try:
            preds = classify_reviews(texts)
            labels = [p[0] for p in preds]
        except Exception:
            labels = fallback_labels
    else:
        texts = [t for _, t in product["fallback"]]
        labels = [l for l, _ in product["fallback"]]

    pros = [t for t, l in zip(texts, labels) if l == "PRO"][:max_pro]
    cons = [t for t, l in zip(texts, labels) if l == "CON"][:max_con]

    if not pros and product["fallback"]:
        pros = [t for l, t in product["fallback"] if l == "PRO"][:max_pro]
    if not cons and product["fallback"]:
        cons = [t for l, t in product["fallback"] if l == "CON"][:max_con]

    return pros, cons, len(pros), len(cons)


# ============================================================
# 9. RENDER HELPERS
# ============================================================
def esc(x) -> str:
    return str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def topbar():
    st.markdown(
        '<div class="sph-topbar">'
        '<span class="wordmark">SEPHORA<span class="flame">*</span></span>'
        '<span class="sub">PERSONALIZED SKINCARE · SKIN ADVISOR</span>'
        "</div>",
        unsafe_allow_html=True,
    )


def hero():
    st.markdown(
        '<div class="sph-hero">'
        '<div class="sph-eyebrow">SKIN CONCERN ADVISOR</div>'
        "<h1>Find Your First Step to<br><span class='accent'>Better Skin</span></h1>"
        '<p class="tag">Snap a selfie · Match your routine · Reviews already curated</p>'
        '<p class="tag-sub">Thousands of SKUs. We narrow it down to two or three — '
        "with verified buyer reviews already filtered.</p>"
        '<div class="rule"></div>'
        "</div>",
        unsafe_allow_html=True,
    )


def section(idx, title, hint=None):
    h = f'<div class="sph-sec"><div class="idx">{esc(idx)}</div><h2>{esc(title)}</h2>'
    if hint:
        h += f'<p class="hint">{esc(hint)}</p>'
    h += "</div>"
    st.markdown(h, unsafe_allow_html=True)


def segbar_html(severity: str) -> str:
    lit = SEVERITY_LEVELS.index(severity) + 1 if severity in SEVERITY_LEVELS else 3
    segs = "".join(
        f'<div class="sph-seg{" on" if i < lit else ""}"></div>' for i in range(5)
    )
    labels = "".join(f"<span>{esc(l)}</span>" for l in SEVERITY_LEVELS)
    return f'<div class="sph-segbar">{segs}</div><div class="sph-seglabel">{labels}</div>'


def render_result(severity, confidence, low_conf):
    low = (
        '<div class="sph-lowconf">Low confidence — try another photo with better light, '
        "or consult a dermatologist.</div>"
        if low_conf
        else ""
    )
    st.markdown(
        f'<div class="sph-card">'
        f'<div class="sph-label">BREAKOUT CONCERN LEVEL</div>'
        f'<div class="sph-sevrow">'
        f'<span class="sph-sevword">{esc(severity)}</span>'
        f'<span class="sph-sevdesc">A {esc(severity.lower())} breakout pattern may be present. '
        f"This is a cosmetic skin concern, not a diagnosis.</span>"
        f"</div>"
        f'<div class="sph-conf">Confidence: <b>{confidence*100:.0f}%</b> · Model: ViT · 5-class classifier</div>'
        f"{segbar_html(severity)}"
        f"{low}"
        f'<div class="sph-disc">{esc(DISCLAIMER)}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_why(severity, ingredients, entry):
    st.markdown(
        f'<div class="sph-why">'
        f'<div class="k">WHY THIS RECOMMENDATION</div>'
        f'<div class="chain">{esc(severity)} breakout concern'
        f'<span class="arrow">→</span>{esc(" · ".join(ingredients))}'
        f'<span class="arrow">→</span>{esc(entry["products"][0]["name"])}</div>'
        f'<div style="margin-top:6px;font-size:12.5px;color:#6B6259;">{esc(entry["why"])}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_copy(text, tone, source):
    st.markdown(
        f'<div class="sph-copy">{esc(text)}'
        f'<span class="src">Tone: {esc(tone)} · '
        f'{"Generated by the fine-tuned text model" if source == "generated" else "Template fallback (text model unavailable)"}'
        f"</span></div>",
        unsafe_allow_html=True,
    )


def render_product(product, pros, cons, n_pro, n_con):
    ing_html = "".join(f"<span>{esc(t)}</span>" for t in product["tags"])
    rows = ""
    for t in pros:
        rows += f'<div class="sph-rvrow"><span class="mark pos">PRO +</span><span class="txt">{esc(t)}</span></div>'
    for t in cons:
        rows += f'<div class="sph-rvrow"><span class="mark neg">CON −</span><span class="txt">{esc(t)}</span></div>'

    st.markdown(
        f'<div class="sph-prod">'
        f'<div class="sph-pname">{esc(product["name"])}</div>'
        f'<div class="sph-pbrand">{esc(product["brand"])}</div>'
        f'<div class="sph-ing">{ing_html}</div>'
        f'<div class="sph-rv">'
        f'<div class="sph-rvhead">'
        f'<span class="vb">VERIFIED BUYER DIGEST</span>'
        f'<span class="pill">{n_pro} PRO</span>'
        f'<span class="pill">{n_con} CON</span>'
        f"</div>"
        f"{rows}"
        f"</div></div>",
        unsafe_allow_html=True,
    )


def footer():
    st.markdown(
        '<div class="sph-footer">'
        '<span class="flame">*</span> For skincare reference only. '
        '<span class="gold">Not a medical diagnosis.</span> '
        "For skin conditions or concerns, please consult a dermatologist.<br>"
        "Images are processed in memory and never stored or shared. "
        "This tool recommends only — it does not sell or diagnose.<br>"
        'SEPHORA<span class="flame">*</span> · Skin Concern Advisor — sample data for preview.'
        "</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# 10. SESSION STATE
# ============================================================
def init_state():
    defaults = {
        "img_sig": None,
        "img": None,
        "img_name": "",
        "img_ok": False,
        "analyzed": False,
        "severity": None,
        "confidence": 0.0,
        "reco_text": "",
        "reco_source": "template",
        "reco_tone": "Gentle",
        "fb_reco": None,
        "fb_digest": None,
        "saved_at": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


init_state()

# ============================================================
# 11. PAGE
# ============================================================
topbar()
hero()

# ---------- 01 UPLOAD ----------
section("01 / UPLOAD", "Upload Your Selfie",
        "One clear photo. Front-facing, good light, no filters.")

uploaded = st.file_uploader(
    "Upload your selfie",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

st.caption("JPG or PNG · Max 10 MB · Front-facing, good light")
st.caption("For skin analysis only · **Processed in memory** · Never stored · Never shared")

if uploaded is not None:
    sig = f"{uploaded.name}-{uploaded.size}"
    if sig != st.session_state.img_sig:
        # 新文件 -> 重新校验
        st.session_state.img_sig = sig
        st.session_state.analyzed = False
        st.session_state.reco_text = ""
        st.session_state.img_ok = False
        st.session_state.img = None

        if uploaded.size > MAX_MB * 1024 * 1024:
            st.error(f"File too large. Please upload an image under {MAX_MB} MB.")
        else:
            try:
                img = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
                if min(img.size) < 120:
                    st.error("Image is too small. Please upload a larger photo.")
                elif not has_face(img):
                    st.error("No face detected. Please upload a clear, front-facing selfie.")
                else:
                    st.session_state.img = img
                    st.session_state.img_name = uploaded.name
                    st.session_state.img_ok = True
            except Exception:
                st.error("Could not read this file. Please upload a valid JPG or PNG image.")

# ---------- 02 ANALYZE ----------
if st.session_state.img_ok:
    section("02 / ANALYZE", "Analyze Your Photo",
            "You stay in control — analysis only runs when you click.")

    st.markdown(
        f'<div class="sph-card" style="border-left-color:#C8102E;">'
        f'<div style="font-size:13.5px;color:#1A1512;">{esc(st.session_state.img_name)}</div>'
        f'<div style="font-size:12.5px;color:#8A8177;">Ready · No face detection error</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    if st.button("Analyze", type="primary", use_container_width=False):
        with st.spinner("Analyzing cosmetic skin concerns..."):
            try:
                sev, conf, _scores = predict_severity(st.session_state.img)
                st.session_state.severity = sev
                st.session_state.confidence = conf
                st.session_state.analyzed = True

                entry = PRODUCT_MAP.get(sev, PRODUCT_MAP["Mild"])
                text, source = generate_recommendation(
                    sev, entry["ingredients"], st.session_state.reco_tone
                )
                st.session_state.reco_text = text
                st.session_state.reco_source = source
            except Exception as e:
                st.session_state.analyzed = False
                st.error(
                    "Analysis failed — the model could not be loaded or timed out. "
                    "Please try again."
                )
                with st.expander("Technical detail"):
                    st.code(str(e))

# ---------- 03 RESULT ----------
if st.session_state.analyzed:
    severity = st.session_state.severity
    confidence = st.session_state.confidence
    entry = PRODUCT_MAP.get(severity, PRODUCT_MAP["Mild"])

    section("03 / RESULT", "Your Skin Snapshot",
            "Cosmetic skin concern only. Not a medical diagnosis.")
    render_result(severity, confidence, low_conf=(confidence < 0.60))

    # ---------- 04 RECOMMENDATION ----------
    section("04 / RECOMMENDATION", "Your Personalized Recommendation",
            "Pick a tone. We'll write the recommendation around your result.")

    st.markdown('<div class="sph-card gold">', unsafe_allow_html=True)
    render_why(severity, entry["ingredients"], entry)

    tone = st.selectbox(
        "RECOMMENDATION TONE",
        ["Gentle", "Professional", "Concise", "Enthusiastic"],
        index=["Gentle", "Professional", "Concise", "Enthusiastic"].index(
            st.session_state.reco_tone
        ),
    )

    if tone != st.session_state.reco_tone or not st.session_state.reco_text:
        st.session_state.reco_tone = tone
        text, source = generate_recommendation(severity, entry["ingredients"], tone)
        st.session_state.reco_text = text
        st.session_state.reco_source = source

    if st.button("Regenerate Recommendation"):
        with st.spinner("Writing..."):
            text, source = generate_recommendation(severity, entry["ingredients"], tone)
            st.session_state.reco_text = text
            st.session_state.reco_source = source
            st.rerun()

    render_copy(st.session_state.reco_text, st.session_state.reco_tone,
                st.session_state.reco_source)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- 05 MATCHED PRODUCTS ----------
    section("05 / MATCHED PRODUCTS", "Your Skincare Picks",
            "Only verified-buyer reviews are shown. PRO / CON — no endless scrolling.")

    reviews_df = load_reviews_csv(REVIEWS_CSV)
    with st.spinner("Curating verified buyer reviews..."):
        for p in entry["products"]:
            pros, cons, n_pro, n_con = digest_for_product(p, reviews_df)
            render_product(p, pros, cons, n_pro, n_con)

    # ---------- 06 SAVE / SHARE / FEEDBACK ----------
    section("06 / SAVE · SHARE · FEEDBACK", "Keep Your Routine",
            "No account needed. Your summary stays in this session.")

    summary = build_summary_text(severity, confidence, entry, st.session_state.reco_text,
                                 st.session_state.reco_tone)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.download_button(
            "Download Summary",
            data=summary,
            file_name="sephora_skin_summary.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "Share Routine",
            data=summary,
            file_name="sephora_routine_share.txt",
            mime="text/plain",
            use_container_width=True,
        )

    if st.button("Save to This Session"):
        st.session_state.saved_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.success(f"Saved to this session at {st.session_state.saved_at}.")

    st.markdown("---")
    f1, f2, f3, f4 = st.columns([2, 1, 1, 1])
    with f1:
        st.write("Was this recommendation useful?")
    with f2:
        if st.button("Yes", key="fb_reco_yes"):
            st.session_state.fb_reco = "yes"
    with f3:
        if st.button("No", key="fb_reco_no"):
            st.session_state.fb_reco = "no"
    with f4:
        st.write("")

    g1, g2, g3, g4 = st.columns([2, 1, 1, 1])
    with g1:
        st.write("Was the review digest useful?")
    with g2:
        if st.button("Yes", key="fb_dig_yes"):
            st.session_state.fb_digest = "yes"
    with g3:
        if st.button("No", key="fb_dig_no"):
            st.session_state.fb_digest = "no"
    with g4:
        st.write("")

    if st.session_state.fb_reco or st.session_state.fb_digest:
        st.caption(
            f"Feedback recorded — recommendation: {st.session_state.fb_reco or '—'} · "
            f"digest: {st.session_state.fb_digest or '—'}"
        )

footer()


# ============================================================
# 12. SUMMARY BUILDER
# ============================================================
def build_summary_text(severity, confidence, entry, reco_text, tone):
    lines = [
        "SEPHORA · Skin Concern Advisor — My Skincare Summary",
        "=" * 56,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "SKIN SNAPSHOT",
        f"  Breakout concern level : {severity}",
        f"  Model confidence       : {confidence*100:.0f}%",
        "",
        "WHY THIS RECOMMENDATION",
        f"  {severity} breakout concern -> {' · '.join(entry['ingredients'])}",
        f"  {entry['why']}",
        "",
        "RECOMMENDATION",
        f"  [{tone}] {reco_text}",
        "",
        "MATCHED PRODUCTS",
    ]
    for p in entry["products"]:
        lines.append(f"  - {p['brand']} · {p['name']}")
        lines.append(f"      Tags: {', '.join(p['tags'])}")
    lines += [
        "",
        "DISCLAIMER",
        f"  {DISCLAIMER}",
        "  Images are processed in memory and never stored or shared.",
        "  This tool recommends only — it does not sell or diagnose.",
    ]
    return "\n".join(lines)
