"""
ISOM5240 Group Project — Sephora Skin Concern Advisor
Streamlit App: ViT acne severity + DistilBERT review sentiment
"""

import streamlit as st
from PIL import Image
from transformers import pipeline

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Skin Concern Advisor",
    page_icon="✨",
    layout="centered",
)

# ============================================================
# Model loading (cached)
# ============================================================
@st.cache_resource
def load_vision_pipeline():
    """Load fine-tuned ViT acne severity classifier."""
    return pipeline(
        "image-classification",
        model="jiefangziyou/vit-acne-severity",  # ← 你的模型
    )


@st.cache_resource
def load_review_pipeline():
    """Load pre-trained DistilBERT sentiment classifier for review PRO/CON."""
    return pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )


# ============================================================
# Mapping tables
# ============================================================
SEVERITY_DESCRIPTIONS = {
    "None": "No obvious breakout pattern detected.",
    "Mild": "A mild breakout pattern may be present.",
    "Moderate": "A moderate breakout pattern may be present.",
    "Severe": "A more noticeable breakout pattern may be present.",
}

INGREDIENT_MAP = {
    "None": ["Gentle cleanser", "Hyaluronic Acid", "SPF 30+"],
    "Mild": ["Salicylic Acid", "Niacinamide", "Azelaic Acid"],
    "Moderate": ["Salicylic Acid", "Niacinamide", "Adapalene"],
    "Severe": ["Benzoyl Peroxide", "Adapalene", "Salicylic Acid"],
}

# ============================================================
# UI
# ============================================================
st.title("✨ Skin Concern Advisor")
st.caption(
    "Snap a selfie · Match your routine · Reviews already curated"
)

st.markdown(
    """
    > **For skincare reference only. Not a medical diagnosis.**
    > Please consult a dermatologist for skin conditions.
    """
)

# ---------- Step 1: Upload ----------
st.header("01 · Upload Your Selfie")
uploaded = st.file_uploader(
    "Choose a front-facing photo (JPG / PNG, max 10 MB)",
    type=["jpg", "jpeg", "png"],
)

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded photo", use_container_width=True)

    # ---------- Step 2: Analyze ----------
    if st.button("🔍 Analyze My Photo", type="primary"):
        with st.spinner("Analyzing cosmetic skin concerns..."):
            vision_clf = load_vision_pipeline()
            predictions = vision_clf(image)

        # Top prediction
        top = predictions[0]
        severity = top["label"]
        confidence = top["score"]

        # ---------- Step 3: Result ----------
        st.header("02 · Your Skin Snapshot")
        st.subheader(f"Breakout Concern Level: **{severity}**")
        st.write(SEVERITY_DESCRIPTIONS.get(severity, ""))

        st.metric("Confidence", f"{confidence:.1%}")

        # Low-confidence warning
        if confidence < 0.60:
            st.warning(
                "Low confidence — try another photo with better light, "
                "or consult a dermatologist."
            )

        # Top-3 probabilities
        with st.expander("See full model output (top-3)"):
            for p in predictions[:3]:
                st.write(f"- **{p['label']}**: {p['score']:.2%}")

        # ---------- Step 4: Recommendation ----------
        st.header("03 · Your Personalized Recommendation")
        ingredients = INGREDIENT_MAP.get(severity, [])
        st.markdown(
            f"**Why this recommendation**  \n"
            f"{severity} breakout concern → "
            + " · ".join(ingredients)
        )
        st.info(
            "Ingredients commonly used in cosmetic skincare "
            "for this concern level."
        )

        # ---------- Step 5: Review Digest ----------
        st.header("04 · Verified Buyer Digest (Demo)")
        st.caption(
            "Paste a verified buyer review below to see if it is PRO or CON."
        )

        review_text = st.text_area(
            "Paste a review here:",
            placeholder="e.g., This serum cleared my acne in two weeks!",
        )

        if review_text.strip():
            review_clf = load_review_pipeline()
            result = review_clf(review_text)[0]
            label = result["label"]      # POSITIVE / NEGATIVE
            score = result["score"]

            if label == "POSITIVE":
                st.success(f"**PRO +** (confidence {score:.1%})")
            else:
                st.error(f"**CON −** (confidence {score:.1%})")

        # ---------- Step 6: Feedback ----------
        st.header("05 · Was this useful?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Yes"):
                st.success("Thanks for your feedback!")
        with col2:
            if st.button("👎 No"):
                st.info("Thanks — we'll keep improving.")

# ---------- Footer ----------
st.divider()
st.caption(
    "For skincare reference only. Not a medical diagnosis. "
    "Images are processed in memory and never stored. "
    "This tool recommends only — it does not sell or diagnose."
)
