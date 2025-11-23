"""
utils.py — Backend utilities for SAXA4 Capstone Streamlit app.

Contains:
- artifact loading (model, vectorizer, label encoder, context table)
- text cleaning
- ML inference + probabilities
- TF-IDF keyword extraction
- agency governance context lookup
- OpenAI policy-note generation
- end-to-end wrapper for Streamlit
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

import joblib
import numpy as np
import pandas as pd
from openai import OpenAI

# ---------------------------------------------------------------------
# Paths + lazy-loaded global caches
# ---------------------------------------------------------------------

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"

_MODEL = None
_VECTORIZER = None
_LABEL_ENCODER = None
_CONTEXT_DF = None

# ---------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------

def basic_clean(text: str) -> str:
    """Replicates the cleaning used in training/EDA."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ---------------------------------------------------------------------
# Artifact loaders (cached)
# ---------------------------------------------------------------------

def load_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = joblib.load(ARTIFACT_DIR / "log_reg_final.joblib")
    return _MODEL

def load_vectorizer():
    global _VECTORIZER
    if _VECTORIZER is None:
        _VECTORIZER = joblib.load(ARTIFACT_DIR / "vectorizer_final.joblib")
    return _VECTORIZER

def load_vectorizer():
    global _VECTORIZER
    if _VECTORIZER is None:
        _VECTORIZER = joblib.load(ARTIFACT_DIR / "vectorizer_final.joblib")

        # --- Debugging: Check if vectorizer is fitted ---
        import streamlit as st
        st.write("Vectorizer fitted?", hasattr(_VECTORIZER, "idf_"))
        # Alternatively (if you don't want UI output):
        # print("Vectorizer fitted?", hasattr(_VECTORIZER, "idf_"))
        # -----------------------------------------------

    return _VECTORIZER

def load_context_df():
    global _CONTEXT_DF
    if _CONTEXT_DF is None:
        _CONTEXT_DF = pd.read_csv(ARTIFACT_DIR / "context_table_agency_scores.csv")
    return _CONTEXT_DF

# ---------------------------------------------------------------------
# ML inference
# ---------------------------------------------------------------------

def predict_impact(text: str) -> Dict[str, Any]:
    """Clean input, vectorize, and predict impact class + probabilities."""
    model = load_model()
    vectorizer = load_vectorizer()
    le = load_label_encoder()

    cleaned = basic_clean(text)
    X_vec = vectorizer.transform([cleaned])

    probs = model.predict_proba(X_vec)[0]
    idx = int(probs.argmax())
    pred_label = le.inverse_transform([idx])[0]

    result = {
        "cleaned_input": cleaned,
        "predicted_class": pred_label,
        "predicted_index": idx,
        "confidence": round(float(probs[idx]), 4),
        "class_probabilities": {
            le.classes_[i]: round(float(p), 4) for i, p in enumerate(probs)
        },
    }
    return result

# ---------------------------------------------------------------------
# TF-IDF keywords (explainability)
# ---------------------------------------------------------------------

def extract_top_tfidf_keywords(
    model,
    vectorizer,
    text: str,
    top_n: int = 8
) -> List[str]:
    """
    Extract top TF-IDF-weighted keywords for the predicted class.
    Mirrors notebook logic.
    """
    cleaned = basic_clean(text)
    X = vectorizer.transform([cleaned])

    # pick coefficient vector for the most likely class
    probs = model.predict_proba(X)[0]
    coef = model.coef_[int(np.argmax(probs))]

    feature_names = vectorizer.get_feature_names_out()
    top_idx = np.argsort(coef)[-top_n:]
    return [feature_names[i] for i in top_idx]

# ---------------------------------------------------------------------
# Governance context lookup
# ---------------------------------------------------------------------

def get_agency_context(agency: str) -> Optional[Dict[str, Any]]:
    """Return governance context row for the selected agency."""
    df = load_context_df()
    row = df[df["3_agency"] == agency].head(1)
    if row.empty:
        return None
    return row.to_dict(orient="records")[0]

# ---------------------------------------------------------------------
# LLM prompt + generation
# ---------------------------------------------------------------------

BASE_PROMPT = """
You are a federal AI governance analyst.
Write a short policy note (6–10 sentences) for a U.S. federal AI use case.

Use the inputs below:
- Cleaned narrative text
- ML-predicted impact class (rights / safety / both / neither)
- Confidence and class probabilities
- Top TF-IDF keywords driving the model
- Agency-level governance context (governance, transparency, fairness, historical impact rates)

Goals:
1) Explain why the use case is likely classified this way.
2) Highlight potential rights/safety risks implied by the narrative & keywords.
3) Relate risks to the agency’s governance maturity.
4) Provide practical, cautious recommendations (monitoring, mitigation, documentation).

Be clear, neutral, and policy-oriented. Do not invent facts not supported by the inputs.
"""

def generate_policy_note(
    cleaned_input: str,
    predicted_class: str,
    confidence: float,
    class_probabilities: Dict[str, float],
    keywords: List[str],
    agency_context: Optional[Dict[str, Any]],
    model_name: str = "gpt-4o-mini",
) -> str:
    """
    Calls OpenAI to generate a short policy note using BASE_PROMPT.
    If OPENAI_API_KEY is missing, returns a safe fallback message.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return (
            "LLM policy note unavailable because OPENAI_API_KEY is not set. "
            "The ML prediction and governance context are shown above."
        )

    # defensive defaults
    if agency_context is None:
        agency_context = {
            "3_agency": "Unknown",
            "mean_governance": "NA",
            "mean_transparency": "NA",
            "mean_fairness": "NA",
            "n_cases": "NA",
            "pct_rights_impacting": "NA",
            "pct_safety_impacting": "NA",
        }

    prompt_filled = f"""
{BASE_PROMPT}

INPUTS
Cleaned narrative:
{cleaned_input}

ML prediction:
- predicted_class: {predicted_class}
- confidence: {confidence}
- class_probabilities: {class_probabilities}

Top TF-IDF keywords:
{keywords}

Agency governance context:
{agency_context}

Now write the policy note.
"""

    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You write concise federal policy notes."},
                {"role": "user", "content": prompt_filled},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return (
            "LLM policy note generation failed. "
            f"Error: {type(e).__name__}: {e}"
        )

# ---------------------------------------------------------------------
# End-to-end wrapper
# ---------------------------------------------------------------------

def score_row_with_llm(text: str, agency: str, model_name: str = "gpt-4o-mini") -> Dict[str, Any]:
    """
    Convenience wrapper for Streamlit:
    ML prediction + keywords + agency context + LLM note.
    """
    model = load_model()
    vectorizer = load_vectorizer()

    pred = predict_impact(text)

    keywords = extract_top_tfidf_keywords(
        model=model,
        vectorizer=vectorizer,
        text=pred["cleaned_input"],
        top_n=8
    )

    agency_ctx = get_agency_context(agency)

    note = generate_policy_note(
        cleaned_input=pred["cleaned_input"],
        predicted_class=pred["predicted_class"],
        confidence=pred["confidence"],
        class_probabilities=pred["class_probabilities"],
        keywords=keywords,
        agency_context=agency_ctx,
        model_name=model_name
    )

    return {
        **pred,
        "keywords": keywords,
        "agency_context": agency_ctx,
        "policy_note": note,
        "llm_model": model_name,
    }
