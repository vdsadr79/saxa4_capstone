import streamlit as st
from utils import (
    load_context_df,
    score_row_with_llm,
)

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Federal AI Impact Classifier – SAXA4 Capstone",
    layout="wide"
)

st.title("🔍 Federal AI Impact Classifier (SAXA4 Capstone)")
st.write(
    """
    This application analyzes a federal AI use case narrative and:
    - predicts whether the use case is **rights-impacting**, **safety-impacting**, **both**, or **neither**,  
    - surfaces **model-influential TF-IDF keywords**,  
    - and highlights **TF-IDF keywords drawn directly from this narrative**,   
    - merges **agency-level governance context**,  
    - and generates a short **policy note** using an LLM.  
    """
)

# ---------------------------------------------------------
# Load agency list
# ---------------------------------------------------------
context_df = load_context_df()
agencies = sorted(context_df["3_agency"].unique().tolist())

# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------
st.sidebar.header("Settings")
selected_agency = st.sidebar.selectbox("Select Agency", agencies)

model_choice = st.sidebar.selectbox(
    "LLM Model",
    ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4-turbo"]
)

st.sidebar.write("---")
st.sidebar.write(
    "Note: LLM output requires a valid `OPENAI_API_KEY` "
    "in Streamlit Cloud (Secrets)."
)

# ---------------------------------------------------------
# Input area
# ---------------------------------------------------------
st.subheader("📝 Enter AI Use Case Narrative")

user_text = st.text_area(
    "Paste the narrative text here:",
    height=180,
    placeholder="Describe the AI system purpose, outputs, and context..."
)

run_button = st.button("Run Analysis")

# ---------------------------------------------------------
# Run analysis
# ---------------------------------------------------------
if run_button:
    if not user_text.strip():
        st.error("Please enter narrative text before running the analysis.")
    else:
        with st.spinner("Analyzing use case..."):
            result = score_row_with_llm(
                text=user_text,
                agency=selected_agency,
                model_name=model_choice
            )

        # ---------------------------------------------------------
        # Display results
        # ---------------------------------------------------------
        st.subheader("📌 ML Prediction")
        st.write(f"**Predicted Class:** {result['predicted_class']}")
        st.write(f"**Confidence:** {result['confidence']}")
        st.json(result["class_probabilities"])

        st.subheader("🔑 TF-IDF Keywords (Model-Influential)")
        # Prefer the new field, fall back to old "keywords" for safety
        st.write(result.get("model_top_terms") or result.get("keywords", []))

        st.subheader("📝 TF-IDF Keywords (From This Narrative)")
        st.write(result.get("input_top_terms", []))
       

        st.subheader("🏛️ Agency Governance Context")
        if result["agency_context"] is None:
            st.warning("No governance context found for this agency.")
        else:
            st.json(result["agency_context"])

        st.subheader("📝 Policy Note (LLM-Generated)")
        st.write(result["policy_note"])

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.write("---")
st.caption("Built by SAXA4 – Georgetown MSBA Capstone Project © 2025")
