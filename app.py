
import os
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from scipy.sparse import hstack
from lime.lime_text import LimeTextExplainer

# -----------------------------
# APPLICATION CONFIGURATION
# -----------------------------

st.set_page_config(
    page_title="Explainable SMS Spam Detection",
    page_icon="🛡️",
    layout="centered"
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(APP_DIR, "application_assets")

THRESHOLD = 0.57

# -----------------------------
# LOAD TRAINED MODEL
# -----------------------------

@st.cache_resource
def load_components():

    model = joblib.load(
        os.path.join(
            ASSETS_DIR,
            "Word_Character_Fusion_LR_Standard.pkl"
        )
    )

    word_vectorizer = joblib.load(
        os.path.join(
            ASSETS_DIR,
            "Validation_Word_TFIDF_Vectorizer.pkl"
        )
    )

    char_vectorizer = joblib.load(
        os.path.join(
            ASSETS_DIR,
            "Validation_Character_TFIDF_Vectorizer.pkl"
        )
    )

    return model, word_vectorizer, char_vectorizer


model, word_vectorizer, char_vectorizer = load_components()

explainer = LimeTextExplainer(
    class_names=["Ham", "Spam"],
    random_state=42
)

if "prediction_history" not in st.session_state:

    history_file = os.path.join(APP_DIR, "prediction_history.csv")

    if os.path.exists(history_file):
        try:
            saved_history = pd.read_csv(history_file)
            st.session_state.prediction_history = saved_history.to_dict("records")
        except Exception as e:
            st.session_state.prediction_history = []
            st.warning(f"Previous history could not be loaded: {e}")
    else:
        st.session_state.prediction_history = []


# -----------------------------
# PREDICTION FUNCTION
# -----------------------------

def predict_proba(texts):

    word_features = word_vectorizer.transform(texts)
    char_features = char_vectorizer.transform(texts)

    combined_features = hstack(
        [word_features, char_features]
    )

    return model.predict_proba(combined_features)


# -----------------------------
# APPLICATION INTERFACE
# -----------------------------

st.title("🛡️ SMS Spam Detection")
st.subheader("Robust and Explainable Cross-Lingual SMS Spam Detection")

st.write(
    "Enter an SMS message below to identify whether it is "
    "Spam or Ham and understand the prediction using LIME."
)

st.info(
    "Research demonstration: The model is trained using English SMS data. "
    "Predictions on Tanglish and Romanized Sinhala are experimental "
    "zero-shot evaluations."
)

language = st.selectbox(
    "Select SMS Language",
    [
        "English",
        "Tanglish (Romanized Tamil)",
        "Singlish (Romanized Sinhala)"
    ]
)

placeholders = {
    "English": "Example: Congratulations! You have won a free prize...",
    "Tanglish (Romanized Tamil)": "Example: Ungaluku prize win aagiyullathu...",
    "Singlish (Romanized Sinhala)": "Example: Obata prize ekak labila thiyenawa..."
}

sms_text = st.text_area(
    "Enter your SMS message",
    height=150,
    placeholder=placeholders[language]
)

if st.button("Analyze SMS", type="primary", use_container_width=True):

    if not sms_text.strip():

        st.warning("Please enter an SMS message first.")

    else:

        with st.spinner("Analyzing message and generating explanation..."):

            probabilities = predict_proba([sms_text])[0]

            ham_probability = float(probabilities[0])
            spam_probability = float(probabilities[1])

            if spam_probability >= THRESHOLD:
                prediction = "Spam"
            elif spam_probability >= 0.50:
                prediction = "Needs Review"
            else:
                prediction = "Ham"

            record = {
                "Date_Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Language": language,
                "SMS_Message": sms_text,
                "Prediction": prediction,
                "Ham_Probability": round(ham_probability * 100, 2),
                "Spam_Probability": round(spam_probability * 100, 2)
            }

            st.session_state.prediction_history.append(record)

            history_file = os.path.join(APP_DIR, "prediction_history.csv")

            history_df = pd.DataFrame(st.session_state.prediction_history)
            history_df.to_csv(history_file, index=False)



        st.divider()

        st.subheader("Prediction Result")

        st.caption(f"Selected SMS Language: {language}")

        if prediction == "Spam":
            st.error("🚨 SPAM MESSAGE DETECTED")
        elif prediction == "Needs Review":
            st.warning("⚠️ SUSPICIOUS MESSAGE — NEEDS REVIEW")
        else:
            st.success("✅ HAM MESSAGE DETECTED")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Ham Probability",
                f"{ham_probability * 100:.2f}%"
            )

        with col2:
            st.metric(
                "Spam Probability",
                f"{spam_probability * 100:.2f}%"
            )

        st.progress(spam_probability)

        st.caption(
            f"Spam decision threshold: {THRESHOLD * 100:.0f}% "
            "(selected using English validation data)"
        )

        st.info(
            "Decision rule: Spam >= 57%; Needs Review = 50% to below 57%; "
            "Ham < 50%. Needs Review is an application warning, not a new "
            "research model class. Thresholds are not separately calibrated "
            "for Tanglish or Singlish."
        )

        # -----------------------------
        # LIME EXPLANATION
        # -----------------------------

        st.divider()
        st.subheader("🔍 Explainable AI — LIME")

        with st.spinner("Generating LIME explanation..."):

            explanation = explainer.explain_instance(
                sms_text,
                predict_proba,
                num_features=10,
                num_samples=1000,
                labels=[0, 1]
            )

            explanation_list = explanation.as_list(
                label=1
            )

        st.write(
            "The following words show their local contribution "
            "towards the Spam class for this message."
        )

        fig = explanation.as_pyplot_figure(label=1)

        fig.set_size_inches(9, 5)

        st.pyplot(fig, use_container_width=True)

        plt.close(fig)

        st.caption(
            "LIME provides a local explanation for this individual "
            "prediction. Bar values are contribution weights, not probabilities."
        )

        with st.expander("View LIME word contributions"):

            for word, contribution in explanation_list:

                st.write(
                    f"**{word}** : {contribution:.4f}"
                )

# -----------------------------
# MODEL INFORMATION
# -----------------------------

with st.expander("About this research model"):

    st.write("**Algorithm:** Logistic Regression")

    st.write(
        "**Feature representation:** Word TF-IDF + Character TF-IDF"
    )

    st.write(
        "**Fusion approach:** Word and character feature fusion"
    )

    st.write(
        "**Training language:** English"
    )

    st.write(
        "**Evaluation languages:** English, Tanglish and Romanized Sinhala"
    )

    st.write(
        "**Explainability:** LIME"
    )

    st.write(
        "**Decision threshold:** 0.57, selected using English validation data"
    )

st.caption(
    "Academic research demonstration — predictions should not be "
    "treated as a guarantee of message safety."
)


# -----------------------------
# PREDICTION HISTORY
# -----------------------------

st.divider()
st.subheader("📋 Prediction History")

st.caption(
    "Predictions are stored in this session and backed up to Google Drive "
    "in CSV format. SMS messages are included in the saved file."
)

if st.session_state.prediction_history:

    history_df = pd.DataFrame(st.session_state.prediction_history)

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    csv_data = history_df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="⬇️ Download Prediction History (CSV)",
        data=csv_data,
        file_name="SMS_Prediction_History.csv",
        mime="text/csv",
        use_container_width=True
    )

    if st.button("🗑️ Clear Prediction History", use_container_width=True):

        st.session_state.prediction_history = []

        history_file = os.path.join(APP_DIR, "prediction_history.csv")

        if os.path.exists(history_file):
            os.remove(history_file)

        st.rerun()

else:
    st.info("No predictions recorded yet. Analyze an SMS to create history.")
