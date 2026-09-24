import ipaddress
import re
from pathlib import Path
from urllib.parse import urlparse

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
FEATURES_PATH = BASE_DIR / "features.pkl"

st.set_page_config(
    page_title="ShopSafe AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(99,102,241,.12), transparent 28%),
            radial-gradient(circle at 90% 10%, rgba(14,165,233,.10), transparent 24%),
            #070a12;
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2.4rem;
        padding-bottom: 3rem;
    }

    .hero {
        text-align: center;
        padding: .4rem 0 1.2rem;
    }

    .badge {
        display: inline-block;
        padding: .35rem .7rem;
        border: 1px solid rgba(148,163,184,.22);
        border-radius: 999px;
        background: rgba(15,23,42,.65);
        color: #cbd5e1;
        font-size: .78rem;
        font-weight: 600;
        letter-spacing: .04em;
    }

    .hero h1 {
        margin: .9rem 0 .35rem;
        font-size: clamp(2.2rem, 5vw, 4rem);
        line-height: 1;
        font-weight: 800;
        letter-spacing: -.045em;
    }

    .hero h1 span {
        background: linear-gradient(90deg, #a78bfa, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero p {
        color: #94a3b8;
        max-width: 720px;
        margin: 0 auto;
        line-height: 1.65;
    }

    .score-card {
        background: rgba(2,6,23,.55);
        border: 1px solid rgba(148,163,184,.11);
        border-radius: 18px;
        padding: 1rem;
        min-height: 150px;
    }

    .score-kicker {
        color: #94a3b8;
        font-size: .78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: .08em;
    }

    .score-value {
        font-size: 2.2rem;
        font-weight: 800;
        margin: .35rem 0;
    }

    .tiny {
        color: #64748b;
        font-size: .74rem;
        line-height: 1.55;
    }

    div[data-testid="stTextInput"] input {
        background: rgba(2,6,23,.70) !important;
        border: 1px solid rgba(148,163,184,.20) !important;
        border-radius: 14px !important;
        color: #f8fafc !important;
        min-height: 52px !important;
    }

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        width: 100%;
        min-height: 50px;
        border: 0;
        border-radius: 14px;
        font-weight: 700;
        background: linear-gradient(90deg, #6366f1, #0ea5e9);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH.name}")
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing feature list: {FEATURES_PATH.name}")

    model = joblib.load(MODEL_PATH)
    feature_order = joblib.load(FEATURES_PATH)

    if list(model.classes_) != [0, 1]:
        raise ValueError("The model must use classes 0 and 1.")

    return model, feature_order


def normalize_url(url):
    url = url.strip()
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url
    return url


def canonicalize_url(url):
    url = normalize_url(url)
    parsed = urlparse(url)
    host = (parsed.hostname or "").rstrip(".").lower()

    if not host:
        raise ValueError("That doesn't look like a valid URL or domain.")

    labels = host.split(".")
    path_only = parsed.path in ("", "/")
    no_extra_parts = not parsed.query and not parsed.fragment

    if len(labels) == 2 and path_only and no_extra_parts and not host.startswith("www."):
        netloc = "www." + host
        if parsed.port:
            netloc = f"{netloc}:{parsed.port}"
        url = parsed._replace(netloc=netloc).geturl()
        return url, True

    return url, False


def extract_url_features(url):
    parsed = urlparse(url)
    domain = parsed.netloc.split("@")[-1].split(":")[0]

    if not domain:
        raise ValueError("That doesn't look like a valid URL or domain.")

    url_length = len(url)
    domain_length = len(domain)

    try:
        ipaddress.ip_address(domain)
        is_domain_ip = 1
    except ValueError:
        is_domain_ip = 0

    domain_parts = domain.split(".")
    no_of_subdomain = max(len(domain_parts) - 2, 0)

    letters = sum(c.isalpha() for c in url)
    digits = sum(c.isdigit() for c in url)

    no_of_equals = url.count("=")
    no_of_qmark = url.count("?")
    no_of_ampersand = url.count("&")

    special_chars = sum(
        not c.isalnum() and c not in [".", "/", "=", "?", "&", "-", "_"]
        for c in url
    )

    obfuscated_chars = url.count("%")
    has_obfuscation = int(obfuscated_chars > 0)

    letter_ratio = letters / url_length if url_length else 0
    digit_ratio = digits / url_length if url_length else 0
    special_ratio = special_chars / url_length if url_length else 0
    obfuscation_ratio = obfuscated_chars / url_length if url_length else 0

    is_https = int(parsed.scheme.lower() == "https")

    return {
        "URLLength": url_length,
        "DomainLength": domain_length,
        "IsDomainIP": is_domain_ip,
        "NoOfSubDomain": no_of_subdomain,
        "HasObfuscation": has_obfuscation,
        "NoOfObfuscatedChar": obfuscated_chars,
        "ObfuscationRatio": obfuscation_ratio,
        "NoOfLettersInURL": letters,
        "LetterRatioInURL": letter_ratio,
        "NoOfDegitsInURL": digits,
        "DegitRatioInURL": digit_ratio,
        "NoOfEqualsInURL": no_of_equals,
        "NoOfQMarkInURL": no_of_qmark,
        "NoOfAmpersandInURL": no_of_ampersand,
        "NoOfOtherSpecialCharsInURL": special_chars,
        "SpacialCharRatioInURL": special_ratio,
        "IsHTTPS": is_https,
    }


def get_risk_level(phishing_probability):
    if phishing_probability >= 0.70:
        return "HIGH RISK"
    if phishing_probability >= 0.40:
        return "MEDIUM RISK"
    return "LOW RISK"


def analyze_url(url, model, feature_order):
    scored_url, canonicalized = canonicalize_url(url)
    feature_dict = extract_url_features(scored_url)

    vector = pd.DataFrame(
        [[feature_dict[name] for name in feature_order]],
        columns=feature_order,
    )

    prediction = int(model.predict(vector)[0])
    probabilities = model.predict_proba(vector)[0]
    class_probabilities = {
        int(cls): float(prob)
        for cls, prob in zip(model.classes_, probabilities)
    }

    phishing_probability = class_probabilities[0]
    legitimate_probability = class_probabilities[1]
    classification = "LEGITIMATE" if prediction == 1 else "PHISHING"

    return {
        "input_url": normalize_url(url),
        "scored_url": scored_url,
        "canonicalized": canonicalized,
        "prediction": classification,
        "risk_level": get_risk_level(phishing_probability),
        "phishing_probability": phishing_probability,
        "legitimate_probability": legitimate_probability,
    }


st.markdown(
    """
    <div class="hero">
        <div class="badge">AI-POWERED URL SECURITY</div>
        <h1>Shop<span>Safe</span> AI</h1>
        <p>Analyze URL structure for potential phishing risk without opening the submitted website.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown("**Analyze a website**")

    with st.form("url_form", clear_on_submit=False):
        url = st.text_input(
            "Website URL",
            value="google.com",
            placeholder="example.com or https://example.com/login",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("🔍 Analyze URL")

if submitted:
    if not url.strip():
        st.warning("Enter a URL first.")
        st.stop()

    try:
        model, feature_order = load_artifacts()
        result = analyze_url(url, model, feature_order)
    except Exception as exc:
        st.error(f"Could not analyze this URL: {exc}")
        st.stop()

    prediction = result["prediction"]
    prediction_color = "#34d399" if prediction == "LEGITIMATE" else "#fb7185"

    phishing_pct = result["phishing_probability"] * 100
    legitimate_pct = result["legitimate_probability"] * 100

    st.markdown("### Analysis result")

    st.markdown(
        f"""
        <div style="text-align:center;padding:.4rem 0 1rem;">
            <div class="score-kicker">Classification</div>
            <div class="score-value" style="color:{prediction_color};">{prediction}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="medium")

    with left:
        st.markdown(
            f"""
            <div class="score-card">
                <div class="score-kicker">Phishing likelihood</div>
                <div class="score-value" style="color:#fb7185;">{phishing_pct:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(max(result["phishing_probability"], 0.0), 1.0))

    with right:
        st.markdown(
            f"""
            <div class="score-card">
                <div class="score-kicker">Legitimate likelihood</div>
                <div class="score-value" style="color:#34d399;">{legitimate_pct:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(max(result["legitimate_probability"], 0.0), 1.0))

    if result["canonicalized"]:
        st.info(f"Analyzed as {result['scored_url']}")

    st.markdown(
        '<div class="tiny" style="text-align:center;margin-top:1rem;">'
        'The application analyzes URL characteristics only. It does not open, crawl, or fetch the submitted website.'
        '</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="tiny" style="text-align:center;margin-top:2rem;">'
    'ShopSafe AI · URL classifier'
    '</div>',
    unsafe_allow_html=True,
)
