import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

# -------------------------------
# App Configuration
# -------------------------------
st.set_page_config(page_title="AI Loan Approval Bias Audit", layout="wide")

st.title("Algorithmic Bias Audit for AI-Based Loan Approval Systems")
st.write(
    """
    This application simulates an AI-driven loan approval system and evaluates
    whether indirect proxy variables introduce bias across demographic groups.
    The model reflects how real financial institutions audit fairness in
    high-impact decision systems.
    """
)

# -------------------------------
# Sidebar Controls
# -------------------------------
st.sidebar.header("Simulation Controls")

n_applicants = st.sidebar.slider(
    "Number of Applicants",
    min_value=500,
    max_value=5000,
    value=2000,
    step=500
)

decision_threshold = st.sidebar.slider(
    "Approval Probability Threshold",
    min_value=0.3,
    max_value=0.7,
    value=0.5,
    step=0.05
)

random_seed = st.sidebar.number_input(
    "Random Seed",
    value=42,
    step=1
)

np.random.seed(random_seed)

# -------------------------------
# Data Generation
# -------------------------------
def generate_applicant_data(n):
    groups = np.random.choice(["Group A", "Group B"], size=n, p=[0.5, 0.5])

    data = []

    for g in groups:
        if g == "Group A":
            credit_score = np.random.normal(720, 40)
            income = np.random.normal(85000, 15000)
            zip_risk = np.random.normal(0.30, 0.08)   # safer zip codes
        else:
            credit_score = np.random.normal(680, 45)
            income = np.random.normal(70000, 18000)
            zip_risk = np.random.normal(0.55, 0.10)   # higher-risk zip codes

        debt_to_income = np.clip(np.random.normal(0.30, 0.10), 0.05, 0.8)
        credit_history = np.clip(np.random.normal(8, 4), 1, 25)

        data.append([
            g,
            credit_score,
            income,
            debt_to_income,
            credit_history,
            zip_risk
        ])

    df = pd.DataFrame(
        data,
        columns=[
            "group",
            "credit_score",
            "income",
            "debt_to_income",
            "credit_history_length",
            "zip_risk_score"
        ]
    )

    return df

df = generate_applicant_data(n_applicants)

# -------------------------------
# Train Loan Approval Model
# -------------------------------
features = [
    "credit_score",
    "income",
    "debt_to_income",
    "credit_history_length",
    "zip_risk_score"
]

# Ground truth approval logic (hidden from model)
latent_score = (
    0.004 * df["credit_score"]
    + 0.00001 * df["income"]
    - 2.0 * df["debt_to_income"]
    + 0.05 * df["credit_history_length"]
    - 1.5 * df["zip_risk_score"]
)

prob_true = 1 / (1 + np.exp(-latent_score))
df["true_approval"] = (prob_true > 0.5).astype(int)

# Train model
X = df[features]
y = df["true_approval"]

model = LogisticRegression(max_iter=2000)
model.fit(X, y)

df["approval_probability"] = model.predict_proba(X)[:, 1]
df["model_decision"] = (df["approval_probability"] >= decision_threshold).astype(int)

# -------------------------------
# Fairness Metrics
# -------------------------------
def statistical_parity_difference(df):
    rates = df.groupby("group")["model_decision"].mean()
    return rates.max() - rates.min()

def disparate_impact_ratio(df):
    rates = df.groupby("group")["model_decision"].mean()
    return rates.min() / rates.max()

def equal_opportunity_difference(df):
    tpr = {}
    for g in df["group"].unique():
        subset = df[df["group"] == g]
        positives = subset[subset["true_approval"] == 1]
        tpr[g] = positives["model_decision"].mean()
    return max(tpr.values()) - min(tpr.values())

spd = statistical_parity_difference(df)
dir_ratio = disparate_impact_ratio(df)
eod = equal_opportunity_difference(df)

# -------------------------------
# Layout
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Approval Rates by Group")
    approval_rates = df.groupby("group")["model_decision"].mean()

    fig, ax = plt.subplots()
    approval_rates.plot(kind="bar", ax=ax)
    ax.set_ylabel("Approval Rate")
    ax.set_ylim(0, 1)
    st.pyplot(fig)

with col2:
    st.subheader("Fairness Metrics")
    st.metric("Statistical Parity Difference", round(spd, 3))
    st.metric("Disparate Impact Ratio", round(dir_ratio, 3))
    st.metric("Equal Opportunity Difference", round(eod, 3))

# -------------------------------
# Distribution Plot
# -------------------------------
st.subheader("Approval Probability Distributions")

fig, ax = plt.subplots()
for g in df["group"].unique():
    sns.kdeplot(
        df[df["group"] == g]["approval_probability"],
        label=g,
        ax=ax
    )

ax.set_xlabel("Predicted Approval Probability")
ax.set_ylabel("Density")
ax.legend()
st.pyplot(fig)

# -------------------------------
# ROC Curves
# -------------------------------
st.subheader("Model Performance by Group")

fig, ax = plt.subplots()

for g in df["group"].unique():
    subset = df[df["group"] == g]
    fpr, tpr, _ = roc_curve(subset["true_approval"], subset["approval_probability"])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{g} (AUC = {roc_auc:.2f})")

ax.plot([0, 1], [0, 1], linestyle="--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()
st.pyplot(fig)

# -------------------------------
# Interpretation
# -------------------------------
st.subheader("Ethical and Financial Interpretation")
st.write(
    """
    Although the model does not explicitly use protected demographic attributes,
    proxy variables such as zip code risk score introduce statistically measurable
    disparities in approval outcomes.

    This reflects real-world financial systems, where indirect correlations can
    produce unequal treatment even when models are formally race-blind.

    Such findings emphasize the importance of continuous auditing, fairness-aware
    modeling, and regulatory oversight in AI-driven financial decision systems.
    """
)
