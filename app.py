import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ===============================
# PAGE CONFIG
# ===============================

st.set_page_config(
    page_title="Algorithmic Bias Audit: Loan Approval AI",
    layout="wide"
)

st.title("Algorithmic Bias Audit for AI-Based Loan Approval Systems")

st.markdown(
    """
This application simulates a real-world AI loan approval system and evaluates
whether indirect proxy variables introduce bias across demographic groups.

The model mirrors how financial institutions audit fairness in high-impact
machine learning systems.
"""
)

# ===============================
# SIDEBAR CONTROLS
# ===============================

st.sidebar.header("Simulation Controls")

num_applicants = st.sidebar.slider(
    "Number of Applicants",
    min_value=500,
    max_value=5000,
    step=250,
    value=1000
)

approval_threshold = st.sidebar.slider(
    "Approval Probability Threshold",
    min_value=0.30,
    max_value=0.70,
    step=0.01,
    value=0.50
)

random_seed = st.sidebar.slider(
    "Random Seed",
    min_value=1,
    max_value=100,
    value=42
)

# ===============================
# DATA GENERATION
# ===============================

def generate_applicants(n, seed):
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "income": rng.normal(75000, 22000, n).clip(20000, 200000),
        "credit_score": rng.normal(680, 55, n).clip(300, 850),
        "debt_ratio": rng.uniform(0.05, 0.65, n),
        "zip_proxy": rng.integers(0, 2, n)  # proxy for socioeconomic status
    })

    df["group"] = df["zip_proxy"].map({0: "Group A", 1: "Group B"})
    return df

# ===============================
# APPROVAL PROBABILITY MODEL
# ===============================

def approval_probability(df):
    z = (
        0.000045 * df["income"]
        + 0.0065 * df["credit_score"]
        - 2.8 * df["debt_ratio"]
        - 0.45 * df["zip_proxy"]
        - 6.2
    )

    return 1 / (1 + np.exp(-z))

# ===============================
# SAFE LABEL GENERATION
# ===============================

def generate_labels(probabilities, threshold):
    y = (probabilities >= threshold).astype(int)

    # Safety check: prevent single-class training
    if y.nunique() < 2:
        return None

    return y

# ===============================
# MODEL TRAINING
# ===============================

def train_model(X, y):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])
    model.fit(X, y)
    return model

# ===============================
# RUN SIMULATION
# ===============================

df = generate_applicants(num_applicants, random_seed)
df["approval_probability"] = approval_probability(df)

y = generate_labels(df["approval_probability"], approval_threshold)

if y is None:
    st.error(
        "Model training halted because all applicants were assigned the same outcome. "
        "Adjust the approval threshold or random seed."
    )
    st.stop()

X = df[["income", "credit_score", "debt_ratio", "zip_proxy"]]
model = train_model(X, y)

df["approved"] = y

# ===============================
# METRICS
# ===============================

st.subheader("Approval Rate Analysis")

group_stats = (
    df.groupby("group")["approved"]
    .mean()
    .reset_index()
    .rename(columns={"approved": "approval_rate"})
)

col1, col2 = st.columns(2)

with col1:
    st.dataframe(group_stats, use_container_width=True)

with col2:
    fig, ax = plt.subplots()
    ax.bar(group_stats["group"], group_stats["approval_rate"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Approval Rate")
    ax.set_title("Approval Rates by Group")
    st.pyplot(fig)

# ===============================
# BIAS METRICS
# ===============================

max_rate = group_stats["approval_rate"].max()
min_rate = group_stats["approval_rate"].min()
disparity = max_rate - min_rate

st.subheader("Bias Metrics")

c1, c2, c3 = st.columns(3)

c1.metric("Max Approval Rate", f"{max_rate:.2%}")
c2.metric("Min Approval Rate", f"{min_rate:.2%}")
c3.metric("Approval Rate Disparity", f"{disparity:.2%}")

# ===============================
# FEATURE IMPORTANCE
# ===============================

st.subheader("Model Feature Influence")

coefficients = model.named_steps["clf"].coef_[0]

feature_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": coefficients
}).sort_values(by="Coefficient", key=abs, ascending=False)

fig2, ax2 = plt.subplots()
ax2.barh(feature_df["Feature"], feature_df["Coefficient"])
ax2.axvline(0)
ax2.set_title("Logistic Regression Coefficients")
st.pyplot(fig2)

# ===============================
# ETHICAL INTERPRETATION
# ===============================

st.subheader("Ethical Interpretation")

st.markdown(
    """
This simulation demonstrates how indirect proxy variables can introduce
systemic bias into AI-driven financial decisions.

Although the model does not explicitly use protected demographic attributes,
the inclusion of correlated proxy features (such as geographic indicators)
leads to measurable disparities in approval outcomes.

Key insight:  
Bias can emerge even when sensitive attributes are excluded, emphasizing
the importance of continuous auditing, transparency, and fairness testing
in financial AI systems.
"""
)

# ===============================
# MITIGATION STRATEGIES
# ===============================

st.subheader("Bias Mitigation Strategies")

st.markdown(
    """
- Remove or constrain proxy variables correlated with protected attributes  
- Apply fairness-aware model constraints during training  
- Perform subgroup performance monitoring post-deployment  
- Use counterfactual testing to evaluate decision robustness  
- Conduct regular independent audits of automated decision systems  
"""
)
