import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="AI Loan Bias Detector",
    layout="wide"
)

st.title("🏦 AI Algorithmic Bias Detector for Loan Approvals")
st.write(
    "This tool simulates a loan approval AI model and evaluates whether its decisions "
    "introduce bias across different demographic and financial groups."
)

# -----------------------------
# Data Generation
# -----------------------------
@st.cache_data
def generate_data(n=1000):
    np.random.seed(42)

    income = np.random.normal(60000, 15000, n)
    credit_score = np.random.normal(680, 50, n)
    zip_code = np.random.choice(["Low Income Area", "Middle Income Area", "High Income Area"], n)

    # Encode zip code bias (real-world proxy issue)
    zip_bias = np.where(zip_code == "Low Income Area", -0.8,
                np.where(zip_code == "Middle Income Area", 0, 0.8))

    approval_probability = (
        0.00003 * income +
        0.01 * credit_score +
        zip_bias -
        8
    )

    approved = (1 / (1 + np.exp(-approval_probability)) > 0.5).astype(int)

    return pd.DataFrame({
        "Income": income,
        "CreditScore": credit_score,
        "ZipCode": zip_code,
        "Approved": approved
    })

df = generate_data()

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("Simulation Controls")
sample_size = st.sidebar.slider("Number of Applicants", 300, 3000, 1000)

df = generate_data(sample_size)

# -----------------------------
# Train AI Model
# -----------------------------
X = df[["Income", "CreditScore"]]
y = df["Approved"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

df["AI_Prediction"] = model.predict(X_scaled)

# -----------------------------
# Display Dataset
# -----------------------------
st.subheader("📊 Simulated Applicant Data")
st.dataframe(df.head(20))

# -----------------------------
# Approval Rate Analysis
# -----------------------------
st.subheader("⚖️ Approval Rates by Group")

approval_by_zip = df.groupby("ZipCode")["AI_Prediction"].mean()

fig, ax = plt.subplots()
approval_by_zip.plot(kind="bar", ax=ax)
ax.set_ylabel("Approval Rate")
ax.set_title("Loan Approval Rate by Zip Code Group")
st.pyplot(fig)

# -----------------------------
# Bias Metrics
# -----------------------------
st.subheader("📉 Bias Metrics")

max_rate = approval_by_zip.max()
min_rate = approval_by_zip.min()
disparity = max_rate - min_rate

st.metric("Max Approval Rate", f"{max_rate:.2%}")
st.metric("Min Approval Rate", f"{min_rate:.2%}")
st.metric("Approval Rate Disparity", f"{disparity:.2%}")

# -----------------------------
# Ethical Analysis
# -----------------------------
st.subheader("🧠 Ethical Interpretation")

st.write(
    """
This simulation demonstrates how **proxy variables** such as zip code can introduce
systemic bias into AI-driven financial decisions.

Although the model does not explicitly use demographic attributes, indirect correlations
can still result in unequal outcomes across socioeconomic groups.

**Key Takeaway:**  
AI systems must be continuously audited, validated, and adjusted to ensure fairness —
especially in high-impact domains like finance.
"""
)

# -----------------------------
# Bias Mitigation Example
# -----------------------------
st.subheader("🛠️ Bias Mitigation Strategy")

st.write(
    """
Possible mitigation strategies include:
- Removing proxy variables correlated with protected attributes
- Rebalancing training data
- Applying fairness-aware model constraints
- Conducting regular post-deployment audits
"""
)

# -----------------------------
# Conclusion
# -----------------------------
st.success(
    "This project emphasizes responsible AI development, transparency, and ethical oversight — "
    "principles critical to modern financial systems."
)
