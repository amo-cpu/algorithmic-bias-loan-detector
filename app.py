import streamlit as st
import numpy as np
import pandas as pd

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="AI Loan Bias Audit Framework", layout="wide")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# -------------------------------
# DATA GENERATION
# -------------------------------
def generate_applicants(n):
    data = pd.DataFrame({
        "income": np.random.normal(65000, 18000, n).clip(20000, 200000),
        "credit_score": np.random.normal(680, 60, n).clip(300, 850),
        "debt_ratio": np.random.uniform(0.1, 0.6, n),
        "socioeconomic_index": np.random.choice(
            ["low", "medium", "high"], n, p=[0.3, 0.5, 0.2]
        )
    })
    return data

# -------------------------------
# DECISION MODEL
# -------------------------------
def loan_decision_model(df):
    score = (
        0.4 * (df["credit_score"] / 850) +
        0.35 * (df["income"] / 200000) -
        0.25 * df["debt_ratio"]
    )

    # Proxy bias via socioeconomic index
    bias_map = {"low": -0.05, "medium": 0.0, "high": 0.05}
    score += df["socioeconomic_index"].map(bias_map)

    df["approval_score"] = score
    df["approved"] = (score >= 0.5).astype(int)
    return df

# -------------------------------
# FAIRNESS METRICS
# -------------------------------
def approval_rates(df):
    return df.groupby("socioeconomic_index")["approved"].mean()

def disparate_impact(rates):
    return rates.min() / rates.max()

def statistical_parity_diff(rates):
    return rates.max() - rates.min()

# -------------------------------
# COUNTERFACTUAL FAIRNESS
# -------------------------------
def counterfactual_analysis(df):
    flips = 0
    for idx, row in df.iterrows():
        original = row["approved"]
        for group in ["low", "medium", "high"]:
            if group != row["socioeconomic_index"]:
                test_row = row.copy()
                test_row["socioeconomic_index"] = group
                test_df = pd.DataFrame([test_row])
                decision = loan_decision_model(test_df)["approved"].iloc[0]
                if decision != original:
                    flips += 1
                    break
    return flips / len(df)

# -------------------------------
# MONTE CARLO STRESS TEST
# -------------------------------
def monte_carlo_bias(runs, n):
    disparities = []
    for _ in range(runs):
        df = generate_applicants(n)
        df = loan_decision_model(df)
        rates = approval_rates(df)
        disparities.append(statistical_parity_diff(rates))
    return np.array(disparities)

# -------------------------------
# MITIGATION
# -------------------------------
def mitigate_bias(df):
    df = df.copy()
    df["approval_score"] -= df["socioeconomic_index"].map(
        {"low": -0.05, "medium": 0.0, "high": 0.05}
    )
    df["approved"] = (df["approval_score"] >= 0.5).astype(int)
    return df

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("AI Algorithmic Fairness & Bias Audit Framework")
st.markdown(
    "This system evaluates algorithmic bias in financial decision models "
    "using statistical, counterfactual, and stress-testing methods."
)

n_applicants = st.slider("Number of Applicants", 300, 5000, 1000)
runs = st.slider("Monte Carlo Simulations", 50, 300, 150)

df = generate_applicants(n_applicants)
df = loan_decision_model(df)

st.subheader("Sample Applicant Data")
st.dataframe(df.head())

rates = approval_rates(df)

st.subheader("Approval Rates by Group")
st.write(rates)

spd = statistical_parity_diff(rates)
di = disparate_impact(rates)

st.subheader("Bias Metrics")
st.write({
    "Statistical Parity Difference": round(spd, 4),
    "Disparate Impact Ratio": round(di, 4)
})

cf_rate = counterfactual_analysis(df)

st.subheader("Counterfactual Fairness")
st.write(f"Decision Flip Rate: {cf_rate:.2%}")

stress = monte_carlo_bias(runs, n_applicants)

st.subheader("Bias Stress Test")
st.write({
    "Mean Bias": round(stress.mean(), 4),
    "Worst-Case Bias": round(stress.max(), 4)
})

df_mitigated = mitigate_bias(df)
rates_mitigated = approval_rates(df_mitigated)

st.subheader("Post-Mitigation Results")
st.write(rates_mitigated)

st.markdown(
    "This audit demonstrates how proxy variables can introduce systemic bias "
    "even when protected attributes are not explicitly used."
)

