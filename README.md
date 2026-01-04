# Algorithmic Bias Audit for AI-Based Loan Approval Systems

This project simulates a real-world AI loan approval model and evaluates whether
indirect proxy variables introduce unfair bias across demographic groups.

## Key Features
- Synthetic applicant population with correlated financial attributes
- Machine learning-based approval model (logistic regression)
- Proxy variable leakage simulation (zip code risk)
- Fairness metrics:
  - Statistical Parity Difference
  - Disparate Impact Ratio
  - Equal Opportunity Difference
- Distributional analysis of approval probabilities
- ROC curve comparison across groups

## Motivation
AI-driven financial systems increasingly influence access to credit.
This project demonstrates how bias can emerge even without explicitly using
demographic variables, highlighting the importance of ethical AI auditing.

## Technologies Used
- Python
- Streamlit
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py

