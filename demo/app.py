# Import streamlit for the web interface
import streamlit as st
# Import pandas to build the input row for the model
import pandas as pd
# Import pickle to load the saved model bundle
import pickle
# Import os to find model.pkl next to this file
import os

# Set the page title and layout
st.set_page_config(page_title="Credit delinquency risk demo", layout="centered")


# Load the saved bundle once and reuse it between clicks
@st.cache_resource
def load_bundle():
    # Path to model.pkl in the same folder as this file
    path = os.path.join(os.path.dirname(__file__), "model.pkl")
    # Open the file and read the bundle
    with open(path, "rb") as f:
        return pickle.load(f)


# Unpack the model, the column order, the threshold and the training caps
bundle = load_bundle()
model = bundle["model"]
inputs = bundle["inputs"]
threshold = bundle["threshold"]
caps = bundle["caps"]

# Title and short description
st.title("Credit delinquency risk demo")
st.write("Enter a borrower's details to see the model's score for becoming 90 or more days past due within 2 years.")

# Input widgets in two columns, with defaults near the typical borrower
st.subheader("Borrower details")
col1, col2 = st.columns(2)
with col1:
    utilisation = st.number_input("Utilisation of unsecured credit lines (balance divided by limit)", min_value=0.0, value=0.15, step=0.05)
    debt_ratio = st.number_input("Debt ratio", min_value=0.0, value=0.30, step=0.05)
    income = st.number_input("Monthly income", min_value=0.0, value=5400.0, step=100.0)
    open_lines = st.number_input("Open credit lines and loans", min_value=0, value=8, step=1)
    real_estate = st.number_input("Real estate loans or lines", min_value=0, value=1, step=1)
with col2:
    dependents = st.number_input("Number of dependents", min_value=0, value=0, step=1)
    late_30 = st.number_input("Times 30 to 59 days past due", min_value=0, value=0, step=1)
    late_60 = st.number_input("Times 60 to 89 days past due", min_value=0, value=0, step=1)
    late_90 = st.number_input("Times 90 or more days late", min_value=0, value=0, step=1)

# Score the borrower when the button is clicked
if st.button("Score borrower"):
    # Build one row using the exact column names the model was trained on
    row = {
        "RevolvingUtilizationOfUnsecuredLines": utilisation,
        "NumberOfTime30-59DaysPastDueNotWorse": late_30,
        "DebtRatio": debt_ratio,
        "MonthlyIncome": income,
        "NumberOfOpenCreditLinesAndLoans": open_lines,
        "NumberOfTimes90DaysLate": late_90,
        "NumberRealEstateLoansOrLines": real_estate,
        "NumberOfTime60-89DaysPastDueNotWorse": late_60,
        "NumberOfDependents": dependents,
    }

    # Cap values above the training 99th percentile, as in cleaning stage B
    capped = []
    for col, cap in caps.items():
        if row[col] > cap:
            row[col] = cap
            capped.append(col)

    # Put the row in the column order the model expects
    X = pd.DataFrame([row], columns=inputs).astype(float)

    # Predicted probability of delinquency and the flag
    score = float(model.predict_proba(X)[0, 1])
    flagged = score >= threshold

    # Show the result
    st.write("## Result")
    st.metric("Model score", f"{score * 100:.1f}%")
    if flagged:
        st.write(f"Flagged: the score is at or above the threshold of {threshold * 100:.1f}%.")
    else:
        st.write(f"Not flagged: the score is below the threshold of {threshold * 100:.1f}%.")

    # Tell the user if any value was capped before scoring
    if capped:
        st.write("Capped to the training limit before scoring: " + ", ".join(capped))

    # Explain where the threshold comes from, using the results from the notebook
    with st.expander("How to read this"):
        st.write(
            "The threshold was chosen on the training data to catch about 60% of delinquent borrowers. "
            "On the held-out test part it caught 61.9% of them, and 32.2% of the borrowers it flagged became delinquent. "
            "About 6.7% of all borrowers in the data became delinquent."
        )

# Limits shown on the page
st.caption("Built on a public Kaggle dataset that does not state its country or year. For demonstration only, not for real lending decisions.")
