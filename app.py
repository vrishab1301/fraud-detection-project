import streamlit as st
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

st.set_page_config(page_title="Fraud Detection: Rules vs AI", page_icon="🛡️", layout="wide")


def rule_based_check(amount):
    return 1 if amount < 10 else 0


def get_scores(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn) * 100, fp / (fp + tn) * 100


@st.cache_resource
def load_everything():
    data = pd.read_csv("creditcard.csv")
    X = data.drop("Class", axis=1)
    y = data["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier()
    model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(model)
    rule_scores = get_scores(y_test, X_test["Amount"].apply(rule_based_check))
    ai_scores = get_scores(y_test, model.predict(X_test))
    return model, explainer, X_test, y_test, rule_scores, ai_scores


with st.spinner("Training the AI guard (this only happens once)..."):
    model, explainer, X_test, y_test, rule_scores, ai_scores = load_everything()


def show_verdict(title, guess, truth):
    st.markdown(f"**{title}**")
    said = "FRAUD" if guess == 1 else "Normal"
    if guess == truth:
        st.success(f"Said {said} ✅ Correct")
    else:
        st.error(f"Said {said} ❌ Wrong")


# ---------- Sidebar ----------
with st.sidebar:
    st.header("About this project")
    st.write(
        "Two guards check the same credit card transactions. "
        "**Guard 1** follows a hand-written rule, like a traditional rules engine. "
        "**Guard 2** is an XGBoost AI model that learned from past transactions."
    )
    st.write("**Data:** 284,807 real card transactions, of which only 0.17% are fraud.")
    st.caption(
        "Columns V1 to V28 are anonymized by the dataset provider, "
        "so their real meaning is hidden."
    )

# ---------- Header ----------
st.title("🛡️ Fraud Detection: Rules vs AI")
st.markdown("Can an AI spot fraud better than hand-written rules? Here are the results.")

# ---------- Scoreboard ----------
st.header("📊 Scoreboard")
st.caption("Both guards took the same exam: 56,962 transactions they never saw during training.")

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.subheader("Guard 1: Rules")
        st.caption("Flags any transaction under $10")
        m1, m2 = st.columns(2)
        m1.metric("Fraud caught", f"{rule_scores[0]:.1f}%",
                  help="Of all real fraud, how much was found")
        m2.metric("False alarm rate", f"{rule_scores[1]:.2f}%",
                  help="Of all innocent transactions, how many were wrongly flagged")

with col2:
    with st.container(border=True):
        st.subheader("Guard 2: AI")
        st.caption("XGBoost model trained on past transactions")
        m1, m2 = st.columns(2)
        m1.metric("Fraud caught", f"{ai_scores[0]:.1f}%",
                  help="Of all real fraud, how much was found")
        m2.metric("False alarm rate", f"{ai_scores[1]:.2f}%",
                  help="Of all innocent transactions, how many were wrongly flagged")

st.divider()

# ---------- Single transaction demo ----------
st.header("🔍 Try it on a single transaction")

choice = st.radio(
    "Which kind of transaction should I test?",
    ["A normal one", "A real fraud one"],
    horizontal=True,
)

if st.button("Test it!", type="primary"):
    if choice == "A real fraud one":
        pool = X_test[y_test == 1]
    else:
        pool = X_test[y_test == 0]

    row = pool.sample(1)
    real_answer = y_test.loc[row.index[0]]
    amount = row["Amount"].iloc[0]

    rule_says = rule_based_check(amount)
    ai_says = model.predict(row)[0]

    st.subheader(f"Transaction amount: ${amount:,.2f}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Real answer**")
        st.info("FRAUD" if real_answer == 1 else "Normal")
    with c2:
        show_verdict("Guard 1 (Rules)", rule_says, real_answer)
    with c3:
        show_verdict("Guard 2 (AI)", ai_says, real_answer)

    st.subheader("Why did the AI decide that?")

    shap_values = explainer.shap_values(row)
    explanation = pd.Series(shap_values[0], index=row.columns)
    top5 = explanation.sort_values(key=abs, ascending=False).head(5)

    left, right = st.columns(2)
    with left:
        st.markdown("**🔴 Pushing toward FRAUD**")
        fraud_side = top5[top5 > 0]
        if fraud_side.empty:
            st.write("None in the top 5")
        for name, score in fraud_side.items():
            st.write(f"`{name}`  +{score:.2f}")
    with right:
        st.markdown("**🟢 Pushing toward Normal**")
        normal_side = top5[top5 < 0]
        if normal_side.empty:
            st.write("None in the top 5")
        for name, score in normal_side.items():
            st.write(f"`{name}`  {score:.2f}")

    with st.expander("How to read this"):
        st.write(
            "Each column acts like a witness. A positive score means that column "
            "argued for FRAUD, and a negative score means it argued for Normal. "
            "The bigger the number, the stronger the push. Only the 5 strongest "
            "columns are shown, and the AI weighs all 30 together to decide."
        )

st.divider()
st.caption("Built with Python, XGBoost, SHAP and Streamlit.")