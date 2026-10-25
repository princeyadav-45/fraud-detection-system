import json
import os
import time
import uuid
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh

# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DATA_FILE = BASE_DIR / "data" / "processed_transactions.jsonl"
API_URL = os.getenv(
    "FRAUDSHIELD_API_URL",
    "https://fraud-detection-system-rg0j.onrender.com"
).rstrip("/")

ADMIN_USERNAME = os.getenv("FRAUDSHIELD_ADMIN_USER", "")
ADMIN_PASSWORD = os.getenv("FRAUDSHIELD_ADMIN_PASSWORD", "")

st.set_page_config(
    page_title="FraudShield | Transaction Protection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# THEME
# =========================================================

st.markdown("""
<style>
.stApp {
    background: #101813;
    color: #F2F5F0;
}

[data-testid="stHeader"] {
    background: #101813;
}

.block-container {
    max-width: 1100px;
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}

h1, h2, h3 {
    color: #B5F5C2 !important;
}

p, label, li {
    color: #E0E9E0;
}

[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p {
    color: #EAF2E9 !important;
}

.stButton button,
.stDownloadButton button,
[data-testid="stFormSubmitButton"] button {
    background: #278744 !important;
    color: #FFFFFF !important;
    border: 1px solid #4CC46D !important;
    border-radius: 9px !important;
    min-height: 42px;
    font-weight: 600;
}

.stButton button:hover {
    background: #36A657 !important;
}

.stTextInput input,
.stNumberInput input,
.stTextArea textarea,
[data-baseweb="select"] > div {
    background: #F4F7F2 !important;
    color: #17251C !important;
    border-radius: 8px !important;
}

[data-testid="stMetric"] {
    background: #1C2B21;
    border: 1px solid #344D3A;
    border-radius: 12px;
    padding: 14px;
}

[data-testid="stMetricValue"] {
    color: #B5F5C2 !important;
}

[data-testid="stMetricLabel"] {
    color: #C1D2C2 !important;
}

[data-testid="stForm"] {
    background: #17231B;
    border: 1px solid #34483A;
    border-radius: 14px;
    padding: 18px;
}

[data-testid="stDataFrame"] {
    border: 1px solid #34483A;
    border-radius: 10px;
}

.portal-card {
    background: linear-gradient(145deg, #1B2B20, #17231B);
    border: 1px solid #35513C;
    border-radius: 15px;
    padding: 22px;
    min-height: 155px;
    margin-bottom: 12px;
}

.portal-card h3 {
    margin-top: 8px;
    margin-bottom: 8px;
}

.small-muted {
    color: #B8C8BA;
    font-size: 14px;
}

hr {
    border-color: #34483A !important;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "page": "Home",
    "user_id": f"DEMO-{uuid.uuid4().hex[:8].upper()}",
    "admin_logged_in": False,
    "last_transaction": None,
    "last_result": None,
    "acknowledged_alerts": [],
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def navigate(page):
    st.session_state.page = page
    st.rerun()


def page_header(title, subtitle=""):
    st.title(title)
    if subtitle:
        st.caption(subtitle)
    st.divider()


def load_transactions():
    if not DATA_FILE.exists():
        return pd.DataFrame()

    records = []

    try:
        with DATA_FILE.open("r", encoding="utf-8") as file:
            for line in file:
                try:
                    if line.strip():
                        records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    if not df.empty:
        for column in ("amount", "risk_score"):
            if column in df.columns:
                df[column] = pd.to_numeric(
                    df[column], errors="coerce"
                )

        if "status" in df.columns:
            df["status"] = df["status"].astype(str).str.upper()

    return df


def get_user_transactions():
    df = load_transactions()

    if df.empty or "user_id" not in df.columns:
        return pd.DataFrame()

    return df[
        df["user_id"].astype(str) == st.session_state.user_id
    ].copy()


def risk_percent(value):
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def is_suspicious(status):
    return str(status).upper() == "SUSPICIOUS"


def api_predict(payload):
    response = requests.post(
        f"{API_URL}/predict",
        json=payload,
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


def explain_result(result):
    score = float(result.get("risk_score", 0) or 0)
    status = str(result.get("status", "UNKNOWN")).upper()

    st.subheader("Risk Explanation")
    st.metric("Estimated Risk", risk_percent(score))

    if is_suspicious(status):
        st.warning(
            "The model flagged this transaction for review. "
            "Verify the merchant, amount and location before proceeding."
        )
    else:
        st.info(
            "The model did not flag this transaction as suspicious. "
            "A low risk score does not guarantee that it is genuine."
        )

    st.write("**Model classification:**", status)

    st.caption(
        "This prototype does not currently provide verified causal "
        "explanations for individual model features."
    )

# =========================================================
# HOME PAGE
# =========================================================

def render_home():
    st.write("")

    # Centered brand
    _, center, _ = st.columns([1, 2, 1])

    with center:
        st.markdown("""
        <div style="text-align:center; padding:8px 0 22px;">
            <div style="font-size:46px;">🛡️</div>
            <h1 style="font-size:38px; margin:8px 0;">
                FraudShield
            </h1>
            <p style="font-size:15px; color:#B8C8BA;">
                Intelligent Transaction Protection
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; margin-bottom:24px;">
        <h2 style="font-size:26px;">Welcome to FraudShield</h2>
        <p style="color:#B8C8BA;">
            Choose your portal to continue
        </p>
    </div>
    """, unsafe_allow_html=True)

    _, user_col, admin_col, _ = st.columns(
        [0.3, 2, 2, 0.3],
        gap="medium",
    )

    with user_col:
        st.markdown("""
        <div class="portal-card">
            <div style="font-size:30px;">👤</div>
            <h3>User Portal</h3>
            <p class="small-muted">
                Analyse transactions, explore risk and download reports.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "Continue as User →",
            key="home_user",
            use_container_width=True,
        ):
            navigate("User")

    with admin_col:
        st.markdown("""
        <div class="portal-card">
            <div style="font-size:30px;">🔐</div>
            <h3>Admin Portal</h3>
            <p class="small-muted">
                Monitor suspicious activity, transactions and system health.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "Admin Login →",
            key="home_admin",
            use_container_width=True,
        ):
            navigate("Admin Login")

    st.write("")

    _, notice, _ = st.columns([0.3, 4, 0.3])

    with notice:
        st.info(
            "FraudShield is a demonstration system. It does not execute, "
            "block or authorise real payments."
        )

# =========================================================
# USER PORTAL
# =========================================================

def render_user():
    page_header(
        "User Portal",
        "Transaction risk analysis and personal security tools",
    )

    left, right = st.columns([3, 1])
    left.caption(f"Demo session ID: {st.session_state.user_id}")

    if right.button("← Home", use_container_width=True):
        navigate("Home")

    tabs = st.tabs([
        "🔎 Risk Check",
        "🧠 Explainable Result",
        "💰 Spending",
        "🚨 Alerts",
        "🛡️ Safety Assistant",
        "🧪 Simulator",
        "📄 Reports",
    ])

    # -----------------------------------------------------
    # 1. RISK CHECK
    # -----------------------------------------------------

    with tabs[0]:
        st.subheader("Analyse a Transaction")

        with st.form("transaction_form"):
            amount = st.number_input(
                "Transaction Amount (₹)",
                min_value=1.0,
                max_value=100000000.0,
                value=1000.0,
                step=100.0,
            )

            merchant = st.text_input(
                "Merchant",
                placeholder="e.g. Online Shopping",
            )

            location = st.text_input(
                "Location",
                placeholder="e.g. Nagpur",
            )

            submitted = st.form_submit_button(
                "Analyse Transaction",
                use_container_width=True,
            )

        if submitted:
            if not merchant.strip() or not location.strip():
                st.error("Please enter the merchant and location.")
            else:
                payload = {
                    "amount": amount,
                    "merchant": merchant.strip(),
                    "location": location.strip(),
                    "user_id": st.session_state.user_id,
                }

                try:
                    with st.spinner("Submitting transaction..."):
                        response = requests.post(
                            f"{API_URL}/transactions",
                            json=payload,
                            timeout=12,
                        )

                    if response.status_code != 202:
                        st.error(
                            f"API returned HTTP {response.status_code}"
                        )
                        st.code(response.text)
                    else:
                        body = response.json()
                        txn_id = body.get("transaction_id")

                        st.session_state.last_transaction = txn_id

                        result = None
                        progress = st.progress(0)
                        message = st.empty()

                        for attempt in range(20):
                            message.info(
                                "Waiting for transaction processing..."
                            )

                            try:
                                check = requests.get(
                                    f"{API_URL}/transactions/{txn_id}",
                                    timeout=3,
                                )

                                if check.ok:
                                    candidate = check.json()

                                    if candidate.get(
                                        "processing_status"
                                    ) == "COMPLETED":
                                        result = candidate
                                        break

                            except requests.RequestException:
                                pass

                            progress.progress((attempt + 1) / 20)
                            time.sleep(1)

                        progress.empty()
                        message.empty()

                        if result:
                            st.session_state.last_result = result

                            score = float(
                                result.get("risk_score", 0) or 0
                            )
                            status = str(
                                result.get("status", "UNKNOWN")
                            ).upper()

                            a, b, c = st.columns(3)

                            a.metric(
                                "Estimated Risk",
                                risk_percent(score),
                            )
                            b.metric("Classification", status)
                            c.metric(
                                "Decision",
                                str(result.get("decision", "REVIEW")),
                            )

                            st.progress(max(0.0, min(1.0, score)))

                            if is_suspicious(status):
                                st.error(
                                    "Suspicious activity flagged. "
                                    "Verify the transaction before proceeding."
                                )
                            else:
                                st.success(
                                    "The model did not flag this transaction. "
                                    "This does not guarantee it is genuine."
                                )

                            st.write("**Transaction ID:**", txn_id)
                            st.json(result)

                        else:
                            st.warning(
                                "The transaction was submitted, but the result "
                                "has not appeared yet. Check that consumer.py "
                                "is running and retry later."
                            )
                            st.code(str(txn_id))

                except requests.RequestException as exc:
                    st.error("Could not connect to the FraudShield API.")
                    st.code(str(exc))

    # -----------------------------------------------------
    # 2. EXPLAINABLE RESULT
    # -----------------------------------------------------

    with tabs[1]:
        st.subheader("Explainable AI")

        result = st.session_state.last_result

        if result:
            explain_result(result)
        else:
            st.info(
                "Analyse a transaction first to view its risk explanation."
            )

    # -----------------------------------------------------
    # 3. PERSONAL SPENDING ANALYTICS
    # -----------------------------------------------------

    with tabs[2]:
        st.subheader("My Spending Analytics")

        st.caption(
            "Analytics use processed transactions matching this demo "
            "session ID. This is not a permanent user account."
        )

        user_df = get_user_transactions()

        if user_df.empty:
            st.info("No transaction history found for this session.")
        else:
            if "amount" not in user_df.columns:
                user_df["amount"] = 0

            total = user_df["amount"].fillna(0).sum()
            count = len(user_df)
            average = total / count if count else 0

            a, b, c = st.columns(3)
            a.metric("Total Amount", f"₹{total:,.2f}")
            b.metric("Transactions", count)
            c.metric("Average Amount", f"₹{average:,.2f}")

            if "merchant" in user_df.columns:
                st.subheader("Spending by Merchant")

                chart_data = (
                    user_df.groupby("merchant")["amount"]
                    .sum()
                    .sort_values(ascending=False)
                    .head(10)
                )

                st.bar_chart(chart_data)

            st.subheader("Recent Transactions")
            st.dataframe(
                user_df.iloc[::-1],
                use_container_width=True,
                hide_index=True,
            )

    # -----------------------------------------------------
    # 4. SMART SECURITY ALERTS
    # -----------------------------------------------------

    with tabs[3]:
        st.subheader("Smart Security Alerts")

        user_df = get_user_transactions()

        if user_df.empty or "status" not in user_df.columns:
            st.info("No processed alerts are available for this session.")
        else:
            alerts = user_df[
                user_df["status"].astype(str).str.upper()
                == "SUSPICIOUS"
            ]

            if alerts.empty:
                st.success(
                    "No suspicious transactions were flagged in your history."
                )
            else:
                st.warning(
                    f"{len(alerts)} transaction(s) flagged for review."
                )

                for _, row in alerts.iloc[::-1].iterrows():
                    txn_id = str(
                        row.get(
                            "transaction_id",
                            row.get("id", "Unknown"),
                        )
                    )

                    with st.expander(f"⚠️ Transaction: {txn_id}"):
                        st.write("**Amount:**", row.get("amount", "N/A"))
                        st.write("**Merchant:**", row.get("merchant", "N/A"))
                        st.write("**Location:**", row.get("location", "N/A"))
                        st.write(
                            "**Risk Score:**",
                            risk_percent(row.get("risk_score")),
                        )

                        if txn_id not in st.session_state.acknowledged_alerts:
                            if st.button(
                                "Acknowledge Alert",
                                key=f"ack_{txn_id}",
                            ):
                                st.session_state.acknowledged_alerts.append(
                                    txn_id
                                )
                                st.success(
                                    "Alert acknowledged in this session."
                                )
                        else:
                            st.caption("Acknowledged in this session.")

                st.caption(
                    "Acknowledgement only records that you viewed the alert. "
                    "It does not resolve or verify the transaction."
                )

    # -----------------------------------------------------
    # 5. FRAUD AWARENESS ASSISTANT
    # -----------------------------------------------------

    with tabs[4]:
        st.subheader("Fraud Awareness Assistant")

        language = st.radio(
            "Language / भाषा",
            ["English", "हिन्दी"],
            horizontal=True,
        )

        question = st.selectbox(
            "What do you need help with?",
            [
                "Someone asked for my OTP or UPI PIN",
                "I received an unknown payment link",
                "I see an unfamiliar transaction",
                "How can I make safer online payments?",
            ],
        )

        answers_en = {
            "Someone asked for my OTP or UPI PIN":
                "Never share your OTP, UPI PIN or password. "
                "Banks do not need your PIN to receive money. "
                "Verify requests through official channels.",

            "I received an unknown payment link":
                "Avoid unexpected links and unknown applications. "
                "Open your bank or payment app directly instead.",

            "I see an unfamiliar transaction":
                "Contact your bank through its official app or website, "
                "secure your account and report the transaction promptly.",

            "How can I make safer online payments?":
                "Verify the recipient and amount, use trusted apps, enable "
                "account alerts and never share authentication details.",
        }

        answers_hi = {
            "Someone asked for my OTP or UPI PIN":
                "Apna OTP, UPI PIN ya password kisi ko mat batao. "
                "Paise receive karne ke liye PIN share karne ki zaroorat nahi.",

            "I received an unknown payment link":
                "Unknown link par click mat karo aur anjaan app install mat "
                "karo. Apni bank ya payment app seedhe kholo.",

            "I see an unfamiliar transaction":
                "Bank ke official app ya website se turant contact karo, "
                "account secure karo aur transaction report karo.",

            "How can I make safer online payments?":
                "Recipient aur amount verify karo, trusted apps use karo aur "
                "OTP, PIN ya password kabhi share mat karo.",
        }

        if st.button("Get Safety Advice"):
            if language == "English":
                st.info(answers_en[question])
            else:
                st.info(answers_hi[question])

        st.caption(
            "This is a rule-based educational assistant, not a bank support service."
        )

    # -----------------------------------------------------
    # 6. WHAT-IF SIMULATOR
    # -----------------------------------------------------

    with tabs[5]:
        st.subheader("What-if Risk Simulator")

        st.write(
            "Test hypothetical transaction details without submitting "
            "them to the Kafka transaction-history pipeline."
        )

        with st.form("simulator_form"):
            sim_amount = st.number_input(
                "Hypothetical Amount (₹)",
                min_value=1.0,
                value=5000.0,
                step=500.0,
            )

            sim_merchant = st.text_input(
                "Hypothetical Merchant",
                value="Online Shopping",
            )

            sim_location = st.text_input(
                "Hypothetical Location",
                value="Nagpur",
            )

            simulate = st.form_submit_button("Run Simulation")

        if simulate:
            if not sim_merchant.strip() or not sim_location.strip():
                st.error("Enter a merchant and location.")
            else:
                try:
                    with st.spinner("Running hypothetical prediction..."):
                        prediction = api_predict({
                            "amount": sim_amount,
                            "merchant": sim_merchant.strip(),
                            "location": sim_location.strip(),
                            "user_id": st.session_state.user_id,
                        })

                    st.success("Simulation prediction returned.")

                    st.metric(
                        "Estimated Risk",
                        risk_percent(prediction.get("risk_score")),
                    )

                    st.write(
                        "**Classification:**",
                        prediction.get("status", "N/A"),
                    )

                    st.json(prediction)

                    st.caption(
                        "This is a hypothetical prediction, not a real payment "
                        "or a confirmed fraud decision."
                    )

                except requests.RequestException as exc:
                    st.error(
                        "Simulation failed. Confirm that the API is running "
                        "and supports the /predict endpoint."
                    )
                    st.code(str(exc))

    # -----------------------------------------------------
    # 7. DOWNLOADABLE REPORTS
    # -----------------------------------------------------

    with tabs[6]:
        st.subheader("Transaction Report")

        st.caption(
            "This is a demo risk-analysis report, not a bank-issued receipt."
        )

        result = st.session_state.last_result

        if not result:
            st.info("Analyse a transaction first to generate a report.")
        else:
            report = {
                "report_type": "FraudShield demo risk analysis",
                "demo_user_id": st.session_state.user_id,
                "transaction_result": result,
                "notice": (
                    "Not a bank-issued receipt or proof of payment."
                ),
            }

            report_json = json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
            )

            st.json(report)

            st.download_button(
                "Download JSON Report",
                data=report_json,
                file_name="fraudshield_report.json",
                mime="application/json",
                use_container_width=True,
            )

            st.download_button(
                "Download Text Report",
                data=(
                    "FRAUDSHIELD - DEMO RISK ANALYSIS\n\n"
                    + report_json
                    + "\n\nNot a bank-issued receipt or proof of payment.\n"
                ),
                file_name="fraudshield_report.txt",
                mime="text/plain",
                use_container_width=True,
            )

# =========================================================
# ADMIN LOGIN
# =========================================================

def render_admin_login():
    page_header(
        "Admin Login",
        "Restricted access to monitoring tools.",
    )

    if st.button("← Back to Home"):
        navigate("Home")

    if not ADMIN_USERNAME or not ADMIN_PASSWORD:
        st.warning(
            "Admin credentials are missing. Set FRAUDSHIELD_ADMIN_USER "
            "and FRAUDSHIELD_ADMIN_PASSWORD in .env."
        )
        return

    with st.form("admin_login_form"):
        username = st.text_input("Admin Username")
        password = st.text_input("Admin Password", type="password")

        login = st.form_submit_button(
            "Sign In",
            use_container_width=True,
        )

    if login:
        if (
            username == ADMIN_USERNAME
            and password == ADMIN_PASSWORD
        ):
            st.session_state.admin_logged_in = True
            navigate("Admin Dashboard")
        else:
            st.error("Invalid username or password.")

# =========================================================
# ADMIN DASHBOARD
# =========================================================

def render_admin():
    if not st.session_state.admin_logged_in:
        navigate("Admin Login")
        return

    st_autorefresh(interval=5000, key="admin_refresh")

    left, right = st.columns([4, 1])

    left.title("FraudShield Admin")
    left.caption("Transaction monitoring and fraud analytics")

    if right.button("Log Out", use_container_width=True):
        st.session_state.admin_logged_in = False
        navigate("Home")

    df = load_transactions()

    total = len(df)

    if df.empty:
        suspicious_count = 0
        safe_count = 0
        total_amount = 0.0
    else:
        suspicious_count = (
            int((df["status"] == "SUSPICIOUS").sum())
            if "status" in df.columns else 0
        )

        safe_count = (
            int(df["status"].isin(["SAFE", "NORMAL"]).sum())
            if "status" in df.columns else 0
        )

        total_amount = (
            float(df["amount"].fillna(0).sum())
            if "amount" in df.columns else 0.0
        )

    tabs = st.tabs([
        "📊 Overview",
        "🚨 Fraud Alerts",
        "📃 Transactions",
        "📈 Analytics & Reports",
        "⚙️ System Health",
    ])

    # ADMIN OVERVIEW
    with tabs[0]:
        a, b, c, d = st.columns(4)

        a.metric("Processed Transactions", total)
        b.metric("Suspicious", suspicious_count)
        c.metric("Safe / Normal", safe_count)
        d.metric("Total Transaction Value", f"₹{total_amount:,.2f}")

        if df.empty:
            st.info("No processed transaction records found.")
        else:
            st.subheader("Recent Transactions")

            st.dataframe(
                df.iloc[::-1].head(20),
                use_container_width=True,
                hide_index=True,
            )

            if "status" in df.columns:
                st.subheader("Transaction Distribution")
                st.bar_chart(df["status"].value_counts())

    # ADMIN ALERTS
    with tabs[1]:
        st.subheader("Suspicious Transactions")

        if df.empty or "status" not in df.columns:
            st.info("No processed transaction data available.")
        else:
            alerts = df[
                df["status"] == "SUSPICIOUS"
            ].iloc[::-1]

            if alerts.empty:
                st.success(
                    "No suspicious classifications in the current history."
                )
            else:
                st.warning(
                    f"{len(alerts)} transaction(s) flagged."
                )

                st.dataframe(
                    alerts,
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    "These are model classifications, not independently "
                    "verified fraud cases."
                )

    # ALL TRANSACTIONS
    with tabs[2]:
        st.subheader("All Processed Transactions")

        if df.empty:
            st.info("No transactions found.")
        else:
            filtered = df.copy()

            if "status" in filtered.columns:
                options = ["All"] + sorted(
                    filtered["status"].dropna().unique().tolist()
                )

                chosen = st.selectbox(
                    "Filter by Status",
                    options,
                )

                if chosen != "All":
                    filtered = filtered[
                        filtered["status"] == chosen
                    ]

            st.dataframe(
                filtered.iloc[::-1],
                use_container_width=True,
                hide_index=True,
                height=450,
            )

    # ANALYTICS AND REPORTS
    with tabs[3]:
        st.subheader("Analytics & Reports")

        if df.empty:
            st.info(
                "Analytics will appear when transactions are processed."
            )
        else:
            left, right = st.columns(2)

            if "status" in df.columns:
                with left:
                    st.write("Transactions by Status")
                    st.bar_chart(df["status"].value_counts())

            if "merchant" in df.columns:
                with right:
                    st.write("Transactions by Merchant")
                    st.bar_chart(
                        df["merchant"].value_counts().head(10)
                    )

            if "amount" in df.columns:
                st.write("Transaction Amounts")
                st.line_chart(
                    df["amount"].reset_index(drop=True)
                )

            st.download_button(
                "Download CSV Report",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="fraudshield_transactions.csv",
                mime="text/csv",
            )

    # SYSTEM HEALTH
    with tabs[4]:
        st.subheader("System Health")

        try:
            response = requests.get(
                f"{API_URL}/health",
                timeout=3,
            )

            if response.ok:
                st.success("FastAPI is responding")
                st.json(response.json())
            else:
                st.error(
                    f"API returned HTTP {response.status_code}"
                )

        except requests.RequestException as exc:
            st.error("FastAPI is unavailable")
            st.caption(str(exc))

        if DATA_FILE.exists():
            st.success("Transaction data file found")
            st.write(
                f"File size: {DATA_FILE.stat().st_size:,} bytes"
            )
        else:
            st.warning("Transaction data file not found yet.")

        model_path = BASE_DIR / "models" / "fraud_model.pkl"

        if model_path.exists():
            st.success("ML model file exists")
        else:
            st.warning("ML model file not found")

        st.write("**API Endpoints**")

        st.code(
            f"API:    {API_URL}\n"
            f"Docs:   {API_URL}/docs\n"
            f"Health: {API_URL}/health"
        )

# =========================================================
# APPLICATION ROUTER
# =========================================================

if st.session_state.page == "Home":
    render_home()

elif st.session_state.page == "User":
    render_user()

elif st.session_state.page == "Admin Login":
    render_admin_login()

elif st.session_state.page == "Admin Dashboard":
    render_admin()

else:
    st.session_state.page = "Home"
    st.rerun()