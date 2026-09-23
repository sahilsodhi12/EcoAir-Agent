
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="EcoAir Agent",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# COFFEE + BEIGE UI
# ============================================================
st.markdown("""
<style>
    .stApp {
        background: #F7F0E6;
        color: #3B2921;
    }

    [data-testid="stHeader"] {
        background: rgba(247,240,230,.95);
    }

    /* Coffee sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg,#3B2921 0%,#50372A 52%,#684936 100%);
    }

    [data-testid="stSidebar"] * {
        color: #FFF8EF;
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    h1,h2,h3 {
        color: #4A2F24 !important;
    }

    /* Hero */
    .hero {
        background: linear-gradient(135deg,#40291F,#6F4E37,#9A6B4F);
        border-radius: 24px;
        padding: 28px 34px;
        color: white;
        box-shadow: 0 12px 30px rgba(74,47,36,.20);
        margin-bottom: 18px;
    }

    .hero h1 {
        color: white !important;
        margin: 0;
        font-size: 2.35rem;
    }

    .hero p {
        color: #F7EBDD;
        margin: 7px 0 0 0;
        font-size: 1.02rem;
    }

    .hero-badge {
        display: inline-block;
        background: #E7C9A9;
        color: #4A2F24;
        padding: 5px 13px;
        border-radius: 999px;
        font-size: .76rem;
        font-weight: 800;
        margin-bottom: 9px;
    }

    /* Top navigation */
    div.stButton > button {
        background: #F3E4D3;
        color: #4A2F24;
        border: 1px solid #D8BFA5;
        border-radius: 12px;
        font-weight: 700;
        min-height: 42px;
    }

    div.stButton > button:hover {
        background: #E8D1B8;
        color: #3B2921;
        border-color: #A97857;
    }

    /* Sidebar navigation */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div {
        gap: 6px;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        background: #F1DFCC;
        border: 1px solid #D5B99B;
        border-radius: 12px;
        padding: 9px 11px;
        margin-bottom: 5px;
        color: #4A2F24 !important;
        font-weight: 700;
        transition: .18s;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: #E7CCB1;
        border-color: #C39774;
        transform: translateX(2px);
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background: #CDA37F;
        border-color: #F1D3B3;
        box-shadow: 0 4px 12px rgba(0,0,0,.20);
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label p {
        color: #4A2F24 !important;
    }

    /* Cards */
    .card {
        background: #FFF9F1;
        border: 1px solid #E4D3C1;
        border-radius: 18px;
        padding: 19px;
        box-shadow: 0 7px 20px rgba(91,61,44,.08);
        height: 100%;
    }

    .card-title {
        color: #6F4E37;
        font-size: .78rem;
        text-transform: uppercase;
        letter-spacing: .8px;
        font-weight: 800;
        margin-bottom: 6px;
    }

    .card-value {
        color: #3B2921;
        font-size: 1.75rem;
        font-weight: 800;
    }

    .card-subtitle {
        color: #8A6B57;
        font-size: .83rem;
        margin-top: 3px;
    }

    /* Status */
    .status-normal,.status-elevated,.status-high {
        border-radius: 18px;
        padding: 21px;
        margin: 14px 0;
    }

    .status-normal {
        background:#EAF5EC;
        border:1px solid #B9DDBF;
        color:#27643A;
    }

    .status-elevated {
        background:#FFF3D8;
        border:1px solid #E6C77A;
        color:#805E12;
    }

    .status-high {
        background:#FBE8E5;
        border:1px solid #E3A7A0;
        color:#8A3028;
    }

    .status-title {
        font-size:1.25rem;
        font-weight:800;
    }

    .status-action {
        font-size:1.02rem;
        font-weight:700;
        margin-top:4px;
    }

    /* Pipeline */
    .pipeline {
        display:flex;
        gap:8px;
        align-items:stretch;
        margin:16px 0 22px 0;
    }

    .pipeline-step {
        flex:1;
        background:#FFF9F1;
        border:1px solid #E4D3C1;
        border-radius:15px;
        padding:13px 8px;
        text-align:center;
        box-shadow:0 5px 14px rgba(91,61,44,.06);
    }

    .pipeline-icon { font-size:1.3rem; }
    .pipeline-label {
        color:#5B3D2C;
        font-weight:800;
        font-size:.87rem;
        margin-top:4px;
    }

    .pipeline-small {
        color:#927663;
        font-size:.71rem;
        margin-top:2px;
    }

    .arrow {
        display:flex;
        align-items:center;
        color:#A27A5E;
        font-size:1.15rem;
    }

    .info-banner {
        background:#EDE0D1;
        border-left:5px solid #8B6248;
        border-radius:12px;
        padding:13px 16px;
        color:#543A2C;
        margin:12px 0 18px 0;
    }

    .nav-title {
        font-size:.76rem;
        letter-spacing:1.5px;
        color:#E1C2A5;
        font-weight:800;
        margin:8px 0;
    }

    .footer {
        text-align:center;
        color:#9A7B66;
        font-size:.76rem;
        padding:24px 0 5px;
    }

    #MainMenu, footer { visibility:hidden; }

    /* ===== FIX TEXT VISIBILITY ===== */

    /* Metric labels and values */
    [data-testid="stMetricLabel"] {
        color: #5A3E2B !important;
    }

    [data-testid="stMetricValue"] {
        color: #3B2618 !important;
    }

    [data-testid="stMetricDelta"] {
        color: #5A3E2B !important;
    }

    /* General Streamlit text */
    .stMarkdown,
    .stText,
    p,
    label {
        color: #3B2618;
    }

    /* Headings */
    h1, h2, h3, h4 {
        color: #3B2618 !important;
    }

    /* Sidebar text */
    section[data-testid="stSidebar"] * {
        color: #F5EBDD !important;
    }

    /* Buttons */
    .stButton button {
        color: #3B2618 !important;
    }

    /* Selectbox / input text */
    .stSelectbox label,
    .stNumberInput label,
    .stSlider label {
        color: #3B2618 !important;
    }

    /* Plotly chart text */
    .js-plotly-plot .plotly text {
        fill: #3B2618 !important;
    }   

    /* Chart axis titles and tick labels */
    .js-plotly-plot .xtick text,
    .js-plotly-plot .ytick text {
        fill: #3B2618 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# MODEL + DATA
# ============================================================
@st.cache_resource
def load_model():
    return joblib.load("ecoair_random_forest.pkl")

@st.cache_data
def load_data():
    X = pd.read_csv("data/X_features.csv")
    y = pd.read_csv("data/y_target.csv")
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    return X, y

model = load_model()
X, y = load_data()

# ============================================================
# SETTINGS
# ============================================================
NORMAL_THRESHOLD = 1.1
HIGH_THRESHOLD = 2.9
MAE = 0.2489
RMSE = 0.3979
R2 = 0.9240

pages = [
    "🏠 Overview",
    "🤖 Agent Monitor",
    "⏱️ Simulation",
    "📊 Model Insights",
    "ℹ️ About"
]

if "page" not in st.session_state:
    st.session_state.page = pages[0]

def make_decision(predicted_co):
    if predicted_co <= NORMAL_THRESHOLD:
        return "NORMAL", "MONITOR", "Air quality is within the normal project range."
    if predicted_co <= HIGH_THRESHOLD:
        return "ELEVATED", "WARNING", "CO is elevated. The agent recommends monitoring."
    return "HIGH", "ALERT", "CO is high according to the project decision policy."

def show_status(status, action, explanation):
    if status == "NORMAL":
        cls, icon = "status-normal", "🟢"
    elif status == "ELEVATED":
        cls, icon = "status-elevated", "🟡"
    else:
        cls, icon = "status-high", "🔴"

    st.markdown(
        f"""
        <div class="{cls}">
            <div class="status-title">{icon} {status}</div>
            <div class="status-action">Agent Action: {action}</div>
            <div style="margin-top:7px">{explanation}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# SIDEBAR — CLICKABLE NAVIGATION
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;padding:8px 0 17px">
            <div style="font-size:3rem">🌿</div>
            <div style="font-size:1.35rem;font-weight:800;color:#FFF8EF">
                EcoAir Agent
            </div>
            <div style="font-size:.78rem;color:#E7D2BC">
                AI Environmental Monitor
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="nav-title">NAVIGATE</div>', unsafe_allow_html=True)

    # This radio acts as the sidebar's clickable tab menu.
    selected = st.radio(
        "Dashboard sections",
        pages,
        index=pages.index(st.session_state.page),
        label_visibility="collapsed"
    )

    if selected != st.session_state.page:
        st.session_state.page = selected
        st.rerun()

    st.divider()

    st.markdown(
        """
        <div style="background:#3D2920;border:1px solid #80624C;
                    border-radius:14px;padding:13px;font-size:.82rem;
                    line-height:1.8;color:#FFF8EF">
            <b style="color:#E8C7A8">AGENT LOOP</b><br>
            📡 Perceive<br>
            🧠 Predict<br>
            ⚖️ Decide<br>
            🚨 Act<br>
            🔄 Repeat
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        '<div style="color:#E8C7A8;font-weight:800">DECISION POLICY</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div style="line-height:1.8;font-size:.84rem">
        🟢 ≤ 1.1 → Monitor<br>
        🟡 1.1–2.9 → Warning<br>
        🔴 > 2.9 → Alert
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
    <div class="hero">
        <span class="hero-badge">AI AGENT • ENVIRONMENTAL MONITORING</span>
        <h1>🌿 EcoAir Agent</h1>
        <p>Perceive environmental conditions • Predict CO • Make an intelligent monitoring decision</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Top clickable navigation
top_cols = st.columns(len(pages))
for i, page in enumerate(pages):
    with top_cols[i]:
        if st.button(page, key=f"top_nav_{i}", width="stretch"):
            st.session_state.page = page
            st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# OVERVIEW
# ============================================================
if st.session_state.page == "🏠 Overview":

    st.header("Dashboard Overview")

    st.markdown(
        """
        <div class="info-banner">
        <b>What is EcoAir Agent?</b><br>
        EcoAir Agent receives environmental sensor readings, uses a trained
        Random Forest model to predict CO concentration, and applies a
        decision policy to select Monitor, Warning, or Alert.
        </div>
        """,
        unsafe_allow_html=True
    )

    cols = st.columns(4)
    cards = [
        ("Records","7,674","usable observations"),
        ("Model","Random Forest","100-tree ensemble"),
        ("R²","0.924","test-set performance"),
        ("Features","11","sensor/environment inputs")
    ]

    for c, (title,value,sub) in zip(cols,cards):
        with c:
            st.markdown(
                f"""
                <div class="card">
                    <div class="card-title">{title}</div>
                    <div class="card-value">{value}</div>
                    <div class="card-subtitle">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.subheader("Agent Pipeline")
    st.markdown(
        """
        <div class="pipeline">
        <div class="pipeline-step"><div class="pipeline-icon">📡</div>
        <div class="pipeline-label">PERCEIVE</div><div class="pipeline-small">Sensor readings</div></div>
        <div class="arrow">→</div>
        <div class="pipeline-step"><div class="pipeline-icon">🧠</div>
        <div class="pipeline-label">PREDICT</div><div class="pipeline-small">Random Forest</div></div>
        <div class="arrow">→</div>
        <div class="pipeline-step"><div class="pipeline-icon">⚖️</div>
        <div class="pipeline-label">DECIDE</div><div class="pipeline-small">Decision policy</div></div>
        <div class="arrow">→</div>
        <div class="pipeline-step"><div class="pipeline-icon">🚨</div>
        <div class="pipeline-label">ACT</div><div class="pipeline-small">Monitor / Warning / Alert</div></div>
        <div class="arrow">→</div>
        <div class="pipeline-step"><div class="pipeline-icon">🔄</div>
        <div class="pipeline-label">REPEAT</div><div class="pipeline-small">Next observation</div></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Project Decision Policy")
    policy = pd.DataFrame({
        "CO Range":["CO ≤ 1.1","1.1 < CO ≤ 2.9","CO > 2.9"],
        "Level":["Normal","Elevated","High"],
        "Agent Action":["Monitor","Warning","Alert"]
    })
    st.dataframe(policy, width="stretch", hide_index=True)

    st.caption(
        "Thresholds are project-level thresholds derived from the dataset distribution, "
        "not official health or regulatory limits."
    )

# ============================================================
# AGENT MONITOR
# ============================================================
elif st.session_state.page == "🤖 Agent Monitor":

    st.header("🤖 Agent Monitor")
    st.write("Select a historical observation to simulate a sensor perception.")

    index = st.slider("Sensor Observation",0,len(X)-1,0,1)

    current = X.iloc[index]
    prediction = float(model.predict(current.to_frame().T)[0])
    status, action, explanation = make_decision(prediction)

    c1,c2,c3 = st.columns(3)
    with c1: st.metric("Predicted CO",f"{prediction:.2f}")
    with c2: st.metric("Agent Status",status)
    with c3: st.metric("Agent Action",action)

    show_status(status,action,explanation)

    st.subheader("📡 Environmental Perception")

    sensors = [
        ("CO Sensor","CO_Sensor_1",""),
        ("Benzene","Benzene_Concentration",""),
        ("NMHC Sensor","NMHC_Sensor",""),
        ("NOx","NOx_Concentration",""),
        ("NO2","NO2_Concentration",""),
        ("Temperature","Temperature_C"," °C"),
        ("Humidity","Relative_Humidity"," %"),
        ("Absolute Humidity","Absolute_Humidity","")
    ]

    sensor_cols = st.columns(4)
    for i,(label,feature,suffix) in enumerate(sensors):
        with sensor_cols[i%4]:
            st.markdown(
                f"""
                <div class="card" style="margin-bottom:14px">
                    <div class="card-title">{label}</div>
                    <div class="card-value" style="font-size:1.35rem">
                    {current[feature]:.2f}{suffix}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.subheader("🔄 Decision Process")
    st.markdown(
        f"""
        <div class="pipeline">
        <div class="pipeline-step"><div class="pipeline-icon">📡</div>
        <div class="pipeline-label">PERCEIVE</div><div class="pipeline-small">Observation #{index}</div></div>
        <div class="arrow">→</div>
        <div class="pipeline-step"><div class="pipeline-icon">🧠</div>
        <div class="pipeline-label">PREDICT</div><div class="pipeline-small">CO = {prediction:.2f}</div></div>
        <div class="arrow">→</div>
        <div class="pipeline-step"><div class="pipeline-icon">⚖️</div>
        <div class="pipeline-label">DECIDE</div><div class="pipeline-small">{status}</div></div>
        <div class="arrow">→</div>
        <div class="pipeline-step"><div class="pipeline-icon">🚨</div>
        <div class="pipeline-label">ACT</div><div class="pipeline-small">{action}</div></div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# SIMULATION
# ============================================================
elif st.session_state.page == "⏱️ Simulation":

    st.header("⏱️ Sequential Agent Simulation")

    st.markdown(
        """
        <div class="info-banner">
        Each observation is processed sequentially: perceive → predict →
        decide → act.
        </div>
        """,
        unsafe_allow_html=True
    )

    c1,c2 = st.columns(2)
    with c1:
        steps = st.slider("Number of observations",5,50,10)
    with c2:
        start = st.number_input(
            "Starting observation",
            0,
            max(0,len(X)-steps),
            0,
            1
        )

    if st.button("▶ Run Agent Simulation", width="stretch"):

        results=[]
        for i in range(int(start),int(start)+int(steps)):
            row=X.iloc[i]
            pred=float(model.predict(row.to_frame().T)[0])
            status,action,_=make_decision(pred)

            results.append({
                "Observation":i,
                "Predicted CO":round(pred,2),
                "Status":status,
                "Action":action
            })

        results_df=pd.DataFrame(results)

        st.subheader("Simulation Results")
        st.dataframe(results_df,width="stretch",hide_index=True)

        st.subheader("📈 Predicted CO Across Observations")

        fig,ax=plt.subplots(figsize=(10,4))
        ax.plot(
            results_df["Observation"],
            results_df["Predicted CO"],
            marker="o"
        )
        ax.axhline(NORMAL_THRESHOLD,linestyle="--",label="Normal threshold")
        ax.axhline(HIGH_THRESHOLD,linestyle="--",label="High threshold")
        ax.set_xlabel("Observation")
        ax.set_ylabel("Predicted CO")
        ax.set_title("Sequential EcoAir Agent Predictions")
        ax.legend()
        ax.grid(alpha=.2)
        st.pyplot(fig)

        a,b,c=st.columns(3)
        with a:
            st.metric("🟢 Normal",(results_df["Status"]=="NORMAL").sum())
        with b:
            st.metric("🟡 Elevated",(results_df["Status"]=="ELEVATED").sum())
        with c:
            st.metric("🔴 High",(results_df["Status"]=="HIGH").sum())

# ============================================================
# MODEL INSIGHTS
# ============================================================
elif st.session_state.page == "📊 Model Insights":

    st.header("📊 Model Insights")

    a,b,c=st.columns(3)
    with a: st.metric("MAE",f"{MAE:.4f}")
    with b: st.metric("RMSE",f"{RMSE:.4f}")
    with c: st.metric("R²",f"{R2:.4f}")

    st.markdown(
        """
        <div class="info-banner">
        <b>Random Forest Regressor</b> • Supervised Learning • Regression •
        100 trees • 80/20 train-test split
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("🔍 Feature Importance")

    importance=pd.DataFrame({
        "Feature":X.columns,
        "Importance":model.feature_importances_
    }).sort_values("Importance",ascending=False)

    st.dataframe(importance,width="stretch",hide_index=True)

    fig,ax=plt.subplots(figsize=(9,5))
    top=importance.head(10).sort_values("Importance")
    ax.barh(top["Feature"],top["Importance"])
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest Feature Importance")
    ax.grid(axis="x",alpha=.2)
    st.pyplot(fig)

    st.subheader("📈 CO Concentration Distribution")

    fig2,ax2=plt.subplots(figsize=(9,4))
    ax2.hist(y,bins=30)
    ax2.set_xlabel("CO Concentration")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution of CO Concentration")
    ax2.grid(axis="y",alpha=.2)
    st.pyplot(fig2)

# ============================================================
# ABOUT
# ============================================================
elif st.session_state.page == "ℹ️ About":

    st.header("ℹ️ About EcoAir Agent")

    st.markdown(
        """
        <div class="card">
        <div class="card-title">Agent Goal</div>
        <div style="font-size:1.05rem">
        Monitor environmental conditions, predict CO concentration,
        and make an appropriate monitoring decision.
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("🌍 Russell & Norvig Environment")

    env=pd.DataFrame({
        "Property":["Observable","Deterministic","Episodic","Static","Discrete","Agents"],
        "EcoAir":["Partially Observable","Stochastic","Sequential","Dynamic","Continuous","Single-Agent"]
    })

    st.dataframe(env,width="stretch",hide_index=True)

    st.subheader("🧠 Agent Architecture")

    st.code(
        """
Environment
     ↓
PERCEIVE
     ↓
Sensor Readings
     ↓
RANDOM FOREST
     ↓
Predicted CO
     ↓
DECISION POLICY
     ↓
Monitor / Warning / Alert
     ↓
Environment
        """,
        language="text"
    )

    st.markdown(
        """
        <div class="info-banner">
        <b>The Random Forest is not the agent.</b><br>
        It is the learned model inside the agent. The agent combines
        perception, prediction and a decision policy to select an action.
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div class="footer">
    🌿 EcoAir Agent • AI Environmental Monitoring System • Random Forest Decision Agent
    </div>
    """,
    unsafe_allow_html=True
)
