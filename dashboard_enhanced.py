import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title='EcoAir Agent', page_icon='🌿', layout='wide')

# ---------------- THEME ----------------
st.markdown('''
<style>
.stApp{background:#F7F0E6;color:#3B2921}
[data-testid="stHeader"]{background:rgba(247,240,230,.92)}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#3B2921,#6F4E37)}
[data-testid="stSidebar"] *{color:#FFF8EF!important}
.block-container{padding-top:1.5rem;max-width:1400px}
h1,h2,h3{color:#4A2F24!important}
.hero{background:linear-gradient(135deg,#4A2F24,#6F4E37,#9A6B4F);padding:28px 34px;border-radius:24px;color:white;box-shadow:0 12px 30px rgba(74,47,36,.2);margin-bottom:18px}
.hero h1{color:white!important;margin:0;font-size:2.35rem}.hero p{color:#F7EBDD;margin:7px 0 0}.badge{display:inline-block;background:#E7C9A9;color:#4A2F24;padding:5px 13px;border-radius:999px;font-size:.78rem;font-weight:700;margin-bottom:10px}
div.stButton>button{border-radius:12px;border:1px solid #D9C2AA;background:#FFF9F1;color:#4A2F24;font-weight:650;min-height:42px}
div.stButton>button:hover{border-color:#9A6B4F;color:#6F4E37}
.card{background:#FFF9F1;border:1px solid #E4D3C1;border-radius:18px;padding:20px;box-shadow:0 7px 20px rgba(91,61,44,.08);height:100%}
.card-title{color:#6F4E37;font-size:.82rem;text-transform:uppercase;letter-spacing:.8px;font-weight:750}.card-value{color:#3B2921;font-size:1.75rem;font-weight:800}.card-subtitle{color:#8A6B57;font-size:.85rem}
.normal{background:#EAF5EC;border:1px solid #B9DDBF;color:#27643A;border-radius:18px;padding:22px}.elevated{background:#FFF3D8;border:1px solid #E6C77A;color:#805E12;border-radius:18px;padding:22px}.high{background:#FBE8E5;border:1px solid #E3A7A0;color:#8A3028;border-radius:18px;padding:22px}
.pipeline{display:flex;gap:9px;align-items:stretch;margin:18px 0}.step{flex:1;background:#FFF9F1;border:1px solid #E4D3C1;border-radius:15px;padding:14px 10px;text-align:center;box-shadow:0 5px 14px rgba(91,61,44,.06)}.icon{font-size:1.35rem}.label{color:#5B3D2C;font-weight:750;font-size:.9rem;margin-top:5px}.small{color:#927663;font-size:.73rem}.arrow{display:flex;align-items:center;color:#A27A5E;font-size:1.2rem}
.banner{background:#EDE0D1;border-left:5px solid #8B6248;border-radius:12px;padding:13px 16px;color:#543A2C;margin:14px 0}
[data-testid="stMetric"]{background:#FFF9F1;border:1px solid #E4D3C1;border-radius:15px;padding:12px}
.footer{text-align:center;color:#9A7B66;font-size:.78rem;padding:25px 0 5px}
#MainMenu,footer{visibility:hidden}
</style>
''', unsafe_allow_html=True)

@st.cache_resource
def load_model(): return joblib.load('ecoair_random_forest.pkl')

@st.cache_data
def load_data():
    X=pd.read_csv('data/X_features.csv')
    y=pd.read_csv('data/y_target.csv')
    return X, y.iloc[:,0] if isinstance(y,pd.DataFrame) else y

model=load_model(); X,y=load_data()
NORMAL_THRESHOLD=1.1; HIGH_THRESHOLD=2.9
MAE=0.2489; RMSE=0.3979; R2=0.9240

def decide(co):
    if co<=NORMAL_THRESHOLD:return 'NORMAL','MONITOR','Air quality is within the normal project range.'
    if co<=HIGH_THRESHOLD:return 'ELEVATED','WARNING','CO is elevated. The agent recommends monitoring.'
    return 'HIGH','ALERT','CO is high according to the project decision policy.'

def show_status(status,action,text):
    cls={'NORMAL':'normal','ELEVATED':'elevated','HIGH':'high'}[status]
    icon={'NORMAL':'🟢','ELEVATED':'🟡','HIGH':'🔴'}[status]
    st.markdown(f'<div class="{cls}"><b style="font-size:1.25rem">{icon} {status}</b><br><b>Agent Action: {action}</b><br><span>{text}</span></div>',unsafe_allow_html=True)

pages=['🏠 Overview','🤖 Agent Monitor','⏱️ Simulation','📊 Model Insights','ℹ️ About']
if 'page' not in st.session_state: st.session_state.page=pages[0]

with st.sidebar:
    st.markdown('<div style="text-align:center;padding:10px 0 18px"><div style="font-size:3rem">🌿</div><div style="font-size:1.35rem;font-weight:800">EcoAir Agent</div><div style="font-size:.8rem">AI Environmental Monitor</div></div>',unsafe_allow_html=True)
    st.markdown('### Navigate')
    for page in pages:
        if st.button(page,key='side_'+page,width="stretch"):
            st.session_state.page=page; st.rerun()
    st.divider(); st.caption('Decision policy'); st.write('🟢 ≤ 1.1 → Monitor'); st.write('🟡 1.1–2.9 → Warning'); st.write('🔴 > 2.9 → Alert')

st.markdown('<div class="hero"><span class="badge">AI AGENT • ENVIRONMENTAL MONITORING</span><h1>🌿 EcoAir Agent</h1><p>Perceive environmental conditions • Predict CO • Make an intelligent monitoring decision</p></div>',unsafe_allow_html=True)
nav=st.columns(len(pages))
for i,p in enumerate(pages):
    with nav[i]:
        if st.button(p,key='top_'+p,width="stretch"): st.session_state.page=p; st.rerun()

if st.session_state.page=='🏠 Overview':
    st.header('Dashboard Overview')
    st.markdown('<div class="banner"><b>EcoAir Agent</b> receives environmental sensor readings, uses a trained Random Forest model to predict CO concentration, and applies a decision policy to select Monitor, Warning, or Alert.</div>',unsafe_allow_html=True)
    a,b,c,d=st.columns(4)
    for col,label,value,sub in [(a,'Records','7,674','usable observations'),(b,'Model','RF','100-tree ensemble'),(c,'R²','0.924','test-set performance'),(d,'Features','11','sensor/environment inputs')]:
        with col: st.markdown(f'<div class="card"><div class="card-title">{label}</div><div class="card-value">{value}</div><div class="card-subtitle">{sub}</div></div>',unsafe_allow_html=True)
    st.subheader('Agent Pipeline')
    pipeline_items = [
        ('📡','PERCEIVE','Sensor readings'),
        ('🧠','PREDICT','Random Forest'),
        ('⚖️','DECIDE','Decision policy'),
        ('🚨','ACT','Monitor / Warning / Alert'),
        ('🔄','REPEAT','Next observation')
    ]
    pipeline_html = '<div class="pipeline">'
    for i, (ic, la, sm) in enumerate(pipeline_items):
        pipeline_html += f'<div class="step"><div class="icon">{ic}</div><div class="label">{la}</div><div class="small">{sm}</div></div>'
        if i < len(pipeline_items) - 1:
            pipeline_html += '<div class="arrow">→</div>'
    pipeline_html += '</div>'
    st.markdown(pipeline_html, unsafe_allow_html=True)
    st.subheader('Project Decision Policy')
    st.dataframe(pd.DataFrame({'CO Range':['CO ≤ 1.1','1.1 < CO ≤ 2.9','CO > 2.9'],'Level':['Normal','Elevated','High'],'Agent Action':['Monitor','Warning','Alert']}),width="stretch",hide_index=True)
    st.caption('Thresholds are project-level rules derived from the dataset distribution, not official health or regulatory limits.')
