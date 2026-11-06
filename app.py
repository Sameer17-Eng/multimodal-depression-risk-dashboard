import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Depression Risk Assessment Dashboard", page_icon="📊", layout="wide")
st.title("Multimodal AI System for Depression Risk Assessment")
st.caption("Real-time monitoring dashboard prototype | ESP32 wearable sensors + laptop camera")
st.warning("Research prototype only. This dashboard does not diagnose depression or provide a clinically validated risk probability.")

def demo_data(n=90):
    rng=np.random.default_rng(21)
    t=pd.date_range(datetime.now()-timedelta(minutes=n-1), periods=n, freq="min")
    return pd.DataFrame({
        "timestamp":t,
        "heart_rate_bpm":np.clip(76+rng.normal(0,3,n).cumsum()*.12+rng.normal(0,1.5,n),55,115),
        "eda_us":np.clip(2.8+rng.normal(0,.18,n).cumsum()*.1+rng.normal(0,.12,n),.4,8),
        "activity_index":np.clip(rng.normal(.4,.18,n),0,1.5),
        "facial_feature_score":np.clip(rng.normal(.55,.08,n),0,1),
        "signal_quality":np.clip(rng.normal(.92,.03,n),.5,1)
    })

def load_data(file):
    d=pd.read_csv(file)
    required=["timestamp","heart_rate_bpm","eda_us","activity_index"]
    missing=[x for x in required if x not in d.columns]
    if missing: raise ValueError("Missing CSV columns: "+", ".join(missing))
    d["timestamp"]=pd.to_datetime(d["timestamp"],errors="coerce")
    for c,default in [("facial_feature_score",.5),("signal_quality",.9)]:
        if c not in d: d[c]=default
    for c in required[1:]+["facial_feature_score","signal_quality"]:
        d[c]=pd.to_numeric(d[c],errors="coerce")
    return d.dropna(subset=required).sort_values("timestamp")

def pattern_index(d):
    r=d.iloc[-1]
    if len(d)<10 or r.signal_quality<.65: return None,"Insufficient data"
    # Transparent placeholder for UI demonstration, not a trained clinical model.
    parts=[np.clip((r.heart_rate_bpm-60)/45,0,1),
           np.clip(r.eda_us/8,0,1),
           1-np.clip(r.activity_index/1.2,0,1),
           1-np.clip(r.facial_feature_score,0,1)]
    score=round(100*float(np.mean(parts)))
    label="Lower pattern index" if score<35 else "Intermediate pattern index" if score<65 else "Higher pattern index"
    return score,label

st.sidebar.header("Data controls")
mode=st.sidebar.radio("Choose data source",["Demo simulation","Upload CSV"])
if mode=="Upload CSV":
    f=st.sidebar.file_uploader("Upload sensor CSV",type="csv")
    if f:
        try: data=load_data(f)
        except Exception as e:
            st.error(str(e)); data=demo_data()
    else:
        st.info("Upload a CSV to view recorded readings. Synthetic demo values are shown meanwhile.")
        data=demo_data()
else:
    data=demo_data()
st.sidebar.markdown("**CSV format**")
st.sidebar.code("timestamp,heart_rate_bpm,eda_us,activity_index\\n2026-11-05 10:00:00,78,2.5,0.32")
st.sidebar.caption("Optional: signal_quality (0–1), facial_feature_score (0–1).")

latest=data.iloc[-1]
score,label=pattern_index(data)
a,b,c=st.columns(3)
a.metric("Data source","Synthetic demo" if mode=="Demo simulation" else "CSV upload")
b.metric("Last sample",pd.to_datetime(latest.timestamp).strftime("%H:%M:%S"))
c.metric("Signal quality",f"{latest.signal_quality*100:.0f}%")

st.subheader("Live sensor overview")
cols=st.columns(4)
cols[0].metric("Heart rate",f"{latest.heart_rate_bpm:.0f} BPM","MAX30102 · PPG")
cols[1].metric("Skin conductance",f"{latest.eda_us:.2f} µS","GSR / EDA")
cols[2].metric("Activity index",f"{latest.activity_index:.2f}","MPU6050")
cols[3].metric("Facial feature score",f"{latest.facial_feature_score:.2f}","Optional camera feature")

left,right=st.columns([1,1.5])
with left:
    st.subheader("AI assessment")
    if score is None:
        st.info("Insufficient data. Check sensor quality and collect a longer observation window.")
    else:
        color="#2e9d69" if score<35 else "#d99b26" if score<65 else "#c84b4b"
        fig=go.Figure(go.Indicator(mode="gauge+number",value=score,title={"text":"Illustrative pattern index / 100"},
            gauge={"axis":{"range":[0,100]},"bar":{"color":color},
                   "steps":[{"range":[0,35],"color":"#e6f4eb"},{"range":[35,65],"color":"#fff2d6"},{"range":[65,100],"color":"#f9e5e5"}]}))
        fig.update_layout(height=240,margin=dict(l=15,r=15,t=55,b=10))
        st.plotly_chart(fig,use_container_width=True)
        st.markdown(f"**Category:** {label}")
    st.caption("This placeholder index is a simple demonstration formula, not a trained or clinically validated depression model.")
with right:
    st.subheader("Sensor trends")
    tabs=st.tabs(["Heart rate","EDA","Activity"])
    for tab,col,title,y in zip(tabs,["heart_rate_bpm","eda_us","activity_index"],["Heart rate","Skin conductance","Activity index"],["BPM","µS","Index"]):
        with tab:
            fig=px.line(data,x="timestamp",y=col,labels={"timestamp":"Time",col:y})
            fig.update_layout(height=260,margin=dict(l=10,r=10,t=15,b=10),title=title)
            st.plotly_chart(fig,use_container_width=True)

st.divider()
st.subheader("Supportive recommendations")
st.caption("General wellbeing suggestions only; not medical treatment.")
if score is None:
    st.info("Recommendations paused because data is insufficient.")
elif score<35:
    recs=["Maintain regular sleep and daily routines.","Continue enjoyable physical activity and social connection.","Review longer-term trends rather than one reading."]
elif score<65:
    recs=["Consider a short relaxation or breathing break.","Keep regular meals, sleep timing and gentle activity.","If distress persists, talk with someone you trust or a qualified professional."]
else:
    recs=["Reach out to a trusted person for support.","Consider contacting a qualified mental health professional.","If you may harm yourself or are in immediate danger, seek urgent local emergency help."]
for item in recs: st.markdown("- "+item)

x,y=st.columns(2)
with x:
    st.subheader("Data quality")
    st.write("Heart-rate, EDA and activity values:", "Available" if len(data)>0 else "Missing")
    st.write("Signal quality:", "Good" if latest.signal_quality>=.65 else "Needs checking")
    st.write("Observation samples:",len(data))
with y:
    st.subheader("Export")
    st.download_button("Download displayed data (CSV)",data.to_csv(index=False).encode(), "sensor_readings.csv","text/csv")

with st.expander("Hardware integration notes"):
    st.markdown("""
    **Current build:** Demo simulation and CSV upload. ESP32 Wi-Fi and camera streams are not connected yet.

    Suggested ESP32 JSON message (one record per timestamp):
    ```json
    {"timestamp":"2026-11-05T10:00:00","heart_rate_bpm":78,"eda_us":2.5,"activity_index":0.32,"signal_quality":0.94}
    ```
    A Python receiver can collect ESP32 messages and feed them into this dashboard. Camera-derived features should be generated locally and synchronized by timestamp.

    MAX30102 provides PPG pulse timing; derive quality-controlled pulse-rate variability (PRV), not ECG HRV. MPU6050 provides accelerometer/gyroscope data from which activity features must be calculated.
    """)

st.caption("Synthetic demo values are not experimental results. Secure consent and protect sensor/camera data in any real deployment.")
