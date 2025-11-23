import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CSS (Dark Mode Sidebar Fix)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="PrediMaint | AI Smart Factory",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #f8f9fa;
    }
    
    /* Main Header */
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        text-align: center;
    }
    
    /* KPI Cards */
    .kpi-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
        text-align: center;
        transition: transform 0.2s;
    }
    .kpi-card:hover { transform: translateY(-5px); box-shadow: 0 10px 20px rgba(0,0,0,0.1); }
    .kpi-title { font-size: 0.9rem; color: #666; font-weight: 600; text-transform: uppercase; }
    .kpi-value { font-size: 1.8rem; font-weight: 700; color: #1e3c72; margin: 10px 0; }
    
    /* SIDEBAR STYLING - FORCED DARK MODE & WHITE TEXT */
    [data-testid="stSidebar"] {
        background-color: #000000;
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }
    /* Fix for selectbox arrow in sidebar to be visible */
    [data-testid="stSidebar"] .stSelectbox svg {
        fill: white !important;
    }
    
    .alert-box {
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        border-left: 5px solid;
    }
    .alert-high { background-color: #ffebee; border-color: #ef5350; color: #c62828; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA LOADING (Robust)
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("ai4i2020.csv")
        # Clean column names (remove hidden spaces)
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_model(model_path):
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None

df = load_data()

# -----------------------------------------------------------------------------
# 3. PREPROCESSING (Fixes 'M' String Error)
# -----------------------------------------------------------------------------
def preprocess_input(df_input):
    data = df_input.copy()
    
    # 1. Force Type to String -> Strip Spaces -> Map to Int
    # This prevents the "could not convert string to float: 'M'" error
    if 'Type' in data.columns:
        data['Type'] = data['Type'].astype(str).str.strip()
        type_map = {'H': 0, 'L': 1, 'M': 2}
        data['Type'] = data['Type'].map(type_map)
        # Fill any missing mappings with mode (usually 1/L) to prevent NaN crash
        data['Type'] = data['Type'].fillna(1).astype(int)
    
    # Feature Sets
    cols_6 = ['Type', 'Air temperature [K]', 'Process temperature [K]', 
              'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    
    cols_10 = ['Air temperature [K]', 'Process temperature [K]', 
               'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
               'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    
    # Create Dataframes
    X_6 = data[cols_6]
    X_10 = data[cols_10] if all(c in data.columns for c in cols_10) else pd.DataFrame()

    # Scale Data
    scaler_6 = StandardScaler().fit(X_6)
    X_6_scaled = scaler_6.transform(X_6)
    
    X_10_scaled = None
    if not X_10.empty:
        scaler_10 = StandardScaler().fit(X_10)
        X_10_scaled = scaler_10.transform(X_10)
    
    return {
        "X_6_raw": X_6, 
        "X_6_scaled": X_6_scaled,
        "X_10_scaled": X_10_scaled
    }

if not df.empty:
    preprocessed = preprocess_input(df)
    y_true = df['Machine failure']

# -----------------------------------------------------------------------------
# 4. SIDEBAR & PREDICTION ENGINE
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("PrediMaint AI")
    st.caption("Industrial IoT Monitor V2.1")
    st.markdown("---")
    
    st.subheader("🤖 AI Model Engine")
    model_options = {
        "Random Forest": "random_forest.pkl",
        "Gradient Boosting": "gradient_boosting.pkl",
        "Support Vector Machine": "svm.pkl",
        "MLP Neural Network": "mlp.pkl"
    }
    selected_model_name = st.selectbox("Active Model", list(model_options.keys()), label_visibility="collapsed")
    model = load_model(model_options[selected_model_name])

    # --- INTELLIGENT PREDICTION LOGIC ---
    probs = np.zeros(len(df)) if not df.empty else []
    
    if model is not None and not df.empty:
        # Ask the model: "How many features do you need?"
        # Default to 6 if the model doesn't say.
        req_features = getattr(model, "n_features_in_", 6)
        
        X_input = None
        
        # Select the correct input based on model requirement
        if req_features == 10:
            if preprocessed["X_10_scaled"] is not None:
                X_input = preprocessed["X_10_scaled"]
            else:
                st.error("Model needs 10 features, but failure columns are missing.")
        elif req_features == 6:
            # Check if model prefers Raw (Trees) or Scaled (SVM/MLP)
            if "SVM" in selected_model_name or "MLP" in selected_model_name:
                X_input = preprocessed["X_6_scaled"]
            else:
                X_input = preprocessed["X_6_raw"]
        
        # Run Prediction if input is valid
        if X_input is not None:
            try:
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_input)[:, 1]
                else:
                    # Fallback for models without probabilities
                    probs = model.predict(X_input).astype(float)
            except Exception as e:
                st.error(f"Prediction Error: {e}")
                probs = np.zeros(len(df))

        # Store Results
        df['Failure Probability'] = probs
        df['Risk Category'] = df['Failure Probability'].apply(lambda p: 'High' if p > 0.7 else ('Medium' if p > 0.4 else 'Low'))
        df['Predicted RUL'] = (250 - df['Tool wear [min]']).clip(lower=0)

    # --- SIDEBAR BENCHMARK CHART ---
    if not df.empty:
        st.markdown("### ⚡ Performance")
        y_pred_binary = (probs > 0.5).astype(int)
        acc = accuracy_score(y_true, y_pred_binary)
        baseline = 1 - y_true.mean()
        
        fig_bench = go.Figure()
        fig_bench.add_trace(go.Bar(
            y=['Baseline', 'AI Model'], x=[baseline, acc],
            orientation='h', marker_color=['#4b5563', '#22c55e'],
            text=[f"{baseline:.1%}", f"{acc:.1%}"], textposition='auto'
        ))
        fig_bench.update_layout(
            margin=dict(l=0, r=0, t=0, b=0), height=80,
            xaxis=dict(showgrid=False, showticklabels=False, range=[0.8, 1.0]),
            yaxis=dict(showgrid=False),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=11)
        )
        st.plotly_chart(fig_bench, use_container_width=True, config={'displayModeBar': False})

    st.markdown("---")
    st.subheader("🧭 Navigation")
    page = st.radio("Go to", ["Dashboard Overview", "Live Machine Monitor", "Model Intelligence", "About System"])
    
    st.markdown("---")
    with st.expander("ℹ️ Legend"):
        st.markdown("""
        * 🔴 **Critical:** > 70%
        * 🟡 **Warning:** 40-70%
        * 🟢 **Normal:** < 40%
        """)
    st.info(f"Units Online: {len(df)}")

# -----------------------------------------------------------------------------
# 5. MAIN DASHBOARD CONTENT
# -----------------------------------------------------------------------------

st.markdown(f"""
<div class="main-header">
    <h1>🏭 Intelligent Factory Monitor</h1>
    <p>Real-time Predictive Maintenance powered by {selected_model_name}</p>
</div>
""", unsafe_allow_html=True)

if page == "Dashboard Overview":
    total = len(df)
    high = len(df[df['Risk Category']=='High'])
    med = len(df[df['Risk Category']=='Medium'])
    safe = len(df[df['Risk Category']=='Low'])
    
    cols = st.columns(4)
    kpis = [("Total Assets", total, "#1e3c72"), ("Critical", high, "#c62828"), 
            ("Warning", med, "#f9a825"), ("Healthy", safe, "#2e7d32")]
    
    for i, (lbl, val, clr) in enumerate(kpis):
        with cols[i]:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">{lbl}</div>
                <div class="kpi-value" style="color: {clr}">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("###")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("📋 Live Fleet Status")
        st.dataframe(
            df[['UDI', 'Type', 'Torque [Nm]', 'Tool wear [min]', 'Failure Probability', 'Risk Category']]
            .sort_values('Failure Probability', ascending=False).head(100),
            column_config={
                "Failure Probability": st.column_config.ProgressColumn("Risk Score", format="%.2f", min_value=0, max_value=1),
                "Risk Category": st.column_config.TextColumn("Status")
            }, height=350, use_container_width=True
        )
    with c2:
        st.subheader("📊 Risk Profile")
        fig_donut = px.pie(df, names='Risk Category', hole=0.6, color='Risk Category',
                           color_discrete_map={'High':'#ef5350', 'Medium':'#ffca28', 'Low':'#66bb6a'})
        fig_donut.update_layout(showlegend=False, margin=dict(t=0,b=0,l=0,r=0), height=250)
        fig_donut.add_annotation(text=f"{high}", showarrow=False, font_size=25, font_color="#ef5350", y=0.55)
        fig_donut.add_annotation(text="Critical", showarrow=False, font_size=12, y=0.45)
        st.plotly_chart(fig_donut, use_container_width=True)

    if high > 0:
        st.subheader("🔥 Active Alerts")
        for _, row in df[df['Risk Category'] == 'High'].head(3).iterrows():
            st.markdown(f"""<div class="alert-box alert-high">
            <strong>UDI {row['UDI']}</strong> | Risk: {row['Failure Probability']:.0%} | Rec: Inspect Tool</div>""", unsafe_allow_html=True)

elif page == "Live Machine Monitor":
    st.markdown("### 🔍 Real-Time Simulation")
    c1, c2 = st.columns([1, 3])
    with c1:
        sel_udi = st.selectbox("Select Machine ID", df['UDI'].unique())
        row = df[df['UDI'] == sel_udi].iloc[0]
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-title">Current Status</div>
            <div class="kpi-value" style="color:{'red' if row['Risk Category']=='High' else 'green'}">{row['Risk Category']}</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("###"); run_sim = st.button("▶️ Run Simulation", type="primary", use_container_width=True)

    with c2:
        sim_box = st.empty()
        if run_sim:
            data = []
            for i in range(20):
                data.append({
                    "Time": i,
                    "Torque": row['Torque [Nm]'] * (1 + np.random.uniform(-0.1, 0.1)),
                    "Temp": row['Process temperature [K]'] * (1 + np.random.uniform(-0.02, 0.05))
                })
                pdf = pd.DataFrame(data)
                with sim_box.container():
                    sc1, sc2 = st.columns(2)
                    sc1.plotly_chart(px.line(pdf, x='Time', y='Torque', title='Live Torque'), use_container_width=True)
                    sc2.plotly_chart(px.line(pdf, x='Time', y='Temp', title='Live Temp', color_discrete_sequence=['red']), use_container_width=True)
                time.sleep(0.1)
        else:
            sim_box.info("Select a machine and click Run Simulation.")

elif page == "Model Intelligence":
    st.subheader("🧠 Model Analysis")
    c1, c2 = st.columns(2)
    y_p = (df['Failure Probability'] > 0.5).astype(int)
    with c1:
        st.markdown("**Confusion Matrix**")
        st.plotly_chart(px.imshow(confusion_matrix(y_true, y_p), text_auto=True, color_continuous_scale='Blues'), use_container_width=True)
    with c2:
        st.markdown("**ROC Curve**")
        fpr, tpr, _ = roc_curve(y_true, df['Failure Probability'])
        st.plotly_chart(px.area(x=fpr, y=tpr, title=f"AUC: {auc(fpr, tpr):.3f}"), use_container_width=True)

elif page == "About System":
    st.image("https://images.unsplash.com/photo-1581091226825-a6a2a5aee158?auto=format&fit=crop&w=1000&q=80", use_container_width=True)
    st.markdown("### System Info\nPredictive Maintenance Dashboard v2.1")