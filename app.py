import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
import plotly.express as px

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Lung Cancer Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. LOAD ACTUAL ML MODEL & ASSETS ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("best_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except FileNotFoundError as e:
        return None, None

model, scaler = load_assets()

# Your exact genes from the training process
BIOMARKERS = ['PTPN21', 'RTKN2', 'CAT', 'C13orf36', 'LOC158376', 
              'GPM6A', 'UBE2T', 'RECQL4', 'RXFP1', 'C16orf59']

# --- 3. SESSION STATE INIT & RESET LOGIC ---
if 'results' not in st.session_state:
    st.session_state.results = None

# THE FIX: This function directly forces the widget keys back to None
def clear_form():
    st.session_state.results = None
    for gene in BIOMARKERS:
        st.session_state[gene] = None

# --- 4. ADVANCED CSS INJECTION ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* Global Theme variables */
:root {
    --bg-color: #0c111d;
    --text-color: #f1f5f9;
    --primary: #2dd4bf;
    --accent: #38bdf8;
    --destructive: #ef4444;
    --success: #10b981;
    --card-bg: rgba(16, 24, 40, 0.6);
    --border-color: rgba(255, 255, 255, 0.1);
}

.stApp {
    background-color: var(--bg-color);
    color: var(--text-color);
    font-family: 'Inter', sans-serif;
}

/* Remove Streamlit Default Footer spacing */
footer { visibility: hidden; display: none !important; }
.block-container { padding-bottom: 2rem !important; }

h1, h2, h3, h4, .font-display {
    font-family: 'Space Grotesk', sans-serif !important;
}

/* Gradients & Animations */
.gradient-text {
    background: linear-gradient(135deg, var(--primary), var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}

.glass-card {
    background: var(--card-bg);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border-color);
    border-radius: 1rem;
    padding: 2rem;
    box-shadow: 0 0 30px -5px rgba(45, 212, 191, 0.15);
    margin-bottom: 2rem;
}

/* Custom Hero Section */
.hero-container {
    text-align: center;
    padding: 4rem 1rem 2rem 1rem;
    position: relative;
    overflow: hidden;
}

.hero-title {
    font-size: 4.5rem;
    line-height: 1.1;
    font-weight: 800;
    margin-bottom: 1.5rem;
    letter-spacing: -0.02em;
}

/* Input Fields styling override */
div[data-baseweb="input"] > div {
    background-color: rgba(0,0,0,0.2) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 0.5rem;
}
div[data-baseweb="input"] > div:focus-within {
    border-color: var(--primary) !important;
}
.stNumberInput label { font-family: 'Space Grotesk'; font-weight: 600; color: #e2e8f0 !important; }

/* Primary Button Styling */
.stButton > button {
    width: 100%;
    border-radius: 0.75rem;
    height: 3.5rem;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.1rem;
    border: none;
    transition: all 0.3s ease;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--primary), var(--accent));
    color: #000;
    box-shadow: 0 0 20px rgba(45, 212, 191, 0.3);
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 30px rgba(45, 212, 191, 0.5);
}

/* Reset / Secondary Button Styling */
.stButton > button[kind="secondary"] {
    background: rgba(255, 255, 255, 0.05);
    color: #f1f5f9;
    border: 1px solid var(--border-color);
}
.stButton > button[kind="secondary"]:hover {
    background: rgba(239, 68, 68, 0.1);
    border-color: var(--destructive);
    color: var(--destructive);
    transform: translateY(-2px);
}

/* Results Banners */
.result-banner {
    display: flex;
    align-items: center;
    gap: 2rem;
    padding: 2rem;
    border-radius: 1rem;
    margin-top: 2rem;
}
.result-banner.positive { background: rgba(239, 68, 68, 0.05); border: 1px solid rgba(239, 68, 68, 0.4); }
.result-banner.negative { background: rgba(16, 185, 129, 0.05); border: 1px solid rgba(16, 185, 129, 0.4); }
.result-prob { font-size: 3.5rem; font-weight: 800; font-family: 'Space Grotesk'; }

/* Floating Particles Animation */
@keyframes float {
    0%, 100% { transform: translateY(0) translateX(0); opacity: 0; }
    50% { transform: translateY(-100px) translateX(50px); opacity: 0.5; }
}
.particle { position: absolute; border-radius: 50%; background: rgba(45, 212, 191, 0.3); animation: float 8s ease-in-out infinite; }
</style>

<div class="particle" style="width: 10px; height: 10px; left: 20%; top: 20%; animation-delay: 0s;"></div>
<div class="particle" style="width: 15px; height: 15px; left: 80%; top: 30%; animation-delay: 2s;"></div>
<div class="particle" style="width: 8px; height: 8px; left: 50%; top: 70%; animation-delay: 4s;"></div>
""", unsafe_allow_html=True)

# --- 5. HERO SECTION ---
st.markdown("""
<div class="hero-container">
    <div class="hero-title font-display">
        Lung Cancer <span class="gradient-text">Predictor</span>
    </div>
    <p style="color: #94a3b8; font-size: 1.25rem; max-width: 600px; margin: 0 auto;">
        Advanced gene expression analysis using trained ML models to predict lung cancer risk with clinical-grade accuracy.
    </p>
</div>
""", unsafe_allow_html=True)

# --- 6. INPUT SECTION ---
with st.container():
    st.markdown("""
    <div class='glass-card' style='padding-top: 1rem; padding-bottom: 1rem; margin-bottom: 2rem; margin-top: 4rem;'>
        <h2 class='font-display' style='text-align: center; margin: 0;'>Gene Expression <span class='gradient-text'>Input Panel</span></h2>
    </div>
    """, unsafe_allow_html=True)
    
    filled_count = 0
    total_genes = len(BIOMARKERS)
    cols = st.columns(5)
    
    # THE FIX: Removed dictionary assignments, relying purely on Streamlit's 'key' architecture
    for i, gene in enumerate(BIOMARKERS):
        with cols[i % 5]:
            val = st.number_input(
                label=gene,
                value=None, # This ensures it starts perfectly empty
                format="%.4f",
                placeholder="e.g. 0.0000",
                min_value=0.0,
                key=gene 
            )
            if val is not None:
                filled_count += 1
            
    st.write("<br>", unsafe_allow_html=True)
    
    progress_percentage = (filled_count / total_genes) * 100
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; padding: 0 1rem;">
        <span style="color: #94a3b8; font-size: 0.85rem;">Input Completion</span>
        <span style="color: #2dd4bf; font-weight: bold; font-family: 'Space Grotesk';">{filled_count} / {total_genes} Genes</span>
    </div>
    <div style="width: calc(100% - 2rem); margin: 0 auto 2rem auto; background-color: rgba(255,255,255,0.05); border-radius: 999px; height: 8px; overflow: hidden; border: 1px solid rgba(255,255,255,0.1);">
        <div style="width: {progress_percentage}%; background: linear-gradient(135deg, var(--primary), var(--accent)); height: 100%; transition: width 0.4s ease; border-radius: 999px;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Centered Buttons
    _, btn_run_col, btn_reset_col, _ = st.columns([1, 1.5, 1.5, 1])
    
    with btn_run_col:
        if st.button("RUN DIAGNOSTICS", type="primary", use_container_width=True):
            if filled_count < total_genes:
                st.warning("⚠️ Action Blocked: Please enter expression values for all 10 genes before running the diagnostic.")
            else:
                if model and scaler:
                    with st.spinner("Analyzing biomarker patterns against neural weights..."):
                        time.sleep(1.5) 
                        
                        # Extract the values directly from Streamlit's session state
                        current_inputs = {gene: st.session_state[gene] for gene in BIOMARKERS}
                        input_df = pd.DataFrame([current_inputs])[BIOMARKERS]
                        
                        input_scaled = scaler.transform(input_df)
                        prediction = model.predict(input_scaled)[0]
                        probabilities = model.predict_proba(input_scaled)[0]
                        
                        st.session_state.results = {
                            "prediction": "positive" if prediction == 1 else "negative",
                            "probability": probabilities[1] * 100,
                            "raw_probs": probabilities,
                            "inputs": current_inputs
                        }
                else:
                    st.error("Error: The machine learning model or scaler was not found in the directory. Ensure 'best_model.pkl' and 'scaler.pkl' are present.")

    with btn_reset_col:
        st.button("RESET DATA", type="secondary", use_container_width=True, on_click=clear_form)

# --- 7. RESULTS PANEL ---
if st.session_state.results:
    res = st.session_state.results
    is_positive = res['prediction'] == 'positive'
    
    st.markdown("<br><h2 class='font-display' style='text-align: center;'>Diagnostic <span class='gradient-text'>Results</span></h2>", unsafe_allow_html=True)
    
    if is_positive:
        st.markdown(f"""
        <div class="glass-card result-banner positive">
            <div style="flex-grow: 1;">
                <p style="color: #94a3b8; text-transform: uppercase; letter-spacing: 2px; font-size: 0.8rem; margin: 0;">Prediction</p>
                <h3 style="color: var(--destructive); margin: 0.5rem 0; font-size: 2rem;">⚠️ Tumor Detected</h3>
                <p style="color: #cbd5e1; margin: 0;">The ML model indicates elevated biomarker patterns consistent with lung cancer. Please consult an oncologist.</p>
            </div>
            <div style="text-align: right;">
                <p style="color: #94a3b8; margin: 0;">Malignancy Probability</p>
                <div class="result-prob" style="color: var(--destructive);">{res['probability']:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="glass-card result-banner negative">
            <div style="flex-grow: 1;">
                <p style="color: #94a3b8; text-transform: uppercase; letter-spacing: 2px; font-size: 0.8rem; margin: 0;">Prediction</p>
                <h3 style="color: var(--success); margin: 0.5rem 0; font-size: 2rem;">✅ Normal Profile</h3>
                <p style="color: #cbd5e1; margin: 0;">Gene expression patterns are within normal ranges. Continue regular screenings as recommended.</p>
            </div>
            <div style="text-align: right;">
                <p style="color: #94a3b8; margin: 0;">Tumor Probability</p>
                <div class="result-prob" style="color: var(--success);">{res['probability']:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("""
        <div class='glass-card' style='padding: 1rem; margin-bottom: 1rem;'>
            <h4 style='margin: 0; text-align: center;'>Probability Distribution</h4>
        </div>
        """, unsafe_allow_html=True)
        
        pie_data = pd.DataFrame({
            'Category': ['Tumor Probability', 'Normal Probability'],
            'Value': [res['raw_probs'][1] * 100, res['raw_probs'][0] * 100]
        })
        
        colors = ['#ef4444', '#1e293b'] if is_positive else ['#1e293b', '#10b981']
        
        fig_pie = px.pie(pie_data, values='Value', names='Category', hole=0.7)
        fig_pie.update_traces(
            marker=dict(colors=colors), 
            textposition='inside', 
            textinfo='percent', 
            hoverinfo='label+percent',
            textfont=dict(size=14, color='white', family='Inter')
        )
        fig_pie.update_layout(
            showlegend=True, 
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5, font=dict(color="#f8fafc")),
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=20, b=20, l=20, r=20),
            height=300
        )
        st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})

    with chart_col2:
        st.markdown("""
        <div class='glass-card' style='padding: 1rem; margin-bottom: 1rem;'>
            <h4 style='margin: 0; text-align: center;'>Patient Expression Profile</h4>
        </div>
        """, unsafe_allow_html=True)
        
        bar_data = pd.DataFrame(list(res['inputs'].items()), columns=['Gene', 'Log2 Expression'])
        bar_data = bar_data.sort_values('Log2 Expression', ascending=True) 
        
        fig_bar = px.bar(bar_data, x='Log2 Expression', y='Gene', orientation='h')
        fig_bar.update_traces(marker_color='#2dd4bf') 
        fig_bar.update_layout(
            xaxis=dict(showgrid=True, gridcolor='#1e293b', zeroline=False, tickfont=dict(color="#94a3b8")),
            yaxis=dict(showgrid=False, tickfont=dict(color="#f8fafc", family="Space Grotesk")),
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=20, b=20, l=20, r=20),
            height=300
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

# --- 8. FOOTER ---
st.markdown("""
<div style='text-align: center; margin-top: 3rem; padding-top: 2rem; border-top: 1px solid rgba(255,255,255,0.05); color: #64748b; font-size: 0.85rem;'>
    <p style='margin: 0; padding-bottom: 0.5rem;'>© 2026 LungBioML — Lung Cancer Predictor using Machine Learning</p>
    <p style='margin: 0; font-size: 0.75rem; opacity: 0.7;'>For research and educational purposes only.</p>
</div>
""", unsafe_allow_html=True)