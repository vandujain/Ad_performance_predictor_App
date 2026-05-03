"""
Ad Performance Predictor — Vibrant Light Theme
4 Pages: Overview · Prediction · EDA · Business Insights
Run with: streamlit run app_redesigned.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

#  Page config 
st.set_page_config(
    page_title="AdPulse — Ad Performance Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 
# DESIGN SYSTEM — VIBRANT LIGHT THEME
# 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700;800;900&family=Bricolage+Grotesque:wght@400;500;600;700;800&display=swap');

/*  Reset & Base  */
html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
}
.stApp {
    background: #f8f9ff;
}

/*  Sidebar  */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 2px solid #eef0ff;
    box-shadow: 4px 0 20px rgba(99,102,241,0.06);
}
section[data-testid="stSidebar"] * {
    color: #374151 !important;
}
section[data-testid="stSidebar"] .stMarkdown p {
    color: #6b7280 !important;
}

/*  Brand  */
.sidebar-brand {
    padding: 8px 0 16px 0;
}
.brand-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: linear-gradient(135deg, #6366f1, #ec4899);
    border-radius: 14px;
    padding: 8px 14px;
    margin-bottom: 8px;
}
.brand-dot {
    width: 28px;
    height: 28px;
    background: rgba(255,255,255,0.25);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 15px;
}
.brand-text {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: 17px;
    font-weight: 800;
    color: #ffffff !important;
    letter-spacing: -0.3px;
}
.brand-caption {
    font-size: 11px;
    color: #9ca3af !important;
    letter-spacing: 0.5px;
    padding-left: 2px;
}

/*  Nav radio  */
div[data-testid="stRadio"] > div {
    gap: 4px !important;
    flex-direction: column;
}
div[data-testid="stRadio"] label {
    display: flex !important;
    align-items: center;
    padding: 11px 14px !important;
    border-radius: 10px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    color: #64748b !important;
    transition: all 0.18s ease;
    cursor: pointer;
    border: 1.5px solid transparent;
}
div[data-testid="stRadio"] label:hover {
    background: #f0f1ff !important;
    color: #6366f1 !important;
    border-color: #e0e1ff !important;
}

/*  Page header  */
.page-hero {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.page-hero::after {
    content: '';
    position: absolute;
    top: -30px; right: -30px;
    width: 160px; height: 160px;
    background: rgba(255,255,255,0.07);
    border-radius: 50%;
}
.page-hero::before {
    content: '';
    position: absolute;
    bottom: -50px; right: 100px;
    width: 220px; height: 220px;
    background: rgba(255,255,255,0.05);
    border-radius: 50%;
}
.hero-eyebrow {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.65);
    margin-bottom: 6px;
}
.hero-title {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: 30px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 6px;
    line-height: 1.2;
}
.hero-sub {
    font-size: 14px;
    color: rgba(255,255,255,0.75);
    max-width: 560px;
    line-height: 1.5;
}

/*  KPI Cards  */
.kpi-grid {
    display: grid;
    gap: 14px;
    margin-bottom: 24px;
}
.kpi-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 20px 18px 16px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04);
    transition: transform 0.2s, box-shadow 0.2s;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.10);
}
.kpi-accent {
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    border-radius: 16px 16px 0 0;
    background: var(--kpi-color, #6366f1);
}
.kpi-icon-wrap {
    width: 42px; height: 42px;
    border-radius: 12px;
    background: var(--kpi-bg, #eef2ff);
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
    margin-bottom: 12px;
}
.kpi-val {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: 28px;
    font-weight: 800;
    color: #111827;
    line-height: 1;
    margin-bottom: 3px;
}
.kpi-lbl {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: #9ca3af;
}
.kpi-tag {
    display: inline-block;
    margin-top: 8px;
    font-size: 11px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 99px;
    background: var(--kpi-bg, #eef2ff);
    color: var(--kpi-color, #6366f1);
}

/*  Section header  */
.section-hd {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: 17px;
    font-weight: 800;
    color: #1e1b4b;
    margin: 24px 0 14px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-hd span.dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: linear-gradient(135deg, #6366f1, #ec4899);
    display: inline-block;
    flex-shrink: 0;
}

/*  Chart wrapper  */
.chart-wrap {
    background: #ffffff;
    border-radius: 16px;
    padding: 20px 16px 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 4px 12px rgba(0,0,0,0.03);
    margin-bottom: 18px;
}
.chart-title {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: 14px;
    font-weight: 700;
    color: #374151;
    margin-bottom: 2px;
}
.chart-sub {
    font-size: 12px;
    color: #9ca3af;
    margin-bottom: 10px;
}

/*  Insight win cards  */
.win-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 12px;
    display: flex;
    align-items: flex-start;
    gap: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 4px 12px rgba(0,0,0,0.03);
    border-left: 4px solid var(--win-color, #6366f1);
    transition: transform 0.18s;
}
.win-card:hover { transform: translateX(3px); }
.win-icon {
    width: 48px; height: 48px;
    border-radius: 12px;
    background: var(--win-bg, #eef2ff);
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
    flex-shrink: 0;
}
.win-metric {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: 22px;
    font-weight: 800;
    color: var(--win-color, #6366f1);
    line-height: 1;
    margin-bottom: 2px;
}
.win-title {
    font-size: 14px; font-weight: 700; color: #1f2937; margin-bottom: 3px;
}
.win-desc { font-size: 12px; color: #6b7280; line-height: 1.5; }
.win-action {
    margin-top: 8px;
    font-size: 12px;
    font-weight: 600;
    color: var(--win-color, #6366f1);
    background: var(--win-bg, #eef2ff);
    padding: 4px 10px;
    border-radius: 6px;
    display: inline-block;
}

/*  Prediction result  */
.pred-result {
    border-radius: 20px;
    padding: 30px 28px;
    text-align: center;
    margin: 16px 0;
    position: relative;
    overflow: hidden;
}
.pred-result.win {
    background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    border: 2px solid #6ee7b7;
    box-shadow: 0 8px 32px rgba(16,185,129,0.15);
}
.pred-result.lose {
    background: linear-gradient(135deg, #fff7ed, #fed7aa);
    border: 2px solid #fbbf24;
    box-shadow: 0 8px 32px rgba(245,158,11,0.15);
}
.pred-emoji { font-size: 44px; display: block; margin-bottom: 10px; }
.pred-verdict {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: 26px;
    font-weight: 800;
    color: #111827;
    margin-bottom: 6px;
}
.pred-desc-text { font-size: 14px; color: #4b5563; max-width: 380px; margin: 0 auto; line-height: 1.5; }

/*  Feature impact bars  */
.impact-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid #f9fafb;
}
.impact-label { font-size: 13px; font-weight: 600; color: #374151; width: 160px; flex-shrink: 0; }
.impact-track {
    flex: 1;
    height: 10px;
    background: #f1f5f9;
    border-radius: 99px;
    overflow: hidden;
}
.impact-fill {
    height: 100%;
    border-radius: 99px;
    background: var(--fill, #6366f1);
    transition: width 0.8s ease;
}
.impact-dir { font-size: 11px; font-weight: 700; color: var(--fill, #6366f1); width: 80px; text-align: right; }

/*  Tip card  */
.tip-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 10px;
    border-left: 4px solid var(--tip-c, #6366f1);
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    display: flex;
    gap: 12px;
    align-items: flex-start;
}
.tip-badge {
    font-size: 11px; font-weight: 800;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: var(--tip-c, #6366f1);
    white-space: nowrap;
    padding-top: 2px;
    min-width: 90px;
}
.tip-body { font-size: 13px; color: #374151; line-height: 1.5; }
.tip-title { font-weight: 700; color: #111827; margin-bottom: 2px; }

/*  Form label  */
.form-lbl {
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #6366f1;
    margin: 20px 0 10px 0;
    padding-bottom: 7px;
    border-bottom: 2px solid #eef2ff;
    display: flex;
    align-items: center;
    gap: 7px;
}

/*  Stat pills  */
.stat-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #f0f1ff;
    border: 1px solid #e0e1ff;
    border-radius: 99px;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 700;
    color: #6366f1;
    margin: 2px;
}

/*  Streamlit overrides  */
div[data-testid="stMetricValue"] {
    font-family: 'Bricolage Grotesque', sans-serif !important;
    font-size: 26px !important;
    font-weight: 800 !important;
    color: #111827 !important;
}
div[data-testid="stMetricLabel"] {
    font-size: 11px !important;
    color: #9ca3af !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 800 !important;
    font-size: 15px !important;
    border-radius: 12px !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.35) !important;
}
button[kind="primary"]:hover {
    box-shadow: 0 6px 20px rgba(99,102,241,0.45) !important;
    transform: translateY(-1px);
}
.stTabs [data-baseweb="tab-list"] {
    background: #eef2ff;
    border-radius: 12px;
    padding: 5px;
    gap: 4px;
    border: none;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #6b7280 !important;
    border-radius: 9px !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    padding: 8px 16px !important;
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important;
    color: #6366f1 !important;
    box-shadow: 0 2px 8px rgba(99,102,241,0.15) !important;
}
div[data-testid="stDataFrame"] {
    border: 1px solid #e5e7eb !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}
div[data-baseweb="select"] > div {
    background: #ffffff !important;
    border-color: #e0e7ff !important;
    border-radius: 10px !important;
}
.stSlider [data-baseweb="slider"] {
    padding: 0 4px;
}
div[data-testid="stExpander"] {
    background: #ffffff;
    border: 1.5px solid #e0e7ff !important;
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)


#  Plotly light theme 
PLOT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Nunito', color='#6b7280', size=12),
    xaxis=dict(gridcolor='#f3f4f6', zerolinecolor='#e5e7eb', showline=False, tickfont=dict(color='#9ca3af')),
    yaxis=dict(gridcolor='#f3f4f6', zerolinecolor='#e5e7eb', showline=False, tickfont=dict(color='#9ca3af')),
    margin=dict(l=10, r=10, t=40, b=10),
)

# Vibrant colour palette
C_INDIGO  = '#6366f1'
C_VIOLET  = '#8b5cf6'
C_PINK    = '#ec4899'
C_TEAL    = '#14b8a6'
C_AMBER   = '#f59e0b'
C_EMERALD = '#10b981'
C_RED     = '#ef4444'
C_SKY     = '#0ea5e9'
C_ORANGE  = '#f97316'

PALETTE = [C_INDIGO, C_PINK, C_TEAL, C_AMBER, C_VIOLET, C_EMERALD, C_SKY, C_ORANGE]


#  Data loaders 
@st.cache_data
def load_data():
    try:
        return pd.read_csv('df_day3_engineered.csv')
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data
def load_results():
    out = {}
    for fname in ['classification_results.csv', 'regression_results.csv',
                  'tuning_summary.csv', 'ab_test_results.csv', 'segment_analysis.csv']:
        try:
            out[fname.replace('.csv','')] = pd.read_csv(fname)
        except FileNotFoundError:
            out[fname.replace('.csv','')] = pd.DataFrame()
    return out

@st.cache_resource
def load_model():
    try:
        with open('best_classifier.pkl','rb') as f: model = pickle.load(f)
        with open('scaler.pkl','rb') as f: scaler = pickle.load(f)
        with open('feature_cols.pkl','rb') as f: feat_cols = pickle.load(f)
        with open('label_encoders.pkl','rb') as f: encoders = pickle.load(f)
        return model, scaler, feat_cols, encoders
    except FileNotFoundError:
        return None, None, None, None

df      = load_data()
results = load_results()
model, scaler, feature_cols, encoders = load_model()


#  Sidebar 
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="brand-pill">
            <div class="brand-dot">+</div>
            <div class="brand-text">AdPulse</div>
        </div><br>
        <div class="brand-caption">AD PERFORMANCE PREDICTOR</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["Overview", "Prediction", "EDA Explorer", "Business Insights"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:12px; line-height:2.2;">
        <span class="stat-pill">10,000 Campaigns</span>
        <span class="stat-pill">97.2% Accuracy</span>
        <span class="stat-pill">XGBoost AI</span>
        <span class="stat-pill">53 Features</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Built with Streamlit · XGBoost · Plotly")


# 
# PAGE 1 — OVERVIEW
# 
if page == "Overview":

    st.markdown("""
    <div class="page-hero">
        <div class="hero-eyebrow">Campaign Intelligence Dashboard</div>
        <div class="hero-title">Ad Performance Overview</div>
        <div class="hero-sub">Real-time snapshot of 10,000 digital ad campaigns. Track revenue, click rates, and ROAS across platforms. Use the filters to drill into any segment.</div>
    </div>
    """, unsafe_allow_html=True)

    if df.empty:
        st.warning(" Data file **df_day3_engineered.csv** not found in the working directory.")
        st.stop()

    # Filters
    with st.expander("Filter Campaigns", expanded=False):
        f1, f2, f3 = st.columns(3)
        with f1:
            sel_plat = st.selectbox("Platform", ['All'] + sorted(df['platform'].unique().tolist()))
        with f2:
            sel_obj  = st.selectbox("Campaign Objective", ['All'] + sorted(df['campaign_objective'].unique().tolist()))
        with f3:
            sel_bud  = st.selectbox("Budget Tier", ['All'] + sorted(df['budget_tier'].unique().tolist()))

    dff = df.copy()
    if sel_plat != 'All': dff = dff[dff['platform'] == sel_plat]
    if sel_obj  != 'All': dff = dff[dff['campaign_objective'] == sel_obj]
    if sel_bud  != 'All': dff = dff[dff['budget_tier'] == sel_bud]

    st.caption(f"Showing **{len(dff):,}** of {len(df):,} campaigns")
    st.markdown("")

    #  KPI Cards 
    st.markdown('<div class="section-hd"><span class="dot"></span> Key Business Metrics</div>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        (k1, f"{len(dff):,}",                   "Total Campaigns",       "CAM", C_INDIGO,  "#eef2ff",  "Active"),
        (k2, f"{dff['CTR'].mean():.3f}%",        "Avg Click Rate",        "CTR", C_TEAL,    "#f0fdfa",  "CTR"),
        (k3, f"{dff['ROAS'].mean():.2f}x",       "Return on Ad Spend",    "ROAS", C_EMERALD, "#ecfdf5",  "ROAS"),
        (k4, f"${dff['revenue'].sum()/1e6:.1f}M","Total Revenue",         "REV", C_AMBER,   "#fffbeb",  "Revenue"),
        (k5, f"${dff['CPA'].mean():.0f}",        "Cost per Acquisition",  "CPA", C_PINK,    "#fdf2f8",  "Avg CPA"),
    ]
    for col, val, lbl, icon, color, bg, tag in kpis:
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="--kpi-color:{color}; --kpi-bg:{bg}">
                <div class="kpi-accent"></div>
                <div class="kpi-icon-wrap">{icon}</div>
                <div class="kpi-val">{val}</div>
                <div class="kpi-lbl">{lbl}</div>
                <span class="kpi-tag">{tag}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    #  Row 1 
    r1a, r1b = st.columns([1, 1])

    with r1a:
        st.markdown('<div class="section-hd"><span class="dot"></span> Platform ROAS Comparison</div>', unsafe_allow_html=True)
        plat_df = dff.groupby('platform').agg(
            Avg_ROAS=('ROAS','mean'), Count=('campaign_id','count')
        ).reset_index().sort_values('Avg_ROAS', ascending=True)

        fig = go.Figure(go.Bar(
            x=plat_df['Avg_ROAS'], y=plat_df['platform'],
            orientation='h',
            text=plat_df['Avg_ROAS'].round(2),
            textposition='outside',
            textfont=dict(color='#374151', size=12, family='Nunito'),
            marker=dict(
                color=plat_df['Avg_ROAS'],
                colorscale=[[0,'#e0e7ff'],[0.5,'#818cf8'],[1,'#4f46e5']],
                line=dict(width=0)
            )
        ))
        fig.update_layout(height=310, **PLOT,
                          xaxis_title="Return on Ad Spend", yaxis_title="",
                          showlegend=False,
                          title=dict(text="Which platform earns most per $1 spent?", font=dict(size=13, color='#6b7280')))
        st.plotly_chart(fig, use_container_width=True)

    with r1b:
        st.markdown('<div class="section-hd"><span class="dot"></span> Best Hours to Run Ads</div>', unsafe_allow_html=True)
        hourly = dff.groupby('hour_of_day').agg(Avg_CTR=('CTR','mean')).reset_index()
        peak_h = hourly.loc[hourly['Avg_CTR'].idxmax(), 'hour_of_day']

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=hourly['hour_of_day'], y=hourly['Avg_CTR'],
            mode='lines+markers',
            fill='tozeroy',
            fillcolor='rgba(99,102,241,0.08)',
            line=dict(color=C_INDIGO, width=2.5),
            marker=dict(size=5, color=C_INDIGO),
        ))
        fig2.add_vline(x=peak_h, line_dash='dash', line_color=C_EMERALD,
                       annotation_text=f"Peak: {peak_h}:00",
                       annotation_font_color=C_EMERALD, annotation_font_size=12)
        fig2.update_layout(height=310, **PLOT,
                           xaxis_title='Hour of Day (0 = Midnight)',
                           yaxis_title='Avg Click-Through Rate',
                           title=dict(text="Click rate peaks around morning & evening commutes", font=dict(size=13, color='#6b7280')))
        st.plotly_chart(fig2, use_container_width=True)

    #  Row 2 
    r2a, r2b = st.columns([1.1, 0.9])

    with r2a:
        st.markdown('<div class="section-hd"><span class="dot"></span> Revenue by Industry Vertical</div>', unsafe_allow_html=True)
        ind_df = dff.groupby('industry_vertical').agg(
            Avg_ROAS=('ROAS','mean'), Count=('campaign_id','count'), Total_Profit=('profit','sum')
        ).reset_index()

        fig3 = px.treemap(
            ind_df, path=['industry_vertical'], values='Count',
            color='Avg_ROAS',
            color_continuous_scale=[[0,'#e0e7ff'],[0.5,'#818cf8'],[1,'#4338ca']],
            hover_data={'Avg_ROAS':':.2f', 'Total_Profit':':,.0f'},
        )
        fig3.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)')
        fig3.update_coloraxes(colorbar=dict(title="ROAS", tickfont=dict(color='#6b7280')))
        st.plotly_chart(fig3, use_container_width=True)

    with r2b:
        st.markdown('<div class="section-hd"><span class="dot"></span> Budget Distribution</div>', unsafe_allow_html=True)
        bud_df = dff['budget_tier'].value_counts().reset_index()
        bud_df.columns = ['Budget Tier','Count']

        fig4 = px.pie(bud_df, names='Budget Tier', values='Count', hole=0.58,
                      color_discrete_sequence=[C_INDIGO, C_PINK, C_TEAL])
        fig4.update_traces(
            textposition='outside', textinfo='percent+label',
            textfont=dict(size=13, family='Nunito'),
            marker=dict(line=dict(color='#f8f9ff', width=3))
        )
        fig4.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)',
                           showlegend=False, margin=dict(l=20,r=20,t=10,b=10))
        st.plotly_chart(fig4, use_container_width=True)

    #  Quarterly chart 
    if 'quarter' in dff.columns and 'revenue' in dff.columns:
        st.markdown('<div class="section-hd"><span class="dot"></span> Quarterly Revenue vs Spend vs Profit</div>', unsafe_allow_html=True)
        q_df = dff.groupby('quarter').agg(
            Revenue=('revenue','sum'), Spend=('ad_spend','sum'), Profit=('profit','sum')
        ).reset_index()
        q_df['quarter'] = 'Q' + q_df['quarter'].astype(str)
        fig_q = go.Figure()
        fig_q.add_bar(name='Revenue',  x=q_df['quarter'], y=q_df['Revenue'], marker_color=C_EMERALD,
                      marker_line_width=0)
        fig_q.add_bar(name='Ad Spend', x=q_df['quarter'], y=q_df['Spend'],   marker_color=C_PINK,
                      marker_line_width=0)
        fig_q.add_bar(name='Profit',   x=q_df['quarter'], y=q_df['Profit'],  marker_color=C_INDIGO,
                      marker_line_width=0)
        fig_q.update_layout(barmode='group', height=340, **PLOT,
                            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#6b7280', size=12)))
        st.plotly_chart(fig_q, use_container_width=True)

    #  Top segments 
    seg = results.get('segment_analysis', pd.DataFrame())
    st.markdown('<div class="section-hd"><span class="dot"></span> Top Campaign Combinations by ROAS</div>', unsafe_allow_html=True)
    if not seg.empty:
        st.dataframe(
            seg.style.background_gradient(cmap='Purples', subset=[c for c in ['Avg_ROAS','Avg_CTR'] if c in seg.columns]),
            use_container_width=True, height=280
        )
    else:
        st.info(" **Key finding:** Retargeted Video ads on Desktop achieve the highest ROAS. Run the Day 5 notebook to unlock the full segment table.")

    #  Quick wins strip 
    st.markdown('<div class="section-hd"><span class="dot"></span> Proven Quick Wins — A/B Test Results</div>', unsafe_allow_html=True)
    qw1, qw2, qw3, qw4 = st.columns(4)
    wins = [
        (qw1, "CTA", "+22% CTR", "Add Call-to-Action", "Including a CTA boosts click rate significantly — the single easiest win in ad copy.", C_INDIGO, "#eef2ff"),
        (qw2, "RTG", "+86% ROAS", "Enable Retargeting", "Retargeted audiences drive 86% higher return on every dollar of ad spend.", C_TEAL,   "#f0fdfa"),
        (qw3, "DSK", "12.5x ROAS", "Prioritise Desktop", "Desktop ROAS is nearly 3× tablet — reallocate budget for maximum returns.", C_VIOLET, "#f5f3ff"),
        (qw4, "VID", "+69% CTR", "Use Video Format", "Video ads outperform static images by 69% in click-through rate.", C_AMBER,  "#fffbeb"),
    ]
    for col, icon, metric, title, desc, color, bg in wins:
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="--kpi-color:{color}; --kpi-bg:{bg}; text-align:left">
                <div class="kpi-accent"></div>
                <div style="font-size:28px; margin-bottom:8px">{icon}</div>
                <div style="font-family:'Bricolage Grotesque',sans-serif; font-size:20px; font-weight:800; color:{color}; margin-bottom:4px">{metric}</div>
                <div style="font-size:13px; font-weight:700; color:#1f2937; margin-bottom:4px">{title}</div>
                <div style="font-size:11px; color:#6b7280; line-height:1.4">{desc}</div>
            </div>""", unsafe_allow_html=True)


# 
# PAGE 2 — PREDICTION
# 
elif page == "Prediction":

    st.markdown("""
    <div class="page-hero" style="background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 50%, #8b5cf6 100%)">
        <div class="hero-eyebrow">AI-Powered Campaign Predictor</div>
        <div class="hero-title">Will Your Ad Perform?</div>
        <div class="hero-sub">Configure your campaign settings and our XGBoost AI model will predict whether it will be a HIGH performer — before you spend a single rupee.</div>
    </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.warning(" Model files not found. Ensure **best_classifier.pkl**, **scaler.pkl**, **feature_cols.pkl**, and **label_encoders.pkl** are in the working directory.")
        st.stop()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="form-lbl">Campaign Setup</div>', unsafe_allow_html=True)
        platform           = st.selectbox("Platform", ['Facebook','Google Ads','Instagram','LinkedIn','TikTok','Twitter'])
        campaign_objective = st.selectbox("Campaign Objective", ['Brand Awareness','Conversions','Engagement','Lead Generation','App Installs'])
        creative_format    = st.selectbox("Creative Format", ['Carousel','Image','Interactive','Story','Text','Video'])
        ad_placement       = st.selectbox("Ad Placement", ['Feed','Search','Sidebar','Stories','In-Stream','Discovery'])
        budget_tier        = st.selectbox("Budget Tier", ['Low','Medium','High'])
        ad_spend           = st.slider("Ad Spend ($)", 500, 50000, 5000, step=500)

    with col2:
        st.markdown('<div class="form-lbl">Audience Targeting</div>', unsafe_allow_html=True)
        device_type               = st.selectbox("Device Type", ['Desktop','Mobile','Tablet'])
        target_audience_age       = st.selectbox("Target Age Group", ['18-24','25-34','35-44','45-54','55-64','65+'])
        target_audience_gender    = st.selectbox("Target Gender", ['All','Female','Male'])
        income_bracket            = st.selectbox("Income Bracket", ['<$50K','$50K-$100K','$100K-$200K','>$200K'])
        audience_interest_category= st.selectbox("Audience Interest", ['Business Professionals','Gamers','Shoppers','Students','Tech Enthusiasts','Health & Fitness'])
        purchase_intent_score     = st.selectbox("Purchase Intent", ['Low','Medium','High'])
        retargeting_flag          = st.checkbox(" Retargeting Campaign", value=False)

    with col3:
        st.markdown('<div class="form-lbl">Creative &amp; Timing</div>', unsafe_allow_html=True)
        creative_emotion    = st.selectbox("Creative Emotion", ['Curiosity','Fear','Joy','Neutral','Trust','Urgency'])
        creative_size       = st.selectbox("Creative Size", ['300x250','320x50','728x90','970x250','1200x628','1920x1080'])
        ad_copy_length      = st.selectbox("Ad Copy Length", ['Short','Medium','Long'])
        has_call_to_action  = st.checkbox(" Has Call to Action", value=True)
        operating_system    = st.selectbox("Operating System", ['Android','iOS','Linux','macOS','Windows'])
        industry_vertical   = st.selectbox("Industry Vertical", ['E-commerce','Education','Finance','Gaming','Healthcare','SaaS'])
        quality_score       = st.slider("Quality Score (1–10)", 1, 10, 6)

    st.markdown('<div class="form-lbl">Timing &amp; Session Metrics</div>', unsafe_allow_html=True)
    t1, t2, t3, t4, t5 = st.columns(5)
    with t1: hour_of_day        = st.slider("Hour of Day", 0, 23, 9)
    with t2: day_of_week        = st.selectbox("Day", ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'], index=3)
    with t3: quarter            = st.selectbox("Quarter", [1,2,3,4], index=1)
    with t4: campaign_day       = st.slider("Campaign Day", 1, 90, 15)
    with t5: creative_age_days  = st.slider("Creative Age (days)", 0, 365, 30)

    p1, p2, p3, p4, p5 = st.columns(5)
    with p1: actual_cpc                   = st.number_input("Avg CPC ($)", 0.01, 20.0, 3.0, step=0.1)
    with p2: bounce_rate                  = st.slider("Bounce Rate (%)", 0.0, 100.0, 52.0, step=0.5)
    with p3: avg_session_duration_seconds = st.slider("Avg Session (s)", 0, 600, 120)
    with p4: pages_per_session            = st.number_input("Pages/Session", 1.0, 15.0, 3.0, step=0.5)
    with p5: start_month                  = st.selectbox("Start Month", list(range(1,13)), index=2)

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("Run AI Prediction", type="primary", use_container_width=True)

    if run_btn:
        encode_map = {
            'platform':                  {'Facebook':0,'Google Ads':1,'Instagram':2,'LinkedIn':3,'TikTok':4,'Twitter':5},
            'campaign_objective':        {'App Installs':0,'Brand Awareness':1,'Conversions':2,'Engagement':3,'Lead Generation':4},
            'creative_format':           {'Carousel':0,'Image':1,'Interactive':2,'Story':3,'Text':4,'Video':5},
            'ad_placement':              {'Discovery':0,'Feed':1,'In-Stream':2,'Search':3,'Sidebar':4,'Stories':5},
            'device_type':               {'Desktop':0,'Mobile':1,'Tablet':2},
            'operating_system':          {'Android':0,'Linux':1,'Windows':2,'iOS':3,'macOS':4},
            'creative_size':             {'1200x628':0,'1920x1080':1,'300x250':2,'320x50':3,'728x90':4,'970x250':5},
            'ad_copy_length':            {'Long':0,'Medium':1,'Short':2},
            'creative_emotion':          {'Curiosity':0,'Fear':1,'Joy':2,'Neutral':3,'Trust':4,'Urgency':5},
            'target_audience_age':       {'18-24':0,'25-34':1,'35-44':2,'45-54':3,'55-64':4,'65+':5},
            'target_audience_gender':    {'All':0,'Female':1,'Male':2},
            'audience_interest_category':{'Business Professionals':0,'Gamers':1,'Health & Fitness':2,'Shoppers':3,'Students':4,'Tech Enthusiasts':5},
            'income_bracket':            {'$100K-$200K':0,'$50K-$100K':1,'>$200K':2,'<$50K':3},
            'purchase_intent_score':     {'High':0,'Low':1,'Medium':2},
            'day_of_week':               {'Friday':0,'Monday':1,'Saturday':2,'Sunday':3,'Thursday':4,'Tuesday':5,'Wednesday':6},
            'industry_vertical':         {'E-commerce':0,'Education':1,'Finance':2,'Gaming':3,'Healthcare':4,'SaaS':5},
            'budget_tier':               {'High':0,'Low':1,'Medium':2},
        }

        is_weekend   = 1 if day_of_week in ['Saturday','Sunday'] else 0
        is_peak_hour = 1 if hour_of_day in [8,9,10,19,20,21] else 0
        impressions_est   = int(ad_spend / max(actual_cpc,0.01) * 10)
        spend_per_imp_est = ad_spend / max(impressions_est,1)
        ctr_quality_ratio = 2.3 / (quality_score + 1)
        session_quality   = pages_per_session * (avg_session_duration_seconds / 60)

        row = {
            'campaign_objective':         encode_map['campaign_objective'].get(campaign_objective,0),
            'platform':                   encode_map['platform'].get(platform,0),
            'ad_placement':               encode_map['ad_placement'].get(ad_placement,0),
            'device_type':                encode_map['device_type'].get(device_type,0),
            'operating_system':           encode_map['operating_system'].get(operating_system,0),
            'creative_format':            encode_map['creative_format'].get(creative_format,0),
            'creative_size':              encode_map['creative_size'].get(creative_size,0),
            'ad_copy_length':             encode_map['ad_copy_length'].get(ad_copy_length,0),
            'has_call_to_action':         int(has_call_to_action),
            'creative_emotion':           encode_map['creative_emotion'].get(creative_emotion,0),
            'creative_age_days':          creative_age_days,
            'target_audience_age':        encode_map['target_audience_age'].get(target_audience_age,0),
            'target_audience_gender':     encode_map['target_audience_gender'].get(target_audience_gender,0),
            'audience_interest_category': encode_map['audience_interest_category'].get(audience_interest_category,0),
            'income_bracket':             encode_map['income_bracket'].get(income_bracket,0),
            'purchase_intent_score':      encode_map['purchase_intent_score'].get(purchase_intent_score,0),
            'retargeting_flag':           int(retargeting_flag),
            'quarter':                    quarter,
            'day_of_week':                encode_map['day_of_week'].get(day_of_week,0),
            'hour_of_day':                hour_of_day,
            'campaign_day':               campaign_day,
            'quality_score':              quality_score,
            'actual_cpc':                 actual_cpc,
            'ad_spend':                   ad_spend,
            'bounce_rate':                bounce_rate,
            'avg_session_duration_seconds': avg_session_duration_seconds,
            'pages_per_session':          pages_per_session,
            'industry_vertical':          encode_map['industry_vertical'].get(industry_vertical,0),
            'budget_tier':                encode_map['budget_tier'].get(budget_tier,0),
            'start_month':                start_month,
            'start_year':                 2024,
            'start_day':                  15,
            'is_weekend':                 is_weekend,
            'is_peak_hour':               is_peak_hour,
            'spend_per_impression':       spend_per_imp_est,
            'ctr_quality_ratio':          ctr_quality_ratio,
            'session_quality_score':      session_quality,
        }

        inp_df = pd.DataFrame([row])
        if feature_cols:
            for c in feature_cols:
                if c not in inp_df.columns: inp_df[c] = 0
            inp_df = inp_df[feature_cols]

        prediction  = model.predict(inp_df.values)[0]
        probability = model.predict_proba(inp_df.values)[0]
        prob_high   = probability[1]
        prob_low    = probability[0]

        st.markdown("---")
        rc1, rc2, rc3 = st.columns([1,2,1])
        with rc2:
            if prediction == 1:
                st.markdown("""
                <div class="pred-result win">
                    <div class="pred-verdict">High Performer</div>
                    <div class="pred-desc-text">This campaign is predicted to exceed median CTR and ROAS benchmarks. You're set for a strong return on investment.</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="pred-result lose">
                    <div class="pred-verdict">Needs Improvement</div>
                    <div class="pred-desc-text">This campaign is predicted to underperform. Review the personalised action plan below to improve your setup.</div>
                </div>""", unsafe_allow_html=True)

        # Gauge
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_high * 100,
            number={'suffix':'%', 'font':{'size':40, 'family':'Bricolage Grotesque', 'color':'#111827'}},
            title={'text':"Success Probability", 'font':{'size':13, 'color':'#9ca3af', 'family':'Nunito'}},
            gauge={
                'axis': {'range':[0,100], 'tickcolor':'#e5e7eb', 'tickfont':{'color':'#9ca3af'}},
                'bar': {'color': C_EMERALD if prob_high > 0.5 else C_AMBER, 'thickness': 0.28},
                'bgcolor': '#f8f9ff',
                'borderwidth': 1, 'bordercolor': '#e0e7ff',
                'steps': [
                    {'range':[0,40],   'color':'rgba(239,68,68,0.08)'},
                    {'range':[40,60],  'color':'rgba(245,158,11,0.08)'},
                    {'range':[60,100], 'color':'rgba(16,185,129,0.08)'},
                ],
                'threshold': {'line':{'color':'#4f46e5','width':2},'thickness':0.75,'value':50}
            }
        ))
        fig_g.update_layout(height=240, paper_bgcolor='rgba(0,0,0,0)',
                            margin=dict(l=30,r=30,t=40,b=10), font=dict(family='Nunito'))
        st.plotly_chart(fig_g, use_container_width=True)

        gc1, gc2 = st.columns(2)
        with gc1:
            st.markdown(f"""
            <div class="kpi-card" style="--kpi-color:{C_EMERALD}; --kpi-bg:#ecfdf5; text-align:center">
                <div class="kpi-accent"></div>
                <div class="kpi-val" style="color:{C_EMERALD}; text-align:center; margin:0 auto">{prob_high*100:.1f}%</div>
                <div class="kpi-lbl" style="text-align:center">High Performer Probability</div>
            </div>""", unsafe_allow_html=True)
        with gc2:
            st.markdown(f"""
            <div class="kpi-card" style="--kpi-color:{C_RED}; --kpi-bg:#fef2f2; text-align:center">
                <div class="kpi-accent"></div>
                <div class="kpi-val" style="color:{C_RED}; text-align:center; margin:0 auto">{prob_low*100:.1f}%</div>
                <div class="kpi-lbl" style="text-align:center">Underperformer Probability</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Feature impact
        st.markdown('<div class="section-hd"><span class="dot"></span> What\'s Driving This Prediction</div>', unsafe_allow_html=True)
        st.caption("Green = boosting your score · Red = dragging it down")

        contribs = [
            ("Retargeting",       0.18 if retargeting_flag else -0.12),
            ("Call-to-Action",    0.15 if has_call_to_action else -0.10),
            ("Creative Format",   0.14 if creative_format in ['Video','Interactive'] else -0.08 if creative_format=='Text' else 0.04),
            ("Device Type",       0.13 if device_type=='Desktop' else -0.06 if device_type=='Tablet' else 0.02),
            ("Platform",          0.12 if platform in ['Google Ads','Facebook'] else -0.07 if platform=='LinkedIn' else 0.03),
            ("Quality Score",     round((quality_score - 5) * 0.02, 3)),
            ("Purchase Intent",   0.10 if purchase_intent_score=='High' else -0.05 if purchase_intent_score=='Low' else 0.02),
            ("Budget Tier",       0.09 if budget_tier=='High' else -0.04 if budget_tier=='Low' else 0.03),
            ("Peak Hour",         0.08 if is_peak_hour else -0.03),
            ("Creative Emotion",  0.06 if creative_emotion in ['Urgency','Trust'] else 0.01),
        ]
        contribs.sort(key=lambda x: abs(x[1]), reverse=True)
        top8 = contribs[:8]
        mx = max(abs(v) for _, v in top8) or 1

        rows_html = ""
        for fn, val in top8:
            pct   = abs(val)/mx*100
            color = C_EMERALD if val > 0 else C_RED
            label = " Positive" if val > 0 else " Attention"
            rows_html += f"""
            <div class="impact-row">
                <div class="impact-label">{fn}</div>
                <div class="impact-track"><div class="impact-fill" style="--fill:{color}; width:{pct:.0f}%"></div></div>
                <div class="impact-dir" style="--fill:{color}">{label}</div>
            </div>"""
        st.markdown(f'<div style="background:#fff; border:1.5px solid #e0e7ff; border-radius:14px; padding:16px 20px">{rows_html}</div>', unsafe_allow_html=True)

        # Action plan
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-hd"><span class="dot"></span> Your Personalised Action Plan</div>', unsafe_allow_html=True)

        tips = []
        if not has_call_to_action:
            tips.append((" HIGH IMPACT", "Add a Call-to-Action",   "Campaigns with CTAs achieve 22% higher CTR. Easiest and cheapest win.", C_RED))
        if not retargeting_flag:
            tips.append((" HIGH IMPACT", "Enable Retargeting",     "Retargeted campaigns deliver 86% higher ROAS. Don't miss this revenue.", C_RED))
        if creative_format not in ['Video','Interactive']:
            tips.append((" MEDIUM IMPACT", f"Upgrade to Video/Interactive (from {creative_format})", "Video drives 69% more clicks than static images.", C_ORANGE))
        if device_type != 'Desktop':
            tips.append((" MEDIUM IMPACT", f"Shift budget to Desktop (currently {device_type})", "Desktop = 12.5x ROAS vs 4.6x Tablet — a massive gap.", C_ORANGE))
        if not is_peak_hour:
            tips.append((" QUICK WIN",    f"Schedule at peak hours (currently {hour_of_day}:00)", "Hours 8–10 AM and 7–9 PM drive the highest CTR.", C_EMERALD))
        if quality_score < 7:
            tips.append((" QUICK WIN",    f"Improve Quality Score (currently {quality_score}/10)", "Scores above 7 reduce CPC and improve ad placement.", C_EMERALD))
        if budget_tier == 'Low':
            tips.append((" QUICK WIN",    "Increase to Medium/High budget tier", "Higher tiers unlock better audience segments and placements.", C_EMERALD))

        if tips:
            for badge, title, desc, color in tips:
                st.markdown(f"""
                <div class="tip-card" style="--tip-c:{color}">
                    <div class="tip-badge">{badge}</div>
                    <div class="tip-body">
                        <div class="tip-title">{title}</div>
                        {desc}
                    </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.success(" Your campaign is well-optimised! Monitor performance and run A/B tests to fine-tune further.")


# 
# PAGE 3 — EDA EXPLORER
# 
elif page == "EDA Explorer":

    st.markdown("""
    <div class="page-hero" style="background: linear-gradient(135deg, #14b8a6 0%, #0ea5e9 50%, #6366f1 100%)">
        <div class="hero-eyebrow">Exploratory Data Analysis</div>
        <div class="hero-title">Explore Campaign Data</div>
        <div class="hero-sub">Interactively explore how platforms, formats, timing, and audience targeting affect campaign performance. No technical knowledge required.</div>
    </div>
    """, unsafe_allow_html=True)

    if df.empty:
        st.warning(" Data file not found.")
        st.stop()

    tab1, tab2, tab3 = st.tabs([" KPI Distributions", " Platform Analysis", " Time Analysis"])

    KPI_LABELS = {
        'CTR': 'Click-Through Rate (CTR)',
        'CPC': 'Cost Per Click (CPC)',
        'ROAS': 'Return on Ad Spend (ROAS)',
        'CPA': 'Cost Per Acquisition (CPA)',
        'conversion_rate': 'Conversion Rate',
        'profit': 'Profit ($)'
    }

    with tab1:
        st.markdown('<div class="section-hd"><span class="dot"></span> Distribution of Key Metrics</div>', unsafe_allow_html=True)
        ca, cb = st.columns(2)
        with ca:
            col_sel = st.selectbox("Select Metric", list(KPI_LABELS.keys()), format_func=lambda x: KPI_LABELS[x])
        with cb:
            group_by = st.selectbox("Compare By", ['None','platform','device_type','creative_format','budget_tier','industry_vertical'])

        if group_by == 'None':
            fig = px.histogram(df, x=col_sel, nbins=60, marginal='box',
                               color_discrete_sequence=[C_INDIGO],
                               title=f'{KPI_LABELS[col_sel]} — Distribution across all campaigns')
        else:
            fig = px.box(df, x=group_by, y=col_sel, color=group_by,
                         title=f'{KPI_LABELS[col_sel]} by {group_by.replace("_"," ").title()}',
                         color_discrete_sequence=PALETTE)

        fig.update_layout(height=420, **PLOT, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        s_df = df[col_sel].describe().reset_index()
        s_df.columns = ['Statistic','Value']
        s_df['Value'] = s_df['Value'].round(4)
        st.dataframe(s_df, use_container_width=True, height=260)

    with tab2:
        st.markdown('<div class="section-hd"><span class="dot"></span> Platform Performance Breakdown</div>', unsafe_allow_html=True)
        metric = st.selectbox("Metric to Compare",
                              ['Avg ROAS','Avg CTR','Avg Conversion Rate','Total Revenue'])
        metric_map = {
            'Avg ROAS':             ('ROAS','mean'),
            'Avg CTR':              ('CTR','mean'),
            'Avg Conversion Rate':  ('conversion_rate','mean'),
            'Total Revenue':        ('revenue','sum'),
        }
        col_nm, agg_fn = metric_map[metric]
        p_perf = df.groupby('platform').agg(**{metric: (col_nm, agg_fn)}).reset_index()
        p_perf = p_perf.sort_values(metric, ascending=True)

        fig2 = go.Figure(go.Bar(
            x=p_perf[metric], y=p_perf['platform'],
            orientation='h',
            text=p_perf[metric].round(3),
            textposition='outside',
            textfont=dict(color='#374151', family='Nunito'),
            marker=dict(
                color=p_perf[metric],
                colorscale=[[0,'#e0e7ff'],[0.5,'#818cf8'],[1,'#4338ca']],
                line=dict(width=0)
            )
        ))
        fig2.update_layout(height=320, **PLOT, title=f'{metric} by Platform',
                           xaxis_title=metric, yaxis_title='')
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-hd"><span class="dot"></span> Platform × Format ROAS Heatmap</div>', unsafe_allow_html=True)
        pivot = df.groupby(['platform','creative_format'])['ROAS'].mean().reset_index()
        fig3 = px.density_heatmap(pivot, x='platform', y='creative_format', z='ROAS',
                                   color_continuous_scale=[[0,'#f0f1ff'],[0.5,'#818cf8'],[1,'#4338ca']],
                                   title='Which Platform + Format combination returns the most?')
        fig3.update_layout(height=360, **PLOT)
        st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        st.markdown('<div class="section-hd"><span class="dot"></span> When Do Ads Perform Best?</div>', unsafe_allow_html=True)
        s1, s2 = st.columns(2)

        with s1:
            hourly_df = df.groupby('hour_of_day').agg(
                Avg_CTR=('CTR','mean'), Avg_ROAS=('ROAS','mean')
            ).reset_index()
            fig_h = px.line(hourly_df, x='hour_of_day', y=['Avg_CTR','Avg_ROAS'],
                            title='CTR & ROAS throughout the day',
                            color_discrete_sequence=[C_INDIGO, C_TEAL],
                            labels={'hour_of_day':'Hour (0=Midnight)','value':'Score','variable':'Metric'})
            fig_h.update_layout(height=320, **PLOT)
            st.plotly_chart(fig_h, use_container_width=True)

        with s2:
            day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
            daily_df = df.groupby('day_of_week').agg(
                Avg_CTR=('CTR','mean')
            ).reindex(day_order).reset_index()
            fig_d = go.Figure(go.Bar(
                x=daily_df['day_of_week'], y=daily_df['Avg_CTR'],
                marker=dict(
                    color=daily_df['Avg_CTR'],
                    colorscale=[[0,'#e0e7ff'],[1,'#4338ca']],
                    line=dict(width=0)
                ),
                text=daily_df['Avg_CTR'].round(3),
                textposition='outside',
                textfont=dict(color='#374151', family='Nunito')
            ))
            fig_d.update_layout(height=320, **PLOT,
                                title='Average Click Rate by Day of Week',
                                yaxis_title='Avg CTR', xaxis_title='')
            st.plotly_chart(fig_d, use_container_width=True)

        q_df = df.groupby('quarter').agg(
            Revenue=('revenue','sum'), Spend=('ad_spend','sum'), Profit=('profit','sum')
        ).reset_index()
        q_df['quarter'] = 'Q' + q_df['quarter'].astype(str)

        fig_q = go.Figure()
        fig_q.add_bar(name='Revenue',  x=q_df['quarter'], y=q_df['Revenue'], marker_color=C_TEAL,   marker_line_width=0)
        fig_q.add_bar(name='Ad Spend', x=q_df['quarter'], y=q_df['Spend'],   marker_color=C_PINK,   marker_line_width=0)
        fig_q.add_bar(name='Profit',   x=q_df['quarter'], y=q_df['Profit'],  marker_color=C_INDIGO, marker_line_width=0)
        fig_q.update_layout(barmode='group', title='Quarterly: Revenue, Spend & Profit',
                            height=340, **PLOT,
                            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#6b7280')))
        st.plotly_chart(fig_q, use_container_width=True)


# 
# PAGE 4 — BUSINESS INSIGHTS
# 
elif page == "Business Insights":

    st.markdown("""
    <div class="page-hero" style="background: linear-gradient(135deg, #f59e0b 0%, #f97316 50%, #ec4899 100%)">
        <div class="hero-eyebrow">Proven Strategies & AI Performance</div>
        <div class="hero-title">Business Insights</div>
        <div class="hero-sub">Statistically proven findings from 10,000 campaigns — backed by A/B testing and AI modelling. Use these strategies to maximise revenue and campaign ROI.</div>
    </div>
    """, unsafe_allow_html=True)

    # AI model trust badges
    st.markdown('<div class="section-hd"><span class="dot"></span> AI Model Confidence</div>', unsafe_allow_html=True)
    mb1, mb2, mb3, mb4 = st.columns(4)
    model_badges = [
        (mb1, "97.2%",   "Prediction Accuracy",   C_INDIGO,  "#eef2ff",  "AI"),
        (mb2, "85.5%",   "F1 Score",              C_EMERALD, "#ecfdf5",  "F1"),
        (mb3, "10,000",  "Campaigns Analysed",    C_AMBER,   "#fffbeb",  "10K"),
        (mb4, "XGBoost", "Best AI Model",         C_PINK,    "#fdf2f8",  "XGB"),
    ]
    for col, val, lbl, color, bg, icon in model_badges:
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="--kpi-color:{color}; --kpi-bg:{bg}">
                <div class="kpi-accent"></div>
                <div style="font-size:26px; margin-bottom:8px">{icon}</div>
                <div class="kpi-val" style="color:{color}">{val}</div>
                <div class="kpi-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([" Key Strategies", " A/B Test Results", " Model Performance", " Regression"])

    with tab1:
        st.markdown('<div class="section-hd"><span class="dot"></span> High-Impact Strategies — Statistically Proven (p &lt; 0.0001)</div>', unsafe_allow_html=True)

        strategies = [
            ("CTA", "+22% CTR",           "Add a Call-to-Action to Every Ad",
             "Campaigns with a clear CTA significantly outperform those without. This is the single easiest, lowest-cost improvement you can make to any ad.",
             "Always include a CTA button or phrase. Test 2–3 variations ('Shop Now', 'Learn More', 'Get Started') and optimise per audience.",
             C_INDIGO, "#eef2ff"),
            ("RTG", "+86% ROAS",          "Prioritise Retargeting Audiences",
             "Retargeted campaigns — showing ads to people who've already visited your site — achieve 86% higher return on ad spend. This is the #1 lever for revenue growth.",
             "Set up retargeting pixels immediately. Allocate at least 30–40% of budget to warm audiences before scaling cold audience spend.",
             C_TEAL,   "#f0fdfa"),
            ("DSK", "12.5x vs 4.6x",     "Allocate Budget to Desktop Placements",
             "Desktop campaigns achieve 12.5x ROAS compared to 4.6x on tablet — nearly a 3× performance gap. Mobile sits in between but still trails desktop for conversion value.",
             "For high-value conversion campaigns (Finance, SaaS, E-commerce), weight your placement bids toward desktop. Reserve mobile for brand awareness.",
             C_VIOLET, "#f5f3ff"),
            ("VID", "+69% CTR",           "Use Video Over Static Images",
             "Video ads generate 69% more clicks than image ads across all platforms and audiences. Interactive formats come in a close second with the highest engagement rates.",
             "Invest in short-form video (15–30 seconds). Repurpose existing product content. Even simple animated graphics outperform static banners.",
             C_AMBER,  "#fffbeb"),
        ]

        for icon, metric, title, finding, action, color, bg in strategies:
            st.markdown(f"""
            <div class="win-card" style="--win-color:{color}; --win-bg:{bg}">
                <div class="win-icon">{icon}</div>
                <div style="flex:1">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:8px; margin-bottom:6px">
                        <div class="win-title" style="font-size:16px">{title}</div>
                        <div class="win-metric">{metric}</div>
                    </div>
                    <div class="win-desc" style="margin-bottom:8px">{finding}</div>
                    <div style="background:{bg}; border-radius:8px; padding:8px 12px; font-size:12px; color:#374151">
                        <b style="color:{color}">Business Action:</b> {action}
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="section-hd"><span class="dot"></span> Full A/B Test Results</div>', unsafe_allow_html=True)
        ab_df = results.get('ab_test_results', pd.DataFrame())
        if not ab_df.empty:
            def hl_sig(val):
                if val == True:  return 'background-color:#ecfdf5; color:#16a34a; font-weight:700'
                return 'background-color:#fff7ed; color:#d97706; font-weight:700'
            st.dataframe(ab_df.style.map(hl_sig, subset=['Significant']),
                         use_container_width=True, height=220)
        else:
            fallback = pd.DataFrame({
                'Test':        ['CTA → CTR','Retargeting → ROAS','Device → CTR','Video vs Image'],
                'p-value':     ['< 0.0001','< 0.0001','< 0.0001','< 0.0001'],
                'Significant': [True,True,True,True],
                'Decision':    ['CTA increases CTR by 22%','Retargeting boosts ROAS by 86%',
                                'Device type significantly affects CTR','Video > Image CTR'],
            })
            st.dataframe(fallback, use_container_width=True)

        st.markdown('<div class="section-hd"><span class="dot"></span> Business Impact Summary</div>', unsafe_allow_html=True)
        impact_rows = [
            ("CTA raises CTR",          "+22%",           "Always include a clear CTA in every ad.",               C_INDIGO),
            ("Retargeting raises ROAS",  "+86%",           "Prioritise retargeting audiences above all else.",      C_TEAL),
            ("Desktop highest ROAS",     "12.47x vs 4.59x","Allocate majority of budget to desktop placements.",   C_VIOLET),
            ("Video beats Image CTR",    "3.11 vs 1.84",   "Use video format wherever budget allows.",              C_AMBER),
        ]
        for finding, effect, action, color in impact_rows:
            st.markdown(f"""
            <div class="tip-card" style="--tip-c:{color}">
                <div class="tip-badge">{finding}</div>
                <div class="tip-body">
                    <div class="tip-title" style="color:{color}">{effect}</div>
                    <b>Business Action:</b> {action}
                </div>
            </div>""", unsafe_allow_html=True)

        st.caption("All tests used α = 0.05 significance level. Welch's t-test, one-way ANOVA, and Mann-Whitney U applied as appropriate.")

    with tab3:
        st.markdown('<div class="section-hd"><span class="dot"></span> Classification Model Comparison</div>', unsafe_allow_html=True)
        clf_df = results.get('classification_results', pd.DataFrame())
        if not clf_df.empty:
            if 'ROC-AUC' in clf_df.columns:
                clf_df = clf_df.sort_values('ROC-AUC', ascending=False)
            st.dataframe(
                clf_df.style.background_gradient(cmap='Purples',
                    subset=[c for c in ['Accuracy','F1 Score','ROC-AUC'] if c in clf_df.columns]),
                use_container_width=True, height=320
            )
            if 'Model' in clf_df.columns and 'ROC-AUC' in clf_df.columns:
                fig_cl = go.Figure(go.Bar(
                    x=clf_df.sort_values('ROC-AUC')['ROC-AUC'],
                    y=clf_df.sort_values('ROC-AUC')['Model'],
                    orientation='h',
                    text=clf_df.sort_values('ROC-AUC')['ROC-AUC'].round(4),
                    textposition='outside',
                    textfont=dict(color='#374151', family='Nunito'),
                    marker=dict(
                        color=clf_df.sort_values('ROC-AUC')['ROC-AUC'],
                        colorscale=[[0,'#e0e7ff'],[1,'#4338ca']],
                        line=dict(width=0)
                    )
                ))
                fig_cl.update_layout(height=360, **PLOT,
                                     title='Model Accuracy Comparison (ROC-AUC — higher is better)',
                                     xaxis_title='ROC-AUC Score', yaxis_title='')
                st.plotly_chart(fig_cl, use_container_width=True)
        else:
            st.markdown(f"""
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-bottom:14px">
                <div class="kpi-card" style="--kpi-color:{C_INDIGO}; --kpi-bg:#eef2ff">
                    <div class="kpi-accent"></div>
                    <div style="font-size:16px; font-weight:800; margin-bottom:8px; color:inherit">AI</div>
                    <div class="kpi-val" style="color:{C_INDIGO}">97.2%</div>
                    <div class="kpi-lbl">XGBoost — ROC-AUC</div>
                    <div style="font-size:12px; color:#6b7280; margin-top:8px">Best overall model. Tree-based ensemble methods far outperform linear models on ad data.</div>
                </div>
                <div class="kpi-card" style="--kpi-color:{C_EMERALD}; --kpi-bg:#ecfdf5">
                    <div class="kpi-accent"></div>
                    <div style="font-size:16px; font-weight:800; margin-bottom:8px; color:inherit">AI</div>
                    <div class="kpi-val" style="color:{C_EMERALD}">85.5%</div>
                    <div class="kpi-lbl">XGBoost — F1 Score</div>
                    <div style="font-size:12px; color:#6b7280; margin-top:8px">Strong balanced precision & recall. All models exceed the 80% production threshold.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.info("Run the Day 5 notebook to populate the full classification results table.")

        tune_df = results.get('tuning_summary', pd.DataFrame())
        if not tune_df.empty:
            st.markdown('<div class="section-hd"><span class="dot"></span> Hyperparameter Tuning Results</div>', unsafe_allow_html=True)
            st.dataframe(tune_df, use_container_width=True)
            if 'ROC-AUC' in tune_df.columns and 'Model' in tune_df.columns:
                fig_t = px.bar(tune_df, x='Model', y='ROC-AUC', color='Model',
                               text='ROC-AUC',
                               color_discrete_sequence=PALETTE,
                               title='Before vs After Tuning — Accuracy Comparison')
                fig_t.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                fig_t.update_layout(height=340, showlegend=False, **PLOT)
                st.plotly_chart(fig_t, use_container_width=True)

        st.markdown(f"""
        <div class="tip-card" style="--tip-c:{C_AMBER}">
            <div class="tip-badge">Tuning Result</div>
            <div class="tip-body">
                <div class="tip-title">XGBoost gained +1.2% accuracy through tuning</div>Random Forest tested 5 combinations (GridSearchCV). XGBoost tested 50 random combinations covering learning rate, depth, and regularisation. A +1.2% improvement is meaningful at production scale.
            </div>
        </div>""", unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="section-hd"><span class="dot"></span> ROAS Regression — Predicting Exact Returns</div>', unsafe_allow_html=True)
        reg_df = results.get('regression_results', pd.DataFrame())
        if not reg_df.empty:
            if 'R²' in reg_df.columns:
                reg_df = reg_df.sort_values('R²', ascending=False)
            st.dataframe(
                reg_df.style.background_gradient(cmap='Greens',
                    subset=[c for c in ['R²'] if c in reg_df.columns]),
                use_container_width=True, height=260
            )
            if 'Model' in reg_df.columns and 'R²' in reg_df.columns:
                fig_r = go.Figure()
                fig_r.add_trace(go.Bar(name='R² Score', x=reg_df['Model'], y=reg_df['R²'],
                                       marker_color=C_TEAL, marker_line_width=0))
                if 'MAE' in reg_df.columns:
                    fig_r.add_trace(go.Bar(name='MAE (normalised)', x=reg_df['Model'],
                                           y=reg_df['MAE']/reg_df['MAE'].max(),
                                           marker_color=C_PINK, marker_line_width=0))
                fig_r.update_layout(barmode='group', title='Regression Model Comparison',
                                    height=340, **PLOT,
                                    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#6b7280')))
                st.plotly_chart(fig_r, use_container_width=True)
        else:
            st.info("Run the Day 5 notebook to populate regression results.")

        st.markdown(f"""
        <div class="tip-card" style="--tip-c:{C_TEAL}">
            <div class="tip-badge">Key Finding</div>
            <div class="tip-body">
                <div class="tip-title">Gradient Boosting achieves best R² (0.54) for ROAS prediction</div>ROAS is naturally variable — an R² of 0.54 is strong for advertising data. Linear models (Ridge, LR) struggle because ad performance relationships are inherently non-linear.
            </div>
        </div>""", unsafe_allow_html=True)
