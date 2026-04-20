import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings

warnings.filterwarnings("ignore")

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IPO Gain Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS for Premium Look ──────────────────────────────────────────────
st.markdown("""
<style>
    /* ─── Import Fonts ─── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ─── Global Styles ─── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ─── Main Background ─── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #161b33 40%, #1a1a2e 100%);
    }

    /* ─── Sidebar Styling ─── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 1px solid rgba(56, 189, 248, 0.1);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li {
        color: #e2e8f0 !important;
    }

    /* ─── Hero Title ─── */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0;
        letter-spacing: -1px;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-top: 0;
        font-weight: 400;
    }

    /* ─── Metric Cards ─── */
    .metric-card {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.08), rgba(129, 140, 248, 0.08));
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(56, 189, 248, 0.15);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 4px;
        font-weight: 600;
    }

    /* ─── Prediction Result Card ─── */
    .prediction-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(56, 189, 248, 0.1));
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        margin: 20px 0;
    }
    .prediction-value {
        font-size: 4rem;
        font-weight: 900;
        margin: 10px 0;
        letter-spacing: -2px;
    }
    .prediction-positive { color: #10b981; }
    .prediction-negative { color: #ef4444; }
    .prediction-label {
        font-size: 1rem;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    /* ─── Risk Badge ─── */
    .risk-badge {
        display: inline-block;
        padding: 8px 24px;
        border-radius: 50px;
        font-weight: 700;
        font-size: 0.9rem;
        letter-spacing: 1px;
        margin-top: 12px;
    }
    .risk-high { background: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); }
    .risk-medium { background: rgba(245, 158, 11, 0.2); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }
    .risk-low { background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }

    /* ─── Glass Card ─── */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(10px);
    }

    /* ─── Section Header ─── */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(56, 189, 248, 0.2);
    }

    /* ─── Input Styling ─── */
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(56, 189, 248, 0.2) !important;
        color: #e2e8f0 !important;
        border-radius: 10px !important;
    }
    .stNumberInput > div > div > input:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2) !important;
    }

    /* ─── Button Styling ─── */
    .stButton > button {
        background: linear-gradient(135deg, #38bdf8, #818cf8) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 48px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 1px !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 32px rgba(56, 189, 248, 0.35) !important;
    }

    /* ─── DataFrame Styling ─── */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ─── Tab Styling ─── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(56, 189, 248, 0.15) !important;
        color: #38bdf8 !important;
    }

    /* ─── Divider ─── */
    .premium-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(56, 189, 248, 0.3), transparent);
        margin: 30px 0;
    }

    /* ─── Footer ─── */
    .footer {
        text-align: center;
        color: #475569;
        font-size: 0.8rem;
        padding: 30px 0 10px 0;
        border-top: 1px solid rgba(255,255,255,0.05);
        margin-top: 60px;
    }

    /* ─── Expander ─── */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.03) !important;
        border-radius: 10px !important;
    }

    /* ─── Restore default Streamlit header for sidebar toggle ─── */
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─── Load Model & Scaler ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    class IPONet(nn.Module):
        def __init__(self, input_size):
            super(IPONet, self).__init__()
            self.fc1 = nn.Linear(input_size, 64)
            self.fc2 = nn.Linear(64, 32)
            self.output = nn.Linear(32, 1)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return self.output(x)

    scaler = joblib.load('scaler.pkl')
    model = IPONet(input_size=9)
    model.load_state_dict(torch.load('ipo_model.pth', map_location='cpu'))
    model.eval()
    return model, scaler, IPONet


@st.cache_data
def load_data():
    df = pd.read_excel('Initial Public Offering.xlsx')
    # Clean subscription columns
    cols_to_clean = ['QIB', 'HNI', 'RII', 'Total']
    for col in cols_to_clean:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace('x', ''), errors='coerce')
    # Clean Listing Gain
    df['Listing Gain'] = pd.to_numeric(
        df['Listing Gain'].astype(str).str.replace('%', '').str.replace('+', ''), errors='coerce'
    )
    # Drop unnamed columns
    unnamed = [c for c in df.columns if 'Unnamed' in c]
    df = df.drop(columns=unnamed)
    # Parse dates
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Year'] = df['Date'].dt.year
    # Fill NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df


# ─── Load Everything ─────────────────────────────────────────────────────────
files_ok = all(os.path.exists(f) for f in ['scaler.pkl', 'ipo_model.pth', 'Initial Public Offering.xlsx'])
if not files_ok:
    st.error("❌ Required files missing. Ensure `scaler.pkl`, `ipo_model.pth`, and `Initial Public Offering.xlsx` are in the app directory.")
    st.stop()

model, scaler, IPONet = load_model()
df = load_data()


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <span style="font-size: 3rem;">📈</span>
        <h2 style="
            background: linear-gradient(135deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-top: 8px;
            font-weight: 800;
        ">IPO Predictor</h2>
        <p style="color: #64748b; font-size: 0.85rem; margin-top: -8px;">
            AI-Powered Listing Gain Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🎯 Predict", "📊 Dashboard", "📈 Analytics"],
        label_visibility="collapsed"
    )

    st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="padding: 16px; background: rgba(56,189,248,0.05); border-radius: 12px; border: 1px solid rgba(56,189,248,0.1);">
        <p style="color: #94a3b8; font-size: 0.8rem; margin: 0;">
            <strong style="color: #38bdf8;">Model Info</strong><br>
            Neural Network (9→64→32→1)<br>
            Dataset: 561 IPOs (2010–2025)<br>
            Architecture: PyTorch MLP
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─── Helper Functions ─────────────────────────────────────────────────────────
def render_metric_card(value, label, prefix="", suffix=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{prefix}{value}{suffix}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def get_risk_badge(gain):
    if gain >= 15:
        return '<span class="risk-badge risk-low">🟢 STRONG LISTING EXPECTED</span>'
    elif gain >= 0:
        return '<span class="risk-badge risk-medium">🟡 MODERATE LISTING</span>'
    else:
        return '<span class="risk-badge risk-high">🔴 WEAK LISTING EXPECTED</span>'


def find_similar_ipos(input_features, df, top_n=5):
    """Find most similar IPOs from historical data based on subscription & issue size."""
    feature_cols = ['Issue_Size(crores)', 'QIB', 'HNI', 'RII', 'Total', 'Offer Price']
    df_sub = df[feature_cols + ['IPO_Name', 'Listing Gain', 'Date']].dropna()
    if df_sub.empty:
        return pd.DataFrame()

    input_vals = np.array([input_features[0], input_features[1], input_features[2],
                           input_features[3], input_features[4], input_features[5]])

    # Normalize for distance calculation
    from sklearn.preprocessing import MinMaxScaler
    mm = MinMaxScaler()
    scaled = mm.fit_transform(df_sub[feature_cols].values)
    input_scaled = mm.transform(input_vals.reshape(1, -1))

    distances = np.linalg.norm(scaled - input_scaled, axis=1)
    df_sub = df_sub.copy()
    df_sub['Similarity'] = 1 / (1 + distances)
    df_sub = df_sub.sort_values('Similarity', ascending=False).head(top_n)
    return df_sub[['IPO_Name', 'Listing Gain', 'Similarity', 'Date']]


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🎯 Predict":
    st.markdown('<h1 class="hero-title">IPO Listing Gain Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Enter IPO details to predict the expected listing day gain using our neural network model</p>', unsafe_allow_html=True)
    st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)

    # Two-column input layout
    col_left, col_spacer, col_right = st.columns([5, 0.5, 5])

    with col_left:
        st.markdown('<p class="section-header">📋 IPO Details</p>', unsafe_allow_html=True)
        issue_size = st.number_input("Issue Size (₹ Crores)", min_value=1.0, value=500.0, step=10.0,
                                     help="Total size of the IPO offering in crores")
        offer_price = st.number_input("Offer Price (₹)", min_value=1, value=500, step=10,
                                      help="Price at which shares are offered to investors")
        list_price = st.number_input("Expected List Price (₹)", min_value=0.0, value=550.0, step=10.0,
                                     help="Expected opening price on listing day")

    with col_right:
        st.markdown('<p class="section-header">📊 Subscription Data</p>', unsafe_allow_html=True)
        qib = st.number_input("QIB Subscription (x)", min_value=0.0, value=15.0, step=0.5,
                              help="Qualified Institutional Buyers subscription times")
        hni = st.number_input("HNI Subscription (x)", min_value=0.0, value=25.0, step=0.5,
                              help="High Net-worth Individual subscription times")
        rii = st.number_input("RII Subscription (x)", min_value=0.0, value=5.0, step=0.5,
                              help="Retail Individual Investors subscription times")
        total = st.number_input("Total Subscription (x)", min_value=0.0, value=12.0, step=0.5,
                                help="Overall subscription times across all categories")

    # Advanced inputs in expander
    with st.expander("⚙️ Advanced Market Parameters", expanded=False):
        adv_col1, adv_col2 = st.columns(2)
        with adv_col1:
            cmp_bse = st.number_input("Current Market Price - BSE (₹)", min_value=0.0, value=600.0, step=10.0,
                                      help="Current trading price on BSE")
        with adv_col2:
            current_gains = st.number_input("Current Gains (%)", value=10.0, step=1.0,
                                            help="Current gains from offer price in percentage")

    st.markdown("")  # spacing

    # Predict button
    predict_col1, predict_col2, predict_col3 = st.columns([2, 3, 2])
    with predict_col2:
        predict_clicked = st.button("🚀 Predict Listing Gain", use_container_width=True)

    if predict_clicked:
        # Prepare input: order must match training data
        # Features: Issue_Size, QIB, HNI, RII, Total, Offer Price, List Price, CMP(BSE), Current Gains
        input_array = np.array([[issue_size, qib, hni, rii, total, offer_price,
                                 list_price, cmp_bse, current_gains]])
        scaled_input = scaler.transform(input_array)

        with torch.no_grad():
            prediction = model(torch.tensor(scaled_input, dtype=torch.float32))
            predicted_gain = prediction.item()

        st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)

        # ─── Result Display ───
        res_col1, res_col2, res_col3 = st.columns([1, 2, 1])
        with res_col2:
            gain_class = "prediction-positive" if predicted_gain >= 0 else "prediction-negative"
            sign = "+" if predicted_gain >= 0 else ""

            st.markdown(f"""
            <div class="prediction-card">
                <p class="prediction-label">Predicted Listing Gain</p>
                <p class="prediction-value {gain_class}">{sign}{predicted_gain:.2f}%</p>
                {get_risk_badge(predicted_gain)}
            </div>
            """, unsafe_allow_html=True)

        # ─── Gauge Chart ───
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=predicted_gain,
            number={'suffix': '%', 'font': {'size': 40, 'color': '#e2e8f0'}},
            title={'text': "Gain Meter", 'font': {'size': 16, 'color': '#94a3b8'}},
            delta={'reference': 0, 'increasing': {'color': '#10b981'}, 'decreasing': {'color': '#ef4444'}},
            gauge={
                'axis': {'range': [-40, 80], 'tickcolor': '#475569', 'tickwidth': 1,
                         'tickfont': {'color': '#94a3b8'}},
                'bar': {'color': '#38bdf8', 'thickness': 0.3},
                'bgcolor': 'rgba(0,0,0,0)',
                'borderwidth': 0,
                'steps': [
                    {'range': [-40, 0], 'color': 'rgba(239, 68, 68, 0.15)'},
                    {'range': [0, 15], 'color': 'rgba(245, 158, 11, 0.15)'},
                    {'range': [15, 80], 'color': 'rgba(16, 185, 129, 0.15)'}
                ],
                'threshold': {
                    'line': {'color': '#c084fc', 'width': 3},
                    'thickness': 0.8,
                    'value': predicted_gain
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=280,
            margin=dict(t=50, b=20, l=40, r=40)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ─── Similar IPOs ───
        st.markdown('<p class="section-header">🔍 Similar Historical IPOs</p>', unsafe_allow_html=True)
        similar = find_similar_ipos(
            [issue_size, qib, hni, rii, total, offer_price], df, top_n=5
        )
        if not similar.empty:
            similar_display = similar.copy()
            similar_display['Date'] = similar_display['Date'].dt.strftime('%d %b %Y')
            similar_display['Similarity'] = (similar_display['Similarity'] * 100).round(1).astype(str) + '%'
            similar_display['Listing Gain'] = similar_display['Listing Gain'].round(2).astype(str) + '%'
            similar_display.columns = ['IPO Name', 'Listing Gain', 'Match Score', 'Date']
            similar_display = similar_display.reset_index(drop=True)
            similar_display.index = similar_display.index + 1
            st.dataframe(similar_display, use_container_width=True)
        else:
            st.info("No similar IPOs found in the dataset.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Dashboard":
    st.markdown('<h1 class="hero-title">Historical IPO Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Explore 561 IPOs from the Indian market (2010–2025)</p>', unsafe_allow_html=True)
    st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)

    # ─── KPI Cards ───
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        render_metric_card(len(df), "Total IPOs")
    with kpi2:
        avg_gain = df['Listing Gain'].mean()
        render_metric_card(f"{avg_gain:.1f}", "Avg Gain", suffix="%")
    with kpi3:
        positive = (df['Listing Gain'] > 0).sum()
        pct = (positive / len(df)) * 100
        render_metric_card(f"{pct:.0f}", "Positive Listings", suffix="%")
    with kpi4:
        max_gain = df['Listing Gain'].max()
        render_metric_card(f"{max_gain:.0f}", "Best Listing", suffix="%")

    st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)

    # ─── Year-wise Performance Chart ───
    st.markdown('<p class="section-header">📅 Year-wise IPO Performance</p>', unsafe_allow_html=True)

    yearly = df.groupby('Year').agg(
        avg_gain=('Listing Gain', 'mean'),
        count=('IPO_Name', 'count'),
        positive=('Listing Gain', lambda x: (x > 0).sum())
    ).reset_index()
    yearly['success_rate'] = (yearly['positive'] / yearly['count'] * 100).round(1)

    fig_yearly = make_subplots(specs=[[{"secondary_y": True}]])
    fig_yearly.add_trace(
        go.Bar(
            x=yearly['Year'], y=yearly['avg_gain'],
            name='Avg Gain %',
            marker=dict(
                color=yearly['avg_gain'],
                colorscale=[[0, '#ef4444'], [0.3, '#f59e0b'], [0.5, '#38bdf8'], [1, '#10b981']],
                cornerradius=6
            ),
            text=yearly['avg_gain'].round(1).astype(str) + '%',
            textposition='outside',
            textfont=dict(color='#94a3b8', size=10)
        ),
        secondary_y=False
    )
    fig_yearly.add_trace(
        go.Scatter(
            x=yearly['Year'], y=yearly['count'],
            name='IPO Count',
            line=dict(color='#c084fc', width=3),
            mode='lines+markers',
            marker=dict(size=8, color='#c084fc')
        ),
        secondary_y=True
    )
    fig_yearly.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8'),
        height=420,
        margin=dict(t=30, b=40, l=50, r=50),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(color='#94a3b8')
        ),
        hovermode='x unified'
    )
    fig_yearly.update_xaxes(gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color='#94a3b8'))
    fig_yearly.update_yaxes(title_text="Average Gain %", gridcolor='rgba(255,255,255,0.05)',
                            tickfont=dict(color='#94a3b8'), title_font=dict(color='#64748b'),
                            secondary_y=False)
    fig_yearly.update_yaxes(title_text="Number of IPOs", tickfont=dict(color='#94a3b8'),
                            title_font=dict(color='#64748b'), secondary_y=True)
    st.plotly_chart(fig_yearly, use_container_width=True)

    st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)

    # ─── Top & Bottom Performers ───
    top_col, bottom_col = st.columns(2)

    with top_col:
        st.markdown('<p class="section-header">🏆 Top 10 Performers</p>', unsafe_allow_html=True)
        top10 = df.nlargest(10, 'Listing Gain')[['IPO_Name', 'Listing Gain', 'Offer Price', 'Year']].reset_index(drop=True)
        top10.index = top10.index + 1
        top10['Listing Gain'] = top10['Listing Gain'].round(2).astype(str) + '%'
        top10['Offer Price'] = '₹' + top10['Offer Price'].astype(str)
        top10.columns = ['IPO Name', 'Gain', 'Price', 'Year']
        st.dataframe(top10, use_container_width=True)

    with bottom_col:
        st.markdown('<p class="section-header">📉 Bottom 10 Performers</p>', unsafe_allow_html=True)
        bottom10 = df.nsmallest(10, 'Listing Gain')[['IPO_Name', 'Listing Gain', 'Offer Price', 'Year']].reset_index(drop=True)
        bottom10.index = bottom10.index + 1
        bottom10['Listing Gain'] = bottom10['Listing Gain'].round(2).astype(str) + '%'
        bottom10['Offer Price'] = '₹' + bottom10['Offer Price'].astype(str)
        bottom10.columns = ['IPO Name', 'Gain', 'Price', 'Year']
        st.dataframe(bottom10, use_container_width=True)

    st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)

    # ─── Full Data Explorer ───
    st.markdown('<p class="section-header">🗂️ Full IPO Database</p>', unsafe_allow_html=True)

    # Filters
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        years = sorted(df['Year'].dropna().unique().astype(int))
        year_range = st.select_slider("Year Range", options=years,
                                      value=(min(years), max(years)))
    with filter_col2:
        gain_filter = st.selectbox("Listing Outcome", ["All", "Positive (> 0%)", "Negative (< 0%)"])
    with filter_col3:
        search = st.text_input("🔍 Search IPO Name", "")

    # Apply filters
    filtered = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]
    if gain_filter == "Positive (> 0%)":
        filtered = filtered[filtered['Listing Gain'] > 0]
    elif gain_filter == "Negative (< 0%)":
        filtered = filtered[filtered['Listing Gain'] < 0]
    if search:
        filtered = filtered[filtered['IPO_Name'].str.contains(search, case=False, na=False)]

    display_cols = ['Date', 'IPO_Name', 'Issue_Size(crores)', 'Offer Price', 'List Price',
                    'Listing Gain', 'QIB', 'HNI', 'RII', 'Total']
    display_df = filtered[display_cols].sort_values('Date', ascending=False).reset_index(drop=True)
    display_df.index = display_df.index + 1

    st.dataframe(display_df, use_container_width=True, height=400)
    st.caption(f"Showing {len(display_df)} of {len(df)} IPOs")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Analytics":
    st.markdown('<h1 class="hero-title">IPO Analytics & Insights</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Deep dive into patterns, correlations, and trends in Indian IPO market</p>', unsafe_allow_html=True)
    st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Distributions", "🔗 Correlations", "📐 Model Performance"])

    # ─── TAB 1: Distributions ───
    with tab1:
        st.markdown('<p class="section-header">Listing Gain Distribution</p>', unsafe_allow_html=True)

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=df['Listing Gain'],
            nbinsx=50,
            marker=dict(
                color='rgba(56, 189, 248, 0.6)',
                line=dict(color='rgba(56, 189, 248, 0.8)', width=1)
            ),
            hovertemplate='Gain: %{x:.1f}%<br>Count: %{y}<extra></extra>'
        ))
        fig_hist.add_vline(x=0, line_dash="dash", line_color="#ef4444", line_width=2,
                           annotation_text="Break Even", annotation_font_color="#ef4444")
        fig_hist.add_vline(x=df['Listing Gain'].median(), line_dash="dash", line_color="#10b981", line_width=2,
                           annotation_text=f"Median: {df['Listing Gain'].median():.1f}%",
                           annotation_font_color="#10b981")
        fig_hist.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8'), height=400,
            xaxis_title="Listing Gain %", yaxis_title="Number of IPOs",
            margin=dict(t=30, b=50, l=50, r=30)
        )
        fig_hist.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
        fig_hist.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
        st.plotly_chart(fig_hist, use_container_width=True)

        # Subscription Distribution
        st.markdown('<p class="section-header">Subscription Distribution by Category</p>', unsafe_allow_html=True)
        sub_cols = ['QIB', 'HNI', 'RII']
        fig_box = go.Figure()
        colors = ['#38bdf8', '#818cf8', '#c084fc']
        for i, col in enumerate(sub_cols):
            fig_box.add_trace(go.Box(
                y=df[col], name=col,
                marker_color=colors[i],
                line=dict(color=colors[i]),
                fillcolor=f'rgba({56 + i*50}, {189 - i*30}, {248 - i*50}, 0.3)'
            ))
        fig_box.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8'), height=400,
            yaxis_title="Subscription (x)",
            margin=dict(t=30, b=50, l=50, r=30),
            showlegend=False
        )
        fig_box.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
        fig_box.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
        st.plotly_chart(fig_box, use_container_width=True)

    # ─── TAB 2: Correlations ───
    with tab2:
        st.markdown('<p class="section-header">Subscription vs Listing Gain</p>', unsafe_allow_html=True)

        fig_scatter = px.scatter(
            df, x='Total', y='Listing Gain',
            color='Listing Gain',
            color_continuous_scale=['#ef4444', '#f59e0b', '#10b981'],
            hover_data=['IPO_Name', 'Offer Price', 'Year'],
            size=np.clip(df['Issue_Size(crores)'], 10, 5000),
            size_max=25,
            opacity=0.7
        )
        fig_scatter.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8'), height=500,
            xaxis_title="Total Subscription (x)",
            yaxis_title="Listing Gain %",
            margin=dict(t=30, b=50, l=50, r=30),
            coloraxis_colorbar=dict(title="Gain %", tickfont=dict(color='#94a3b8'),
                                    title_font=dict(color='#94a3b8'))
        )
        fig_scatter.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
        fig_scatter.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Correlation Heatmap
        st.markdown('<p class="section-header">Feature Correlation Heatmap</p>', unsafe_allow_html=True)
        corr_cols = ['Issue_Size(crores)', 'QIB', 'HNI', 'RII', 'Total',
                     'Offer Price', 'Listing Gain']
        corr_matrix = df[corr_cols].corr()

        fig_heat = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=['Issue Size', 'QIB', 'HNI', 'RII', 'Total', 'Offer Price', 'Listing Gain'],
            y=['Issue Size', 'QIB', 'HNI', 'RII', 'Total', 'Offer Price', 'Listing Gain'],
            colorscale=[[0, '#1e1b4b'], [0.25, '#312e81'], [0.5, '#0f172a'],
                        [0.75, '#0e7490'], [1, '#38bdf8']],
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont=dict(size=12, color='#e2e8f0'),
            hoverongaps=False
        ))
        fig_heat.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8'), height=450,
            margin=dict(t=30, b=50, l=100, r=30),
            xaxis=dict(tickfont=dict(color='#94a3b8')),
            yaxis=dict(tickfont=dict(color='#94a3b8'), autorange='reversed')
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # ─── TAB 3: Model Performance ───
    with tab3:
        st.markdown('<p class="section-header">Model Architecture</p>', unsafe_allow_html=True)

        arch_col1, arch_col2 = st.columns([1, 1])
        with arch_col1:
            st.markdown("""
            <div class="glass-card">
                <h3 style="color: #38bdf8; margin-top: 0;">🧠 Neural Network (IPONet)</h3>
                <table style="width: 100%; color: #e2e8f0; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
                        <td style="padding: 10px; color: #94a3b8;">Input Layer</td>
                        <td style="padding: 10px; font-weight: 700;">9 Features</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
                        <td style="padding: 10px; color: #94a3b8;">Hidden Layer 1</td>
                        <td style="padding: 10px; font-weight: 700;">64 Neurons (ReLU)</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
                        <td style="padding: 10px; color: #94a3b8;">Hidden Layer 2</td>
                        <td style="padding: 10px; font-weight: 700;">32 Neurons (ReLU)</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; color: #94a3b8;">Output Layer</td>
                        <td style="padding: 10px; font-weight: 700;">1 (Predicted Gain %)</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        with arch_col2:
            st.markdown("""
            <div class="glass-card">
                <h3 style="color: #818cf8; margin-top: 0;">⚙️ Training Configuration</h3>
                <table style="width: 100%; color: #e2e8f0; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
                        <td style="padding: 10px; color: #94a3b8;">Framework</td>
                        <td style="padding: 10px; font-weight: 700;">PyTorch</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
                        <td style="padding: 10px; color: #94a3b8;">Optimizer</td>
                        <td style="padding: 10px; font-weight: 700;">Adam (lr=0.01)</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
                        <td style="padding: 10px; color: #94a3b8;">Loss Function</td>
                        <td style="padding: 10px; font-weight: 700;">MSE Loss</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; color: #94a3b8;">Epochs</td>
                        <td style="padding: 10px; font-weight: 700;">100</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)

        # Actual vs Predicted Simulation
        st.markdown('<p class="section-header">📊 Model Test — Actual vs Predicted (Sample)</p>', unsafe_allow_html=True)

        # Run predictions on a sample of the dataset
        from sklearn.model_selection import train_test_split

        df_model = df.drop(['Date', 'IPO_Name', 'CMP(NSE)', 'Year'], axis=1, errors='ignore')
        unnamed = [c for c in df_model.columns if 'Unnamed' in c]
        df_model = df_model.drop(columns=unnamed, errors='ignore')
        df_model = df_model.fillna(df_model.median(numeric_only=True))

        X_all = df_model.drop('Listing Gain', axis=1).values
        y_all = df_model['Listing Gain'].values

        _, X_test, _, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

        X_test_scaled = scaler.transform(X_test)
        with torch.no_grad():
            y_pred = model(torch.tensor(X_test_scaled, dtype=torch.float32)).numpy().flatten()

        fig_avp = go.Figure()
        fig_avp.add_trace(go.Scatter(
            x=y_test, y=y_pred,
            mode='markers',
            marker=dict(color='#38bdf8', size=8, opacity=0.6,
                        line=dict(color='#818cf8', width=1)),
            name='Predictions',
            hovertemplate='Actual: %{x:.1f}%<br>Predicted: %{y:.1f}%<extra></extra>'
        ))
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig_avp.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines',
            line=dict(color='#ef4444', dash='dash', width=2),
            name='Perfect Prediction'
        ))
        fig_avp.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8'), height=450,
            xaxis_title="Actual Gain %", yaxis_title="Predicted Gain %",
            margin=dict(t=30, b=50, l=50, r=30),
            legend=dict(font=dict(color='#94a3b8'))
        )
        fig_avp.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
        fig_avp.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
        st.plotly_chart(fig_avp, use_container_width=True)

        # Metrics
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        m1, m2, m3 = st.columns(3)
        with m1:
            render_metric_card(f"{rmse:.2f}", "RMSE")
        with m2:
            render_metric_card(f"{r2:.4f}", "R² Score")
        with m3:
            render_metric_card(f"{mae:.2f}", "MAE")


# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <p>Built with ❤️ using Streamlit & PyTorch | IPO Gain Predictor v2.0</p>
</div>
""", unsafe_allow_html=True)
