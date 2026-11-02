import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="MHS Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'DM Serif Display', serif; }

.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1.5rem; color: white;
}
.hero h1 { color: white; font-size: 2rem; margin: 0 0 0.25rem 0; }
.hero p  { color: #94a3b8; margin: 0; font-size: 0.95rem; }

.price-card {
    background: linear-gradient(135deg, #0f3460, #533483);
    border-radius: 16px; padding: 2rem; text-align: center; color: white;
    box-shadow: 0 8px 32px rgba(83,52,131,0.3);
}
.price-card .label { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 2px; color: #c4b5fd; }
.price-card .price { font-family: 'DM Serif Display', serif; font-size: 3rem; margin: 0.5rem 0; }
.price-card .range { font-size: 0.9rem; color: #e2e8f0; }

.metric-row { display: flex; gap: 1rem; margin-top: 1rem; }
.metric-card {
    flex: 1; background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 1rem 1.25rem; text-align: center;
}
.metric-card .m-label { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; }
.metric-card .m-value { font-size: 1.3rem; font-weight: 600; color: #1e293b; margin-top: 0.2rem; }

.context-card {
    background: #f0fdf4; border: 1px solid #bbf7d0;
    border-radius: 12px; padding: 1.25rem 1.5rem; margin-top: 1rem;
}
.context-card h4 { color: #15803d; margin: 0 0 0.5rem 0; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; }
.context-card p  { color: #166534; margin: 0; font-size: 0.95rem; line-height: 1.7; }

.sidebar-section {
    font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 2px; color: #94a3b8;
    margin: 1.25rem 0 0.5rem 0; padding-bottom: 0.25rem; border-bottom: 1px solid #e2e8f0;
}
.warn-box {
    background: #fef3c7; border: 1px solid #fcd34d; border-radius: 10px;
    padding: 0.75rem 1rem; font-size: 0.88rem; color: #92400e; margin-top: 0.75rem;
}
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──
REGION_MAP = {1:"Northeast", 2:"Midwest", 3:"South", 4:"Southwest", 5:"West"}
REGION_INV = {v:k for k,v in REGION_MAP.items()}
REGION_AVG = {1:108000, 2:88000, 3:95000, 4:112000, 5:158000}
YEAR_AVG   = {2014:70000,2015:76000,2016:82000,2017:89000,2018:96000,
              2019:102000,2020:105000,2021:115000,2022:128000,2023:132000,2024:135000,2025:142000,2026:149000,2027:156000}
YEAR_INDEX = {2014:0.62,2015:0.67,2016:0.72,2017:0.79,2018:0.87,
              2019:0.93,2020:0.96,2021:1.09,2022:1.28,2023:1.34,2024:1.35,2025:1.42,2026:1.49,2027:1.56}
MONTH_INDEX= {1:0.97,2:0.98,3:1.00,4:1.01,5:1.02,6:1.02,
              7:1.01,8:1.00,9:0.99,10:0.99,11:0.98,12:0.97}
MONTH_NAMES= {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
              7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
LOCATION_VAL = {"On Dealer Lot":1, "Private Land":2, "Rented Land":3}

# ── Load model ──
@st.cache_resource
def load_model():
    return joblib.load("src/model.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error("**model.pkl not found.** Run `train_and_export.py` first.")
    st.stop()

# ── Sidebar ──
with st.sidebar:
    st.markdown("## 🏠 Home Details")

    st.markdown('<div class="sidebar-section">🏗️ Structure</div>', unsafe_allow_html=True)
    sqft     = st.slider("Square Footage", 200, 4000, 1200, 50)
    sections = st.selectbox("Sections", [1, 2, 3], index=1,
                            help="Number of factory-built sections")
    bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5], index=2)

    st.markdown('<div class="sidebar-section">📍 Location & Site</div>', unsafe_allow_html=True)
    region   = st.selectbox("Region", list(REGION_MAP.values()), index=2)
    location = st.selectbox("Site Type", list(LOCATION_VAL.keys()), index=1,
                            help="Where the home will be placed")

    st.markdown('<div class="sidebar-section">📅 Timing</div>', unsafe_allow_html=True)
    year  = st.slider("Year Shipped", 2014, 2027, 2022)
    month = st.selectbox("Month Shipped", list(MONTH_NAMES.keys()),
                         format_func=lambda x: MONTH_NAMES[x], index=5)

    st.markdown('<div class="sidebar-section">💰 Financing & Title</div>', unsafe_allow_html=True)
    titled     = st.selectbox("Titled as Real Property?", ["Yes","No"], index=0,
                               help="Real property title typically increases value")
    foundation = st.selectbox("Foundation", ["Permanent","Non-Permanent"], index=0)
    secured    = st.selectbox("Financially Secured?", ["Yes","No"], index=0)
    status     = st.selectbox("Home Status", ["New","Used"], index=0)

    # Optional fields hidden — defaulting to unknown
    footings_known  = False
    footings        = -1
    finaldest_known = False
    finaldest       = -1

# ── Parse ──
region_code    = REGION_INV[region]
titled_val     = 1 if titled == "Yes" else 2
location_val   = LOCATION_VAL[location]
foundation_val = 1 if foundation == "Permanent" else 2
secured_val    = 1 if secured == "Yes" else 2
status_val     = 1 if status == "New" else 2

# ── Validation ──
warnings = []
if bedrooms >= 4 and sections == 1:
    warnings.append("⚠️ 4+ bedrooms in a single-section home is unusual — prediction may be less reliable.")
if sqft < 400 and sections == 2:
    warnings.append("⚠️ Very small square footage for a double-section home.")
if year > 2024:
    warnings.append(f"🔮 {year} is a future estimate — extrapolated beyond training data (2014–2024). Treat as speculative.")

# ── Feature engineering ──
def build_features():
    quarter        = ((month - 1) // 3) + 1
    year_since2014 = year - 2014
    return pd.DataFrame([{
        "SQFT":sqft,"SECTIONS":sections,"BEDROOMS":bedrooms,
        "REGION":region_code,"FOUNDATION":foundation_val,
        "TITLED":titled_val,"SECURED":secured_val,"LOCATION":location_val,"YEAR":year,
        "BEDROOMS_MISSING":0,"TITLED_MISSING":0,"LOCATION_MISSING":0,
        "FOUNDATION_MISSING":0,"SECURED_MISSING":0,
        "SQFT_per_SECTION":    sqft/sections,
        "SQFT_squared":        sqft**2,
        "SECTIONS_BEDROOMS":   sections*bedrooms,
        "SQFT_log":            np.log1p(sqft),
        "YEAR_since_2014":     year_since2014,
        "SQFT_YEAR":           sqft*year_since2014,
        "SECTION_YEAR":        sections*year_since2014,
        "REGION_SECTION":      region_code*10+sections,
        "REGION_YEAR":         region_code*year_since2014,
        "FOUNDATION_LOCATION": foundation_val*10+location_val,
        "TITLED_SECURED":      titled_val*10+secured_val,
        "YEAR_MARKET_INDEX":   YEAR_INDEX.get(year,1.0),
        "STATUS":status_val,"MONTH":month,"QUARTER":quarter,
        "MONTH_MARKET_INDEX":  MONTH_INDEX.get(month,1.0),
        "FOOTINGS":footings,"FOOTINGS_MISSING":0 if footings_known else 1,
        "FINALDEST":finaldest,"FINALDEST_MISSING":0 if finaldest_known else 1,
    }])

prediction = model.predict(build_features())[0]
MAE        = 13380
low, high  = prediction - MAE, prediction + MAE
region_avg = REGION_AVG[region_code]
year_avg   = YEAR_AVG.get(year, 100000)
vs_region  = ((prediction - region_avg) / region_avg) * 100
vs_year    = ((prediction - year_avg)   / year_avg)   * 100

# ── Hero ──
st.markdown(f"""
<div class="hero">
    <h1>🏠 Manufactured Home Price Predictor</h1>
    <p>MHS Dataset · LightGBM + Anomaly Removal · Trained on 60,000+ homes (2014–2024)</p>
</div>
""", unsafe_allow_html=True)

col_pred, col_context = st.columns([1, 1], gap="large")

with col_pred:
    st.markdown(f"""
    <div class="price-card">
        <div class="label">Estimated Price</div>
        <div class="price">${prediction:,.0f}</div>
        <div class="range">Likely range &nbsp;·&nbsp; ${low:,.0f} — ${high:,.0f}</div>
    </div>
    <div class="metric-row">
        <div class="metric-card"><div class="m-label">Model R²</div><div class="m-value">0.806</div></div>
        <div class="metric-card"><div class="m-label">Avg Error</div><div class="m-value">±$13,380</div></div>
        <div class="metric-card"><div class="m-label">Homes Trained</div><div class="m-value">60K+</div></div>
    </div>
    """, unsafe_allow_html=True)

    for w in warnings:
        st.markdown(f'<div class="warn-box">{w}</div>', unsafe_allow_html=True)

with col_context:
    if vs_region < -15:
        tier, tier_color = "Budget", "#1d4ed8"
    elif vs_region < 15:
        tier, tier_color = "Mid-Range", "#15803d"
    else:
        tier, tier_color = "Premium", "#b91c1c"

    section_label = {1:"single",2:"double",3:"triple"}[sections]
    size_note = "large" if sqft > 2000 else ("compact" if sqft < 800 else "mid-size")
    dollar_diff = abs(int(prediction - region_avg))
    more_less = "more" if prediction > region_avg else "less"

    st.markdown(f"""
    <div class="context-card">
        <h4>📊 Price Tier</h4>
        <p>
            This home falls in the <strong style="color:{tier_color}">{tier}</strong> range
            for the {region} region in {year}.<br><br>
            Key drivers: <strong>{section_label}-section build</strong> with a
            <strong>{size_note} {sqft:,} sqft floor plan</strong>.<br>
            {region} regional average is <strong>${region_avg:,}</strong> —
            this estimate is <strong>${dollar_diff:,} {more_less}</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Regional bar chart
    st.markdown("#### Price vs. All Regions")
    fig, ax = plt.subplots(figsize=(5, 2.8))
    names  = [REGION_MAP[k] for k in REGION_AVG]
    avgs   = [REGION_AVG[k]/1000 for k in REGION_AVG]
    colors = ["#0f3460" if REGION_MAP[k]==region else "#cbd5e1" for k in REGION_AVG]
    ax.barh(names, avgs, color=colors, height=0.5, zorder=2)
    ax.axvline(prediction/1000, color="#f59e0b", lw=2, ls="--", zorder=3,
               label=f"Your estimate ${prediction/1000:.0f}k")
    for i, (name, val) in enumerate(zip(names, avgs)):
        ax.text(val+1, i, f"${val:.0f}k", va="center", fontsize=8, color="#475569")
    ax.set_xlabel("Price ($k)", fontsize=8)
    ax.set_xlim(0, max(avgs)+35)
    ax.tick_params(labelsize=8)
    ax.spines[["top","right","bottom"]].set_visible(False)
    ax.grid(axis="x", alpha=0.2, zorder=1)
    ax.legend(fontsize=7.5, loc="lower right")
    fig.patch.set_alpha(0); ax.set_facecolor("none")
    plt.tight_layout()
    st.pyplot(fig)

st.divider()

with st.expander("📋 Input Summary"):
    st.dataframe(pd.DataFrame({
        "Field": ["SQFT","Sections","Bedrooms","Region","Year","Month",
                  "Site Type","Foundation","Titled","Secured","Status"],
        "Value": [sqft, sections, bedrooms, region, year, MONTH_NAMES[month],
                  location, foundation, titled, secured, status]
    }), use_container_width=True, hide_index=True)

st.caption(
    "Model: LightGBM + Anomaly Removal (2.5σ) · R² = 0.806 on filtered test set · "
    "Price range = prediction ± MAE ($13,380) · Regional averages are MHS survey medians."
)