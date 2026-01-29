import streamlit as st
import prediction_helper
import pandas as pd
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(page_title="CodeX Pricing Strategy", page_icon="📊", layout="wide")

# --- GLOBAL STYLING ---
st.markdown("""
<style>

/* ===== WIDGET LABEL (Age, Gender, etc.) ===== */
div[data-testid="stWidgetLabel"] {
    transform: scale(1.15) !important;
    transform-origin: left center !important;
    opacity: 1 !important;
}

/* Also ensure text inside stays bold */
div[data-testid="stWidgetLabel"] * {
    font-weight: 700 !important;
    color: white !important;
}

/* ===== Selected value inside dropdown ===== */
div[data-baseweb="select"] div {
    font-size: 20px !important;
    font-weight: 500 !important;
}

/* ===== Number input value (Age box) ===== */
input[type="number"] {
    font-size: 20px !important;
    font-weight: 500 !important;
}

/* ===== Dropdown options ===== */
ul[role="listbox"] li {
    font-size: 18px !important;
}

/* ===== Section headings ===== */
h3 {
    font-size: 32px !important;
    font-weight: 800 !important;
}

/* ===== Button styling ===== */
.stButton button {
    font-size: 22px !important;
    padding: 14px 0 !important;
    border-radius: 12px !important;
    background: linear-gradient(90deg, #ff512f, #dd2476) !important;
    border: none !important;
}

</style>
""", unsafe_allow_html=True)


# --- HEADER ---
st.title("CodeX Beverage Pricing Strategy")
st.markdown(
    "<p style='font-size:24px; font-weight:500; color:white;'>"
    "Enter customer details below to predict the optimal price range."
    "</p>",
    unsafe_allow_html=True
)
st.divider()

# --- INPUT SECTIONS ---
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("1. Demographics")
    age = st.number_input("Age", 18, 100, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    zone = st.selectbox("Zone", ["Urban", "Metro", "Rural", "Semi-Urban"])
    occupation = st.selectbox("Occupation", ["Student", "Working Professional", "Entrepreneur", "Retired"])
    income = st.selectbox("Income Level", ["<10L", "10L - 15L", "16L - 25L", "26L - 35L", "> 35L", "Not Reported"])
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("2. Consumption Habits")
    freq = st.selectbox("Weekly Consumption", ["0-2 times", "3-4 times", "5-7 times"])
    brand = st.selectbox("Current Brand", ["Newcomer", "Established"])
    size = st.selectbox("Preferred Size", ["Small (250ml)", "Medium (500ml)", "Large (1L)"])
    flavor = st.selectbox("Flavor Preference", ["Traditional", "Exotic"])
    st.markdown('</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2, gap="large")

with col3:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("3. Brand Perception")
    awareness = st.selectbox("Brand Awareness", ["0 to 1", "2 to 4", "more than 4"])
    reasons = st.selectbox("Reason for Choosing", ["Price", "Quality", "Availability", "Brand Reputation"])
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("4. Purchase Context")
    channel = st.selectbox("Purchase Channel", ["Online", "Retail Store"])
    packaging = st.selectbox("Packaging Preference", ["Simple", "Premium", "Eco-friendly"])
    health = st.selectbox("Health Concern", ["Low", "Medium", "High"])
    situation = st.selectbox("Typical Situation", ["Active (Sports, Gym)", "Social (Parties)", "Casual (At home)"])
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# --- BUTTON ---
b1, b2, b3 = st.columns([1, 2, 1])
with b2:
    predict_btn = st.button("Predict Price", type="primary", use_container_width=True)

# --- PREDICTION ---
if predict_btn:
    input_data = {
        "age": age, "gender": gender, "zone": zone, "occupation": occupation, "income_levels": income,
        "consume_frequency(weekly)": freq, "current_brand": brand, "preferable_consumption_size": size,
        "awareness_of_other_brands": awareness, "reasons_for_choosing_brands": reasons,
        "flavor_preference": flavor, "purchase_channel": channel, "packaging_preference": packaging,
        "health_concerns": health, "typical_consumption_situations": situation
    }

    with st.spinner("Analyzing customer profile and pricing sensitivity..."):
        price_range, probabilities = prediction_helper.predict_price(input_data)

    # --- HERO PRICE CARD ---
    st.markdown(f"""
<div style="background: linear-gradient(135deg, #1f4037, #2c7744);
padding: 25px;
border-radius: 14px;
text-align: center;
box-shadow: 0 8px 20px rgba(0,0,0,0.25);
margin-top: 15px;">
<div style="font-size:20px; color:#cdeccd;">RECOMMENDED PRICE RANGE</div>
<div style="font-size:48px; font-weight:800; color:white; margin-top:10px;">
₹ {price_range}
</div>
</div>
""", unsafe_allow_html=True)

    # --- STRATEGIC INSIGHT ---
    insight_map = {
        "0-2 times": "infrequent purchase behavior",
        "3-4 times": "moderate repeat purchase behavior",
        "5-7 times": "high-frequency consumption behavior"
    }

    behavior = insight_map.get(freq, "consistent consumption behavior")
    sensitivity = "high" if income in ['<10L','10L - 15L'] else "moderate"

    st.markdown(f"""
<div style="background: linear-gradient(135deg, #1e3c72, #2a5298);
padding: 26px;
border-radius: 14px;
margin-top: 20px;
box-shadow: 0 8px 20px rgba(0,0,0,0.25);">

<div style="font-size:18px; color:#c9dcff; font-weight:600;">
STRATEGIC CUSTOMER INSIGHT
</div>

<div style="font-size:24px; color:white; margin-top:12px; line-height:1.7;">
This customer falls into a <b>{occupation}</b> segment within a <b>{zone}</b> market
and demonstrates <b>{behavior}</b>. Price sensitivity is expected to be
<b>{sensitivity}</b>, making the <b>₹ {price_range}</b> band optimal for maximizing
conversion while maintaining margin integrity.
</div>

</div>
""", unsafe_allow_html=True)

    # --- CONFIDENCE CHART ---
    if probabilities:
        st.markdown("### 🎯 Pricing Band Probability Distribution")

        df_probs = pd.DataFrame({
            "Price Range": list(probabilities.keys()),
            "Confidence": [v * 100 for v in probabilities.values()]
        }).sort_values("Confidence", ascending=False)

        fig = px.bar(df_probs, x="Price Range", y="Confidence", text="Confidence",
                     color="Confidence", color_continuous_scale="Blues")

        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                          font=dict(color="white", size=14),
                          xaxis_title="Price Band", yaxis_title="Model Confidence (%)",
                          coloraxis_showscale=False, height=420)
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
        fig.update_traces(marker_line_width=0)

        st.plotly_chart(fig, use_container_width=True)
