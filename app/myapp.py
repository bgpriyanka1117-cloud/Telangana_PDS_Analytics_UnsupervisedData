import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk
from sklearn.preprocessing import MinMaxScaler

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Telangana pds Clustering and Anomaly",
    layout="wide"
)
st.markdown(
    """
    <h2 style='
        color:#FF5733;
        font-weight:bold;
        text-align:center;
        font-size:32px;
        text-shadow:2px 2px 4px gray;
    '>
     Telangana PDS Shop Performance Dashboard
    </h2>
    """,
    unsafe_allow_html=True
)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv(
        "../data/Master_data/Final_Cluster_Lables_Profile.csv"
    )
    return df


df = load_data()

# =========================
# FEATURE ENGINEERING
# =========================
scaler = MinMaxScaler()

df["fraud_score"] = scaler.fit_transform(
    df[["trans_per_card_FRAUD_DETECTION"]]
)

df["performance_score"] = scaler.fit_transform(
    df[["totalAmount"]]
)

df["utilization_score"] = scaler.fit_transform(
    df[["utilization_ratio"]]
)

# Final AI Shop Score
df["shop_score"] = (
    df["performance_score"] * 0.4 +
    df["utilization_score"] * 0.3 +
    (1 - df["fraud_score"]) * 0.3
)

# =========================
# SIDEBAR FILTERS
# =========================
st.sidebar.header(" Filters")

district = st.sidebar.selectbox(
    "Select District",
    sorted(df["distName"].dropna().unique())
)

year = st.sidebar.selectbox(
    "Select Year",
    sorted(df["year"].dropna().unique())
)

# Filter Data
filtered_df = df[
    (df["distName"] == district) &
    (df["year"] == year)
]

# =========================
# SHOP SEARCH
# =========================
st.sidebar.subheader("🔍 Shop Search")

shop_id = st.sidebar.number_input(
    "Search Shop No",
    min_value=int(df["shopNo"].min()),
    max_value=int(df["shopNo"].max()),
    step=1
)

shop_df = df[df["shopNo"] == shop_id]


# =========================
# KPI SECTION
# =========================
st.markdown(
    """
    <h3 style='
        font-weight:bold;
        color:#FF5733;
        text-align:center;
        font-size:28px;
    '>
     Key Performance Indicators (KPI)
    </h3>
    """,
    unsafe_allow_html=True
)
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="🏪 Total Shops",
        value=filtered_df["shopNo"].nunique()
    )

with col2:
    st.metric(
        label="💰 Total Revenue",
        value=f"₹ {filtered_df['totalAmount'].sum():,.0f}"
    )

with col3:
    st.metric(
        label="📈 Avg Utilization",
        value=f"{filtered_df['utilization_ratio'].mean():.2f}"
    )

with col4:
    st.metric(
        label="⚠ Avg Fraud Risk",
        value=f"{filtered_df['fraud_score'].mean():.2f}"
    )

with col5:
    st.metric(
        label="⭐ Avg Shop Score",
        value=f"{filtered_df['shop_score'].mean():.2f}"
    )

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺 Geo Heatmap",
    "🔥 AI Fraud + Score",
    "📊 Cluster Analysis",
    "🏆 Shop Ranking"
])

# =========================
# TAB 1 - GEO HEATMAP
# =========================
with tab1:

    st.subheader(
        "🗺 Geospatial Heatmap (Demand + Activity)"
    )

    if not filtered_df.empty:

        st.pydeck_chart(
            pdk.Deck(
                initial_view_state=pdk.ViewState(
                    latitude=filtered_df[
                        "latitude"
                    ].mean(),
                    longitude=filtered_df[
                        "longitude"
                    ].mean(),
                    zoom=6,
                    pitch=50,
                ),

                layers=[
                    pdk.Layer(
                        "HeatmapLayer",
                        data=filtered_df,
                        get_position="[longitude, latitude]",
                        get_weight="totalAmount",
                        radiusPixels=60,
                    )
                ],
            )
        )

    else:
        st.warning(
            "No data available for selected filter."
        )

# =========================
# TAB 2 - FRAUD + SCORE
# =========================
with tab2:

    st.subheader(
        "🔥 AI Fraud Detection + Shop Score"
    )

    score_df = filtered_df[[
        "shopNo",
        "fraud_score",
        "performance_score",
        "utilization_score",
        "shop_score"
    ]].sort_values(
        "shop_score",
        ascending=False
    )

    st.dataframe(
        score_df,
        use_container_width=True
    )

    fig = px.histogram(
        filtered_df,
        x="fraud_score",
        color="Cluster",
        title="Fraud Score Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =========================
# TAB 3 - CLUSTER ANALYSIS
# =========================
with tab3:

    st.subheader(
        "📊 Cluster Behavior Analysis"
    )

    cluster_summary = filtered_df.groupby(
        "Cluster"
    )[[
        "totalAmount",
        "utilization_ratio",
        "fraud_score",
        "shop_score"
    ]].mean().reset_index()

    fig = px.bar(
        cluster_summary,
        x="Cluster",
        y="totalAmount",
        color="Cluster",
        title="Cluster Revenue Analysis"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(
        cluster_summary,
        use_container_width=True
    )

# =========================
# TAB 4 - SHOP RANKING
# =========================
with tab4:

    st.subheader(
        "🏆 Top Performing Shops"
    )

    top_shops = filtered_df.sort_values(
        "shop_score",
        ascending=False
    ).head(20)

    st.dataframe(
        top_shops[[
            "shopNo",
            "distName",
            "totalAmount",
            "shop_score",
            "Cluster"
        ]],
        use_container_width=True
    )

    fig = px.bar(
        top_shops,
        x="shopNo",
        y="shop_score",
        color="shop_score",
        title="Top 20 Shops Ranking"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =========================
# SHOP DETAILS ANALYSIS
# =========================
st.subheader("🔍 Shop Deep Analysis")

if not shop_df.empty:

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "⭐ Shop Score",
            round(
                float(
                    shop_df["shop_score"].values[0]
                ),
                2
            )
        )

    with col2:
        st.metric(
            "⚠ Fraud Risk",
            round(
                float(
                    shop_df["fraud_score"].values[0]
                ),
                2
            )
        )

    st.write(
        "### Shop Full Details"
    )

    st.dataframe(
        shop_df.T,
        use_container_width=True
    )

else:
    st.warning(
        "Shop ID not found."
    )
# =========================
# FOOTER / AUTHOR CREDIT
# =========================
st.markdown(
    """
    <hr style="margin-top:30px; margin-bottom:10px;">

    <div style="
        text-align:center;
        padding:12px;
        border-radius:10px;
        background-color:#f5f5f5;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
    ">

    <h4 style="
        color:#2E86C1;
        font-weight:bold;
        margin-bottom:5px;
    ">
        Developed & Designed by Priyanka Kumari
    </h4>

    <p style="
        color:gray;
        font-size:15px;
        margin-top:0;
    ">
        Telangana PDS Shop Performance Dashboard | Data Analytics Project
    </p>

    </div>
    """,
    unsafe_allow_html=True
)