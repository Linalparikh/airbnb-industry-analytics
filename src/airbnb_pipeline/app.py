import math
import pandas as pd
import streamlit as st
import plotly.express as px
import pydeck as pdk
from datetime import datetime

from airbnb_pipeline.db import get_db, get_cols, collections_exist
from airbnb_pipeline.config import *


st.set_page_config(
    page_title="Airbnb Analytics Dashboard",
    page_icon="🏠",
    layout="wide"
)

CUSTOM_CSS = """
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
div[data-testid="stMetricValue"] { font-size: 28px; }
div[data-testid="stMetricLabel"] { font-size: 14px; opacity: 0.85; }
.small-note { opacity: 0.75; font-size: 0.9rem; }
hr { margin: 1.2rem 0; opacity: 0.2; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def pick_first(cols, candidates):
    cols_l = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_l:
            return cols_l[cand.lower()]
    return None

@st.cache_data(ttl=300)
def get_counts():
    db = get_db()
    listings, reviews = get_cols(db)
    return {
        "listings": listings.estimated_document_count(),
        "reviews": reviews.estimated_document_count(),
    }

@st.cache_data(ttl=300)
def get_sample_listings(limit=SAMPLE_ROWS):
    db = get_db()
    listings, _ = get_cols(db)

    pipeline = [{"$sample": {"size": int(limit)}}]
    rows = list(listings.aggregate(pipeline, allowDiskUse=True))
    df = pd.DataFrame(rows)

    if "_id" in df.columns:
        df = df.drop(columns=["_id"])

    return df

@st.cache_data(ttl=300)
def agg_room_type_counts():
    db = get_db()
    listings, _ = get_cols(db)

    pipeline = [
        {"$group": {"_id": "$room_type", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]
    rows = list(listings.aggregate(pipeline, allowDiskUse=True))
    df = pd.DataFrame(rows).rename(columns={"_id": "room_type"})
    df["room_type"] = df["room_type"].fillna("Unknown")
    return df

@st.cache_data(ttl=300)
def agg_top_locations(top_n=TOP_N_LOCATIONS):
    db = get_db()
    listings, _ = get_cols(db)

    sample = listings.find_one({}, projection={"host_location": 1, "city": 1, "neighbourhood": 1, "neighborhood": 1, "location": 1})
    possible_fields = ["host_location", "city", "neighbourhood", "neighborhood", "location"]
    chosen = None
    if sample:
        for f in possible_fields:
            if f in sample:
                chosen = f
                break
    if not chosen:
        chosen = "host_location"  # fallback

    pipeline = [
        {"$group": {"_id": f"${chosen}", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": int(top_n)},
    ]
    rows = list(listings.aggregate(pipeline, allowDiskUse=True))
    df = pd.DataFrame(rows).rename(columns={"_id": "location"})
    df["location"] = df["location"].fillna("Unknown")
    return df, chosen

@st.cache_data(ttl=300)
def agg_price_stats():
    db = get_db()
    listings, _ = get_cols(db)

    pipeline = [
        {"$match": {"price": {"$exists": True, "$ne": None}}},
        {"$group": {
            "_id": None,
            "avg_price": {"$avg": "$price"},
            "min_price": {"$min": "$price"},
            "max_price": {"$max": "$price"},
        }},
    ]
    rows = list(listings.aggregate(pipeline, allowDiskUse=True))
    if not rows:
        return {"avg_price": None, "min_price": None, "max_price": None}
    r = rows[0]
    return {
        "avg_price": r.get("avg_price"),
        "min_price": r.get("min_price"),
        "max_price": r.get("max_price"),
    }

st.sidebar.title("Filters")

price_cap = st.sidebar.slider(
    "Price cap (for charts)",
    min_value=50,
    max_value=5000,
    value=PRICE_CAP_DEFAULT,
    step=50
)

sample_n = st.sidebar.select_slider(
    "Listings sample size",
    options=[3000, 5000, 10000, 15000, 25000],
    value=SAMPLE_ROWS
)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: If charts look squeezed, lower the price cap.")

st.title("🏠 Airbnb Analytics Dashboard (Industry Style)")
st.markdown('<div class="small-note">MongoDB-backed, scalable aggregations + interactive visual analytics.</div>', unsafe_allow_html=True)

db = get_db()
if not collections_exist(db):
    st.error("Mongo collections not found. Make sure you ingested data into MongoDB (listings_raw, reviews_raw).")
    st.stop()

counts = get_counts()
price_stats = agg_price_stats()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Listings (raw)", f"{counts['listings']:,}")
c2.metric("Reviews (raw)", f"{counts['reviews']:,}")

if price_stats["avg_price"] is not None:
    c3.metric("Avg price", f"${price_stats['avg_price']:.2f}")
    c4.metric("Max price", f"${price_stats['max_price']:.0f}")
else:
    c3.metric("Avg price", "N/A")
    c4.metric("Max price", "N/A")

st.markdown("<hr/>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 Overview", "🧭 Explore Listings", "🗺️ Map (if coords exist)"])

with tab1:
    left, right = st.columns([1, 1])

    room_df = agg_room_type_counts()
    top_loc_df, loc_field = agg_top_locations()

    with left:
        st.subheader("Listings by Room Type")
        fig = px.bar(room_df, x="room_type", y="count")
        fig.update_layout(height=380, xaxis_title="", yaxis_title="Listings")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader(f"Top Locations (grouped by: {loc_field})")
        fig2 = px.bar(top_loc_df, x="location", y="count")
        fig2.update_layout(height=380, xaxis_title="", yaxis_title="Listings")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Price Distribution (Capped)")
    df = get_sample_listings(sample_n)

    price_col = pick_first(df.columns, ["price", "price_usd", "nightly_price"])
    if price_col is None:
        st.warning("No price column found in your listings sample. Add/verify a price field.")
    else:
        dfx = df.copy()
        dfx[price_col] = pd.to_numeric(dfx[price_col], errors="coerce")
        dfx = dfx.dropna(subset=[price_col])
        dfx["price_capped"] = dfx[price_col].clip(upper=price_cap)

        fig3 = px.histogram(dfx, x="price_capped", nbins=50)
        fig3.update_layout(height=350, xaxis_title=f"Price (capped at {price_cap})", yaxis_title="Count")
        st.plotly_chart(fig3, use_container_width=True)


with tab2:
    st.subheader("Interactive Listing Explorer")

    df = get_sample_listings(sample_n)

    room_col = pick_first(df.columns, ["room_type", "roomType"])
    name_col = pick_first(df.columns, ["name", "listing_name", "title"])
    host_col = pick_first(df.columns, ["host_id", "hostId"])
    superhost_col = pick_first(df.columns, ["host_is_superhost", "superhost", "is_superhost"])
    loc_col = pick_first(df.columns, ["host_location", "city", "neighbourhood", "neighborhood", "location"])
    price_col = pick_first(df.columns, ["price", "price_usd", "nightly_price"])

    f1, f2, f3, f4 = st.columns(4)

    if room_col:
        room_types = sorted([x for x in df[room_col].dropna().unique().tolist() if str(x).strip() != ""])
        selected_rooms = f1.multiselect("Room type", room_types, default=room_types)
    else:
        selected_rooms = None
        f1.info("room_type not found")

    if loc_col:
        top_locs = df[loc_col].fillna("Unknown").value_counts().head(50).index.tolist()
        selected_locs = f2.multiselect("Location (top 50)", top_locs, default=top_locs[:10])
    else:
        selected_locs = None
        f2.info("location not found")

    if superhost_col:
        superhost_only = f3.checkbox("Superhost only", value=False)
    else:
        superhost_only = False
        f3.info("superhost flag not found")

    if price_col:
        df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
        pmin = float(df[price_col].min(skipna=True)) if df[price_col].notna().any() else 0.0
        pmax = float(df[price_col].max(skipna=True)) if df[price_col].notna().any() else 0.0
        pmax = max(pmax, 1.0)
        pr = f4.slider("Price filter", min_value=0.0, max_value=pmax, value=(0.0, min(pmax, float(price_cap))))
    else:
        pr = None
        f4.info("price not found")

    dff = df.copy()

    if room_col and selected_rooms is not None:
        dff = dff[dff[room_col].isin(selected_rooms)]

    if loc_col and selected_locs is not None:
        dff[loc_col] = dff[loc_col].fillna("Unknown")
        dff = dff[dff[loc_col].isin(selected_locs)]

    if superhost_col and superhost_only:
        s = dff[superhost_col].astype(str).str.lower()
        dff = dff[s.isin(["t", "true", "1", "yes"])]

    if price_col and pr is not None:
        dff = dff[dff[price_col].between(pr[0], pr[1], inclusive="both")]

    st.markdown(f"✅ Filtered rows: **{len(dff):,}** (from sample {len(df):,})")

    cA, cB = st.columns(2)

    with cA:
        if room_col:
            tmp = dff[room_col].fillna("Unknown").value_counts().reset_index()
            tmp.columns = ["room_type", "count"]
            fig = px.pie(tmp, names="room_type", values="count", hole=0.45)
            fig.update_layout(height=360)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No room_type column available for pie chart.")

    with cB:
        if price_col and dff[price_col].notna().any():
            dfx = dff.dropna(subset=[price_col]).copy()
            dfx["price_capped"] = dfx[price_col].clip(upper=price_cap)
            fig = px.box(dfx, y="price_capped")
            fig.update_layout(height=360, yaxis_title=f"Price (capped at {price_cap})")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No price data available for box plot.")

    st.subheader("Sample Records")
    show_cols = []
    for c in [name_col, room_col, loc_col, price_col, host_col, superhost_col]:
        if c and c in dff.columns:
            show_cols.append(c)

    if show_cols:
        st.dataframe(dff[show_cols].head(200), use_container_width=True)
    else:
        st.dataframe(dff.head(200), use_container_width=True)


with tab3:
    st.subheader("Map View (requires latitude/longitude fields)")

    df = get_sample_listings(sample_n)
    lat_col = pick_first(df.columns, ["latitude", "lat"])
    lon_col = pick_first(df.columns, ["longitude", "lon", "lng"])

    if not lat_col or not lon_col:
        st.warning("No latitude/longitude columns found. If your dataset contains coordinates, ensure they are ingested.")
    else:
        df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
        df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
        dfm = df.dropna(subset=[lat_col, lon_col]).head(5000).copy()

        st.caption(f"Plotting {len(dfm):,} points (sampled).")

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=dfm,
            get_position=[lon_col, lat_col],
            get_radius=40,
            pickable=True,
            auto_highlight=True
        )

        view_state = pdk.ViewState(
            latitude=float(dfm[lat_col].median()),
            longitude=float(dfm[lon_col].median()),
            zoom=2.5,
            pitch=0
        )

        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))
