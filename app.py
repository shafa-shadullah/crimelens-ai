import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap

from reportlab.pdfgen import canvas
import datetime
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="CrimeLens Enterprise",
    page_icon="🚔",
    layout="wide"
)

# =========================
# SIMPLE LOGIN SYSTEM
# =========================
st.sidebar.title("🔐 Login Panel")

user = st.sidebar.text_input("Username")
pwd = st.sidebar.text_input("Password", type="password")

if user != "admin" or pwd != "1234":
    st.warning("Enter valid credentials (admin / 1234)")
    st.stop()

st.sidebar.success("Logged in as Admin 🚔")

# =========================
# LOAD DATA
# =========================
df = pd.read_excel("crimelens.coimbatore.xlsx").dropna()

# =========================
# ENCODING
# =========================
loc_encoder = LabelEncoder()
crime_encoder = LabelEncoder()

df["loc_enc"] = loc_encoder.fit_transform(df["Location"])
df["crime_enc"] = crime_encoder.fit_transform(df["Crime_Type"])

# =========================
# MODEL
# =========================
X = df[["loc_enc"]]
y = df["crime_enc"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# =========================
# PDF REPORT FUNCTION
# =========================
def generate_report(location, crime, risk):
    file_name = "crime_report.pdf"
    c = canvas.Canvas(file_name)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(200, 800, "CrimeLens AI Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, 750, f"Location: {location}")
    c.drawString(50, 730, f"Predicted Crime: {crime}")
    c.drawString(50, 710, f"Risk Score: {risk}/100")
    c.drawString(50, 690, f"Generated: {datetime.datetime.now()}")

    c.save()
    return file_name

# =========================
# TITLE
# =========================
st.title("🚔 CrimeLens ENTERPRISE CONTROL SYSTEM")

# =========================
# SIDEBAR NAV
# =========================
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "AI Prediction", "Heatmap", "Analytics"]
)

# =========================
# DASHBOARD
# =========================
if page == "Dashboard":

    st.subheader("📊 Live Crime Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Locations", df["Location"].nunique())
    col2.metric("Crime Types", df["Crime_Type"].nunique())
    col3.metric("Records", len(df))

    fig, ax = plt.subplots()
    df["Crime_Type"].value_counts().plot(kind="bar", ax=ax)
    st.pyplot(fig)

# =========================
# AI PREDICTION + PDF
# =========================
elif page == "AI Prediction":

    st.subheader("🤖 Enterprise AI Prediction System")

    location = st.selectbox("Select Location", df["Location"].unique())

    if st.button("Run AI Analysis", use_container_width=True):

        loc_val = loc_encoder.transform([location])[0]

        pred = model.predict([[loc_val]])
        crime_pred = crime_encoder.inverse_transform(pred)[0]

        prob = model.predict_proba([[loc_val]]).max()
        risk = int(prob * 100)

        if risk >= 70:
            st.error(f"🔴 HIGH RISK - {risk}/100")
        elif risk >= 40:
            st.warning(f"🟠 MEDIUM RISK - {risk}/100")
        else:
            st.success(f"🟢 LOW RISK - {risk}/100")

        st.info(f"Predicted Crime: {crime_pred}")

        # =========================
        # PDF DOWNLOAD BUTTON
        # =========================
        pdf_file = generate_report(location, crime_pred, risk)

        with open(pdf_file, "rb") as f:
            st.download_button(
                "📄 Download Crime Report",
                f,
                file_name="Crime_Report.pdf"
            )

# =========================
# HEATMAP
# =========================
elif page == "Heatmap":

    st.subheader("🗺 Crime Intelligence Heatmap")

    heat_data = df[["Latitude", "Longitude"]].values.tolist()

    m = folium.Map(
        location=[df["Latitude"].mean(), df["Longitude"].mean()],
        zoom_start=12,
        tiles="CartoDB dark_matter"
    )

    HeatMap(heat_data).add_to(m)

    st_folium(m, width=1100, height=600)

# =========================
# ANALYTICS
# =========================
elif page == "Analytics":

    st.subheader("📈 Intelligence Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        df["Location"].value_counts().head(10).plot(kind="bar", ax=ax)
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        df["Crime_Type"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
        ax.set_ylabel("")
        st.pyplot(fig)

st.sidebar.markdown("---")
st.sidebar.write("🚔 CrimeLens Enterprise System")