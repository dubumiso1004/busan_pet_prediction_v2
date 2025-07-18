import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import haversine_distances
from math import radians
from streamlit_folium import st_folium
import folium

st.set_page_config(page_title="PET 예측 시스템", layout="wide")
st.title("🌡️ PET 예측 시스템 (실측 기반 + 활동 권장도 표시)")

@st.cache_data
def load_data():
    df = pd.read_csv("measured_data.csv")
    def dms_to_dd(dms): parts = list(map(float, dms.split(";"))); return parts[0] + parts[1]/60 + parts[2]/3600
    df["lat_dd"] = df["Lat"].apply(dms_to_dd)
    df["lon_dd"] = df["Lon"].apply(dms_to_dd)
    return df

df = load_data()
model = joblib.load("pet_rf_model_trained.pkl")

# 지도 표시
m = folium.Map(location=[35.2313, 129.0805], zoom_start=16)
folium.Marker([35.2313, 129.0805], popup="부산대학교").add_to(m)
clicked = st_folium(m, width=700, height=500)

if clicked and clicked.get("last_clicked"):
    lat = clicked["last_clicked"]["lat"]
    lon = clicked["last_clicked"]["lng"]
    st.success(f"선택한 위치: {lat:.5f}, {lon:.5f}")

    clicked_point = np.array([[radians(lat), radians(lon)]])
    points = df[["lat_dd", "lon_dd"]].applymap(radians).values
    distances = haversine_distances(clicked_point, points)[0] * 6371
    nearest = df.iloc[distances.argmin()]

    svf = st.slider("SVF", 0.0, 1.0, float(nearest["SVF"]), 0.01)
    gvi = st.slider("GVI", 0.0, 1.0, float(nearest["GVI"]), 0.01)
    bvi = st.slider("BVI", 0.0, 1.0, float(nearest["BVI"]), 0.01)

    temp = st.number_input("기온 (°C)", value=27.0)
    hum = st.number_input("습도 (%)", value=60.0)
    wind = st.number_input("풍속 (m/s)", value=1.5)

    if temp and hum and wind:
        input_data = [[svf, gvi, bvi, temp, hum, wind]]
        pet = model.predict(input_data)[0]
        st.success(f"🌡️ 예측된 PET: {pet:.2f}°C")

        # 활동 권장 메시지
        if pet <= 18:
            status = "⚠️ 쌀쌀함: 겉옷을 챙기세요."
        elif pet <= 23:
            status = "🟢 쾌적함: 야외 활동하기 좋습니다!"
        elif pet <= 29:
            status = "🟡 따뜻함: 활동 가능하지만 약간 더울 수 있어요."
        elif pet <= 35:
            status = "🟠 더움: 격한 활동은 피하는 것이 좋아요."
        else:
            status = "🔴 매우 더움: 실외 활동은 피하고 휴식을 권장합니다."

        st.info(f"👤 체감 활동 권장도: {status}")
else:
    st.info("지도를 클릭하여 위치를 선택해 주세요.")
