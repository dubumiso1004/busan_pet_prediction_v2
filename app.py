import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium
import math

# 모델과 데이터 불러오기
model = joblib.load("model/pet_rf_model_trained.pkl")
df = pd.read_excel("data/total_svf_gvi_bvi_250618.xlsx", sheet_name="gps 포함")

# 위경도 변환 함수
def dms_to_decimal(dms_str):
    parts = list(map(float, dms_str.split(";")))
    return parts[0] + parts[1] / 60 + parts[2] / 3600

df["lat_decimal"] = df["lat"].apply(dms_to_decimal)
df["lon_decimal"] = df["lon"].apply(dms_to_decimal)

# 월별 평균 기상값 (예: 2024년 평균값)
monthly_weather = {
    6: {"temp": 25.4, "humi": 70, "wind": 1.5},
    7: {"temp": 28.3, "humi": 75, "wind": 1.3},
    8: {"temp": 29.7, "humi": 73, "wind": 1.2},
    9: {"temp": 26.1, "humi": 68, "wind": 1.6}
}

# Streamlit 인터페이스
st.set_page_config(layout="centered")
st.title("📍 클릭 위치 기반 PET 예측 시스템")
st.markdown("지도에서 위치를 클릭하면 측정된 기온/습도/풍속이 자동 적용되며, SVF/GVI/BVI를 조절하여 월별 PET을 예측합니다.")

# 사용자 월 선택
selected_month = st.selectbox("예측할 월을 선택하세요", [6, 7, 8, 9], index=2)

# 지도 생성
m = folium.Map(location=[df["lat_decimal"].mean(), df["lon_decimal"].mean()], zoom_start=17)
map_data = st_folium(m, width=700, height=500)

# 지도 클릭 시 동작
if map_data and map_data.get("last_clicked"):
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]

    # 최근접 측정지점 탐색
    df["distance"] = ((df["lat_decimal"] - lat) ** 2 + (df["lon_decimal"] - lon) ** 2)
    nearest = df.loc[df["distance"].idxmin()]
    nearest_name = nearest["Location_Name"]

    # 해당 지점 실측값 불러오기
    temp = nearest["AirTemperature"]
    humi = nearest["Humidity"]
    wind = nearest["WindSpeed"]
    default_svf = nearest["SVF"]
    default_gvi = nearest["GVI"]
    default_bvi = nearest["BVI"]

    # 시각환경 조정 슬라이더
    st.markdown(f"### 🏷️ 선택된 지점: **{nearest_name}**")
    svf = st.slider("SVF", 0.0, 1.0, float(default_svf), 0.01)
    gvi = st.slider("GVI", 0.0, 1.0, float(default_gvi), 0.01)
    bvi = st.slider("BVI", 0.0, 1.0, float(default_bvi), 0.01)

    # 월별 평균 기상데이터 적용
    weather = monthly_weather[selected_month]
    temp = weather["temp"]
    humi = weather["humi"]
    wind = weather["wind"]

    # PET 예측
    X = [[svf, gvi, bvi, temp, humi, wind]]
    pet_pred = model.predict(X)[0]

    # 결과 출력
    st.markdown("### 📊 예측 결과")
    st.info(f"🗓️ {selected_month}월 평균 기상: {temp}°C / {humi}% / {wind} m/s")
    st.success(f"🌡️ 예측된 PET: {pet_pred:.2f} °C")
else:
    st.warning("🖱 지도를 클릭하면 예측이 시작됩니다.")
