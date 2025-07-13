import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium
import math

# 모델과 데이터 불러오기
model = joblib.load("model/pet_rf_model_trained.pkl")
df = pd.read_excel("data/total_svf_gvi_bvi_250618.xlsx", sheet_name="gps 포함")

# 위경도 변환
def dms_to_decimal(dms_str):
    parts = list(map(float, dms_str.split(";")))
    return parts[0] + parts[1] / 60 + parts[2] / 3600

df["lat_decimal"] = df["lat"].apply(dms_to_decimal)
df["lon_decimal"] = df["lon"].apply(dms_to_decimal)

# 월별 평균 기상 데이터 (2024년 예시)
monthly_weather = {
    6: {"temp": 25.4, "humi": 70, "wind": 1.5},
    7: {"temp": 28.3, "humi": 75, "wind": 1.3},
    8: {"temp": 29.7, "humi": 73, "wind": 1.2},
    9: {"temp": 26.1, "humi": 68, "wind": 1.6}
}

# Streamlit 설정
st.set_page_config(layout="centered")
st.title("🌞 여름철(6~9월) PET 예측 시뮬레이션")
st.markdown("지도에서 위치를 클릭하고, SVF / GVI / BVI를 조절하면 해당 월의 PET을 예측합니다.")

# 월 선택
selected_month = st.selectbox("예측할 월 선택", [6, 7, 8, 9], index=2)

# 지도 생성
m = folium.Map(location=[df["lat_decimal"].mean(), df["lon_decimal"].mean()], zoom_start=17)
map_data = st_folium(m, width=700, height=500)

# 지도 클릭 이벤트
if map_data and map_data.get("last_clicked"):
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]

    # 가장 가까운 지점 찾기
    df["distance"] = ((df["lat_decimal"] - lat)**2 + (df["lon_decimal"] - lon)**2)
    nearest = df.loc[df["distance"].idxmin()]

    # 실측값 가져오기
    temp = nearest["AirTemperature"]
    humi = nearest["Humidity"]
    wind = nearest["WindSpeed"]

    # 시각환경 기본값
    default_svf = float(nearest["SVF"])
    default_gvi = float(nearest["GVI"])
    default_bvi = float(nearest["BVI"])

    # 사용자 조절
    st.markdown("### 🌿 시각환경 조정")
    svf = st.slider("SVF", 0.0, 1.0, default_svf, step=0.01)
    gvi = st.slider("GVI", 0.0, 1.0, default_gvi, step=0.01)
    bvi = st.slider("BVI", 0.0, 1.0, default_bvi, step=0.01)

    # 선택한 월의 평균 기상 적용
    weather = monthly_weather[selected_month]
    temp = weather["temp"]
    humi = weather["humi"]
    wind = weather["wind"]

    # 예측
    X = [[svf, gvi, bvi, temp, humi, wind]]
    pet_pred = model.predict(X)[0]

    # 결과 출력
    st.markdown("### 🔍 예측 결과")
    st.info(f"📅 선택한 월: {selected_month}월")
    st.info(f"🌡️ 기상 조건 → 기온: {temp}°C, 습도: {humi}%, 풍속: {wind} m/s")
    st.success(f"🔮 예측 PET: {pet_pred:.2f} °C")

    # 지도 재렌더링
    st.markdown("----")
    st.markdown("🗺️ 선택한 위치 확인")
    st_folium(m, width=700, height=500)

else:
    st.warning("🖱 지도를 클릭하면 예측이 시작됩니다.")
