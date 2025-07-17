import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from sklearn.ensemble import RandomForestRegressor
import joblib
from geopy.distance import geodesic

# 실측 데이터 로드
df = pd.read_csv("measured_data.csv")  # SVF, GVI, BVI, lat, lon 포함

# 모델 로드
model = joblib.load("pet_rf_model_trained.pkl")

st.title("🌡️ PET 예측 시스템 (부산대학교 대상지)")

st.subheader("📍 지도에서 위치를 선택하세요")
st.write("지도를 클릭하면 해당 위치의 시각환경 값을 자동으로 불러옵니다.")

# 기본 지도 위치
default_location = [35.2320, 129.0845]  # 부산대학교 중심 예시

# 지도 표시
clicked_location = st.map(center={"lat": default_location[0], "lon": default_location[1]}, zoom=16)

# 위도, 경도 추출
if clicked_location is not None and "latitude" in clicked_location:
    lat = clicked_location["latitude"]
    lon = clicked_location["longitude"]
    st.success(f"선택된 위치: {lat:.5f}, {lon:.5f}")

    # 📌 가장 가까운 실측 지점 찾기
    def get_nearest_row(lat, lon, df):
        df["distance"] = df.apply(lambda row: geodesic((lat, lon), (row["lat"], row["lon"])).meters, axis=1)
        return df.loc[df["distance"].idxmin()]

    nearest = get_nearest_row(lat, lon, df)
    svf = nearest["SVF"]
    gvi = nearest["GVI"]
    bvi = nearest["BVI"]

    st.write(f"🔹 자동 불러온 시각환경 값: SVF={svf:.2f}, GVI={gvi:.2f}, BVI={bvi:.2f}")

    # 기상 요소 (예시 평균값 또는 실시간 API 사용 가능)
    temp, rh, wind = 29.7, 73, 1.2  # 8월 평균
    input_features = np.array([[svf, gvi, bvi, temp, rh, wind]])
    predicted_pet = model.predict(input_features)[0]

    # 결과 출력
    st.subheader("📊 예측 결과")
    st.info(f"8월 평균 기상조건: {temp}°C / {rh}% / {wind} m/s")
    st.success(f"예측된 PET: {predicted_pet:.2f}°C")

    # 스마트워치 메시지
    st.subheader("⌚ 스마트워치 알림용 메시지")
    st.code(f"📍 Busan Univ.\n🌡️ PET: {predicted_pet:.2f}°C\n🟢 체감 쾌적 수준: 자동 메시지")

else:
    st.warning("지도를 클릭하면 결과가 표시됩니다.")
