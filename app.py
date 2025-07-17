import streamlit as st
import numpy as np
import joblib

# 모델 불러오기
model = joblib.load("model/pet_rf_model_trained.pkl")  # 모델 경로에 맞게 조정하세요

# 페이지 제목
st.title("🌡️ PET 예측 시스템 (부산대학교 대상지)")
st.write("지점 선택 또는 시각환경 조정 후 PET 예측 결과와 스마트워치 알림 메시지를 확인하세요.")

# 사용자 입력 (예: SVF, GVI, BVI)
svf = st.slider("SVF (하늘 가시성)", 0.0, 1.0, 0.87)
gvi = st.slider("GVI (녹지 시야율)", 0.0, 1.0, 0.85)
bvi = st.slider("BVI (건물 시야율)", 0.0, 1.0, 0.31)

# 고정 기상조건 예시 (8월 평균)
temp = 29.7  # °C
humidity = 73  # %
wind = 1.2  # m/s

# 모델 입력 및 예측
input_data = np.array([[svf, gvi, bvi, temp, humidity, wind]])
predicted_pet = model.predict(input_data)[0]

# 예측 결과 출력
st.subheader("📊 예측 결과")
weather_info = f"{temp}°C / {humidity}% / {wind} m/s"
st.info(f"📅 8월 평균 기상조건: {weather_info}")
st.success(f"🌡️ 예측된 PET: {predicted_pet:.2f}°C")

# 선택된 지점 이름 (예시)
selected_location = "Busan Univ.19"

# 스마트워치용 요약 메시지 생성
summary = f"""📍 {selected_location}
🌡️ 예측 PET: {predicted_pet:.1f}°C
📅 8월 평균 기상: {weather_info}
⚠️ 체감: 매우 더움 (주의 필요)"""

# 스마트워치 알림용 메시지 출력
st.markdown("### 📲 스마트워치 알림용 메시지")
st.text_area("아래 내용을 복사해서 워치 알림 앱에 붙여넣으세요:", summary, height=120)

# 다운로드 버튼으로 저장도 가능하게
st.download_button("📥 메시지 저장 (텍스트 파일)", summary, file_name="pet_message.txt")

