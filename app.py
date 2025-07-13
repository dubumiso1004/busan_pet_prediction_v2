import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import requests
import math

# 기상청 API key
api_key = "AXlyYXdXs4EpFcTq0KlNv0lz3KmcQwQSNtgRAWui8TTwin709Ki5DMkQ5tfDup1t79CZKhtaJOKPYw6VFxot2A%3D%3D"

# 모델 불러오기
model = joblib.load("model/pet_rf_model_trained.pkl")
df = pd.read_excel("data/total_svf_gvi_bvi_250618.xlsx", sheet_name="gps 포함")

# 위경도 변환 함수
def dms_to_decimal(dms_str):
    parts = list(map(float, dms_str.split(";")))
    return parts[0] + parts[1] / 60 + parts[2] / 3600

df["lat_decimal"] = df["lat"].apply(dms_to_decimal)
df["lon_decimal"] = df["lon"].apply(dms_to_decimal)

# 위경도 → 기상청 격자
def convert_to_grid(lat, lon):
    RE = 6371.00877
    GRID = 5.0
    SLAT1, SLAT2, OLON, OLAT = 30.0, 60.0, 126.0, 38.0
    XO, YO = 43, 136
    DEGRAD = math.pi / 180.0
    re = RE / GRID
    slat1 = SLAT1 * DEGRAD
    slat2 = SLAT2 * DEGRAD
    olon = OLON * DEGRAD
    olat = OLAT * DEGRAD
    sn = math.tan(math.pi * 0.25 + slat2 * 0.5) / math.tan(math.pi * 0.25 + slat1 * 0.5)
    sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(sn)
    sf = (math.tan(math.pi * 0.25 + slat1 * 0.5)**sn * math.cos(slat1)) / sn
    ro = re * sf / (math.tan(math.pi * 0.25 + olat * 0.5)**sn)
    ra = re * sf / (math.tan(math.pi * 0.25 + lat * DEGRAD * 0.5)**sn)
    theta = lon * DEGRAD - olon
    theta = (theta + math.pi) % (2 * math.pi) - math.pi
    theta *= sn
    x = ra * math.sin(theta) + XO + 0.5
    y = ro - ra * math.cos(theta) + YO + 0.5
    return int(x), int(y)

# 기상청 API 호출 함수
def get_weather_from_kma(api_key, lat, lon, target_dt):
    base_dt = target_dt - timedelta(hours=target_dt.hour % 3)
    base_date = base_dt.strftime("%Y%m%d")
    base_time = base_dt.strftime("%H%M")
    forecast_time = target_dt.strftime("%Y%m%d%H%M")
    nx, ny = convert_to_grid(lat, lon)
    url = (
        f"http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst?"
        f"serviceKey={api_key}&numOfRows=1000&pageNo=1&dataType=JSON"
        f"&base_date={base_date}&base_time={base_time}&nx={nx}&ny={ny}"
    )

    try:
        response = requests.get(url)
        data = response.json()
        items = data["response"]["body"]["items"]["item"]
    except (KeyError, ValueError):
        return {}

    weather = {}
    for item in items:
        fcst_dt = item["fcstDate"] + item["fcstTime"]
        if fcst_dt == forecast_time:
            if item["category"] == "TMP":
                weather["temp"] = float(item["fcstValue"])
            elif item["category"] == "REH":
                weather["humi"] = float(item["fcstValue"])
            elif item["category"] == "WSD":
                weather["wind"] = float(item["fcstValue"])
    return weather

# Streamlit 시작
st.title("📍 클릭 위치 기반 PET 예측 시스템")
st.markdown("SVF/GVI/BVI를 조절하고, 현재와 예보 PET를 비교해보세요.")

# 날짜/시간 선택
col1, col2 = st.columns(2)
with col1:
    date_input = st.date_input("예보 기준 날짜", datetime.now().date())
with col2:
    time_input = st.time_input("예보 기준 시간", datetime.now().time())
selected_dt = datetime.combine(date_input, time_input)

# 지도 표시
m = folium.Map(location=[df["lat_decimal"].mean(), df["lon_decimal"].mean()], zoom_start=17)
map_data = st_folium(m, width=700, height=500)

# 지도 클릭 시 처리
if map_data and map_data.get("last_clicked"):
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]

    folium.Marker(
        location=[lat, lon],
        tooltip="선택 위치",
        icon=folium.Icon(color="red", icon="map-marker")
    ).add_to(m)

    # 최근접 측정지점에서 SVF/GVI/BVI 가져오기
    df["distance"] = ((df["lat_decimal"] - lat)**2 + (df["lon_decimal"] - lon)**2)
    nearest = df.loc[df["distance"].idxmin()]
    default_svf = float(nearest["SVF"])
    default_gvi = float(nearest["GVI"])
    default_bvi = float(nearest["BVI"])

    # 슬라이더 입력
    svf = st.slider("SVF", 0.0, 1.0, default_svf, 0.01)
    gvi = st.slider("GVI", 0.0, 1.0, default_gvi, 0.01)
    bvi = st.slider("BVI", 0.0, 1.0, default_bvi, 0.01)

    st.markdown("### ✅ 실측 기반 PET 예측")
    now_temp, now_humi, now_wind = 28.0, 50.0, 1.0
    X_now = [[svf, gvi, bvi, now_temp, now_humi, now_wind]]
    pet_now = model.predict(X_now)[0]
    st.success(f"🌡 현재 기준 PET: {pet_now:.2f} °C")

    st.markdown("### 🔮 예보 기반 PET 예측")
    with st.spinner("기상청 예보 데이터를 불러오는 중..."):
        weather = get_weather_from_kma(api_key, lat, lon, selected_dt)

    if "temp" in weather and "humi" in weather and "wind" in weather:
        st.info(f"📡 예보: {weather['temp']}°C / {weather['humi']}% / {weather['wind']} m/s")
        X_forecast = [[svf, gvi, bvi, weather["temp"], weather["humi"], weather["wind"]]]
        pet_forecast = model.predict(X_forecast)[0]
        st.success(f"📅 {selected_dt.strftime('%Y-%m-%d %H:%M')} 기준 PET: {pet_forecast:.2f} °C")
    else:
        st.warning("⚠ 선택한 시각에 기상 데이터가 없습니다. 다른 날짜/시간을 선택해보세요.")

    st_folium(m, width=700, height=500)

else:
    st.warning("🖱 지도를 클릭하면 위치가 선택됩니다.")
