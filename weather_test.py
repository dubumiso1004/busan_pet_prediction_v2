import requests
from datetime import datetime, timedelta
import math

# 위도/경도 → 기상청 격자 좌표 변환
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
    sf = (math.tan(math.pi * 0.25 + slat1 * 0.5) ** sn * math.cos(slat1)) / sn
    ro = re * sf / (math.tan(math.pi * 0.25 + olat * 0.5) ** sn)

    ra = re * sf / (math.tan(math.pi * 0.25 + lat * DEGRAD * 0.5) ** sn)
    theta = lon * DEGRAD - olon
    theta = (theta + math.pi) % (2 * math.pi) - math.pi
    theta *= sn

    x = ra * math.sin(theta) + XO + 0.5
    y = ro - ra * math.cos(theta) + YO + 0.5
    return int(x), int(y)

# 사용자가 선택한 시간
selected_dt = datetime(2025, 7, 13, 15, 0)

# 기상청 API 키
api_key = "AXlyYXdXs4EpFcTq0KlNv0lz3KmcQwQSNtgRAWui8TTwin709Ki5DMkQ5tfDup1t79CZKhtaJOKPYw6VFxot2A=="

# 3시간 단위 기준시로 변환
base_dt = selected_dt - timedelta(hours=selected_dt.hour % 3)
base_date = base_dt.strftime("%Y%m%d")
base_time = base_dt.strftime("%H%M")
forecast_time = selected_dt.strftime("%Y%m%d%H%M")

# 위경도 → nx, ny
lat, lon = 35.23133, 129.07868
nx, ny = convert_to_grid(lat, lon)

# API 호출 URL
url = (
    f"http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst?"
    f"serviceKey={api_key}&numOfRows=1000&pageNo=1&dataType=JSON"
    f"&base_date={base_date}&base_time={base_time}&nx={nx}&ny={ny}"
)

# 호출 및 결과 추출
response = requests.get(url)
data = response.json()
items = data['response']['body']['items']['item']

# 해당 시간의 TMP, REH, WSD 추출
target_values = {}
for item in items:
    fcst_dt = item['fcstDate'] + item['fcstTime']
    if fcst_dt == forecast_time:
        if item['category'] == 'TMP':
            target_values['temp'] = float(item['fcstValue'])
        elif item['category'] == 'REH':
            target_values['humi'] = float(item['fcstValue'])
        elif item['category'] == 'WSD':
            target_values['wind'] = float(item['fcstValue'])

print("✅ 예측 시각:", selected_dt.strftime("%Y-%m-%d %H:%M"))
print("📡 예보 기온(°C):", target_values.get('temp'))
print("💧 예보 습도(%):", target_values.get('humi'))
print("🍃 예보 풍속(m/s):", target_values.get('wind'))
