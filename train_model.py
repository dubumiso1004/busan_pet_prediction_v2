# train_model.py
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# 엑셀 데이터 불러오기
df = pd.read_excel('data/total_svf_gvi_bvi_250618.xlsx', sheet_name='gps 포함')

# DMS → decimal 변환 함수
def dms_to_decimal(dms_str):
    parts = list(map(float, dms_str.split(';')))
    return parts[0] + parts[1] / 60 + parts[2] / 3600

df['lat_decimal'] = df['lat'].apply(dms_to_decimal)
df['lon_decimal'] = df['lon'].apply(dms_to_decimal)

# 학습용 데이터셋 구성
X = df[['SVF', 'GVI', 'BVI', 'AirTemperature', 'Humidity', 'WindSpeed']]
y = df['PET']

# 모델 학습
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 모델 저장
joblib.dump(model, 'model/pet_rf_model_trained.pkl')
print("✅ 모델 다시 저장 완료")
