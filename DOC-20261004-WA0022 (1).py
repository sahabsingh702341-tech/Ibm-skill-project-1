import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from pathlib import Path

DATA_PATH = Path('data/house_price.csv')
MODEL_PATH = Path('model/model.pkl')

df = pd.read_csv(DATA_PATH)
X = df[['area', 'rooms']]
y = df['price']

model = LinearRegression()
model.fit(X, y)

MODEL_PATH.parent.mkdir(exist_ok=True)
joblib.dump(model, MODEL_PATH)
print('Model trained successfully.')
print(f'Model saved to: {MODEL_PATH}')
