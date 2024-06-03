# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from math import sqrt
from xgboost import XGBRegressor

# %%
df = pd.read_parquet('data_bronze/data_filtred.parquet')

# %%
df = df.sort_values(by='DATA_ACAO', ascending=True)

# %%
df.set_index('DATA_ACAO', inplace=True)
df = df.resample('M').sum()

# %%
df['TOTAL_POSTAGEM'].plot(figsize=(8,3))
plt.title('Série Temporal - TOTAL_POSTAGEM')
plt.show()

# %%
decomposition = seasonal_decompose(df['TOTAL_POSTAGEM'], model='additive', period=12)
decomposition.plot()
plt.show()

# %% [markdown]
# TESTE DE ESTACIONARIEDADE

result = adfuller(df['TOTAL_POSTAGEM'].dropna())
print(f'Teste ADF: {result[0]}')
print(f'p-valor: {result[1]}')

# %%
df['LOG_TOTAL_POSTAGEM'] = np.log(df['TOTAL_POSTAGEM'])
df['DIFF_TOTAL_POSTAGEM'] = df['TOTAL_POSTAGEM'].diff()

# %%
# Preparação dos dados para XGBoost

# Criando características de lag
def create_lags(df, n_lags=12):
    for lag in range(1, n_lags + 1):
        df[f'lag_{lag}'] = df['LOG_TOTAL_POSTAGEM'].shift(lag)
    return df

df = create_lags(df)
df = df.dropna()

# Separando as features (X) e o target (y)
X = df.drop(['TOTAL_POSTAGEM', 'LOG_TOTAL_POSTAGEM', 'DIFF_TOTAL_POSTAGEM'], axis=1)
y = df['LOG_TOTAL_POSTAGEM']

# Dividindo em treino e teste
train_size = int(len(df) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# %%
# Treinando o modelo XGBoost
model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
model.fit(X_train, y_train)

# %%
# Fazendo previsões no conjunto de teste
y_pred_log = model.predict(X_test)

# Revertendo a transformação logarítmica
y_test_exp = np.exp(y_test)
y_pred_exp = np.exp(y_pred_log)

# %%
# Plotando os resultados
plt.figure(figsize=(10,6))
plt.plot(y_test_exp.index, y_test_exp, label='Valores Reais', color='blue')
plt.plot(y_test_exp.index, y_pred_exp, label='Valores Preditos', color='red')
plt.legend()
plt.title('Valores Reais e Preditos (XGBoost)')
plt.show()

# %%
# Cálculo do RMSE para o conjunto de teste
rmse_xgb = sqrt(mean_squared_error(y_test_exp, y_pred_exp))
print(f'RMSE: {rmse_xgb}')

# %%
# Previsão dos próximos 12 meses usando o último ponto de treino como base
def forecast_future(model, df, n_periods=12):
    future_preds = []
    current_input = X.iloc[-1:].copy()

    for _ in range(n_periods):
        pred_log = model.predict(current_input)[0]
        future_preds.append(pred_log)

        current_input = current_input.shift(-1, axis=1)
        current_input.iloc[:, -1] = pred_log

    return np.exp(future_preds)

future_forecast = forecast_future(model, df, 12)

# %%
# Plot das previsões futuras
future_dates = pd.date_range(start=df.index[-1], periods=12, freq='M')
plt.figure(figsize=(10,6))
plt.plot(df.index, np.exp(df['LOG_TOTAL_POSTAGEM']), label='Valores Reais', color='blue')
plt.plot(future_dates, future_forecast, label='Previsão Futura', color='green')
plt.legend()
plt.title('Previsão dos Próximos 12 Meses (XGBoost)')
plt.show()

# %%
# Criando um DataFrame para as previsões futuras
future_forecast_df = pd.DataFrame({
    'forecast_medio': future_forecast
}, index=future_dates)

# Exibindo o DataFrame com as previsões futuras
print(future_forecast_df)
