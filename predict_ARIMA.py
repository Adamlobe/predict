# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from math import sqrt

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
df['LOG_TOTAL_POSTAGEM'].plot(figsize=(8,3))
plt.title('Série Temporal - TOTAL_POSTAGEM')
plt.show()

#%%
df['DIFF_TOTAL_POSTAGEM'].plot(figsize=(8,3))
plt.title('Série Temporal - TOTAL_POSTAGEM')
plt.show()

# %%
fit_arima = auto_arima(
    df["LOG_TOTAL_POSTAGEM"].dropna(),
    start_p=1, 
    start_q=1, 
    max_p=6, 
    max_q=6,
    m=12,
    start_P=0,
    seasonal=True,
    d=1,
    D=1,
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=False
)

print(fit_arima.summary())

# %%
print(fit_arima.order, fit_arima.seasonal_order)

# %%
model = SARIMAX(df["LOG_TOTAL_POSTAGEM"].dropna(), 
                order=fit_arima.order, 
                seasonal_order=fit_arima.seasonal_order)

resultado_sarimax = model.fit()

# %%
resultado_sarimax.summary()

# %%
# Backtesting - Predição dos últimos 12 meses
predicoes = resultado_sarimax.get_prediction(start=-12)
predicao_media_log = predicoes.predicted_mean

# Intervalo de confiança
intervalo_confianca_log = predicoes.conf_int()
limites_abaixo_log = intervalo_confianca_log.iloc[:,0]
limites_acima_log = intervalo_confianca_log.iloc[:,1]

# Revertendo a transformação logarítmica
predicao_media = np.exp(predicao_media_log)
limites_abaixo = np.exp(limites_abaixo_log)
limites_acima = np.exp(limites_acima_log)

# Plot dos valores reais e preditos
plt.figure(figsize=(10,6))
plt.plot(df.index[-24:], np.exp(df["LOG_TOTAL_POSTAGEM"][-24:]), label='Valores Reais', color='blue')
plt.plot(predicao_media.index, predicao_media, label='Valores Preditos (Backtesting)', color='red')
plt.fill_between(predicao_media.index, limites_abaixo, limites_acima, color='red', alpha=0.3)
plt.legend()
plt.title('Valores Reais e Preditos (Backtesting)')
plt.show()

# %%
# Cálculo do RMSE para Backtesting
rmse_sarima = sqrt(mean_squared_error(np.exp(df["LOG_TOTAL_POSTAGEM"][-12:]).dropna().values, predicao_media.values))
print(f'RMSE: {rmse_sarima}')

# %%
# Previsão dos próximos 12 meses
forecast = resultado_sarimax.get_forecast(steps=12)
forecast_medio_log = forecast.predicted_mean

# Intervalo de confiança
intervalo_confianca_forecast_log = forecast.conf_int()
intervalo_abaixo_f_log = intervalo_confianca_forecast_log.iloc[:,0]
intervalo_acima_f_log = intervalo_confianca_forecast_log.iloc[:,1]

# Revertendo a transformação logarítmica
forecast_medio = np.exp(forecast_medio_log)
intervalo_abaixo_f = np.exp(intervalo_abaixo_f_log)
intervalo_acima_f = np.exp(intervalo_acima_f_log)

# Plot das previsões futuras
plt.figure(figsize=(10,6))
plt.plot(df.index, np.exp(df["LOG_TOTAL_POSTAGEM"]), label='Valores Reais', color='blue')
plt.plot(forecast_medio.index, forecast_medio, label='Previsão Futura', color='green')
plt.fill_between(forecast_medio.index, intervalo_abaixo_f, intervalo_acima_f, color='green', alpha=0.3)
plt.legend()
plt.title('Previsão dos Próximos 12 Meses')
plt.show()

# %%
# Criando um DataFrame para as previsões
forecast_df = pd.DataFrame({
    'intervalo_abaixo_f': intervalo_abaixo_f,
    'forecast_medio': forecast_medio,
    'intervalo_acima_f': intervalo_acima_f
}, index=forecast_medio.index)

forecast_df