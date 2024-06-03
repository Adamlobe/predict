# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from math import sqrt
import fbprophet

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
# Preparar os dados para o Prophet
df_prophet = df[['LOG_TOTAL_POSTAGEM']].reset_index()
df_prophet.columns = ['ds', 'y']

# %%
# Treinar o modelo Prophet
model = Prophet()
model.fit(df_prophet)

# %%
# Fazer previsões para os próximos 12 meses
future = model.make_future_dataframe(periods=12, freq='M')
forecast = model.predict(future)

# %%
# Plotar os resultados
fig = model.plot(forecast)
plt.title('Previsão com Prophet')
plt.show()

# %%
# Revertendo a transformação logarítmica
forecast['yhat_exp'] = np.exp(forecast['yhat'])
forecast['yhat_lower_exp'] = np.exp(forecast['yhat_lower'])
forecast['yhat_upper_exp'] = np.exp(forecast['yhat_upper'])

# %%
# Plot dos valores reais e preditos
plt.figure(figsize=(10,6))
plt.plot(df.index, df["TOTAL_POSTAGEM"], label='Valores Reais', color='blue')
plt.plot(forecast['ds'], forecast['yhat_exp'], label='Previsão', color='green')
plt.fill_between(forecast['ds'], forecast['yhat_lower_exp'], forecast['yhat_upper_exp'], color='green', alpha=0.3)
plt.legend()
plt.title('Previsão dos Próximos 12 Meses com Prophet')
plt.show()

# %%
# Cálculo do RMSE para o conjunto de teste (últimos 12 meses do treino)
y_test = df['TOTAL_POSTAGEM'][-12:]
y_pred = forecast['yhat_exp'][-24:-12].values  # Previsões correspondentes ao período de teste
rmse_prophet = sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse_prophet}')

# %%
# Criando um DataFrame para as previsões futuras
forecast_df = forecast[['ds', 'yhat_exp', 'yhat_lower_exp', 'yhat_upper_exp']].set_index('ds')
forecast_df.columns = ['forecast_medio', 'intervalo_abaixo_f', 'intervalo_acima_f']

# Exibindo o DataFrame com as previsões futuras
print(forecast_df.tail(12))
