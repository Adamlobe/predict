#%%
import pandas as pd
import zipfile
import os

# %%
diretorio_de_extracao = 'data_raw/'

# %%
arquivos_extraidos = os.listdir(diretorio_de_extracao)
arquivos_extraidos.sort()
arquivos_extraidos

#%%
data_filtred = pd.DataFrame(columns=['DATA_ACAO', 'TOTAL_POSTAGEM'])

for arquivo in arquivos_extraidos:
    caminho_arquivo = os.path.join(diretorio_de_extracao, arquivo)
    df = pd.read_excel(caminho_arquivo)
    print(arquivo)
    df = df[df['ACAO'] == 'SUBIDA_DOCUMENTO']
    df = df['DATA_ACAO'].value_counts().reset_index()
    df.columns = ['DATA_ACAO', 'TOTAL_POSTAGEM']
    data_filtred = pd.concat([data_filtred, df], axis=0, ignore_index=True)

#%%
data_filtred = data_filtred.drop_duplicates()
data_filtred['DATA_ACAO'] = pd.to_datetime(data_filtred['DATA_ACAO'], errors='coerce')


# %%
data_filtred.to_excel('data_bronze/data_filtred.xlsx')

#%%
data_filtred.to_parquet('data_bronze/data_filtred.parquet')
# %%
