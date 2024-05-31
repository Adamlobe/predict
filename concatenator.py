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
    df = df.resample('W').sum()
    df.reset_index(inplace=True)
    df.set_index('DATA_ACAO', inplace=True)
    df.columns = ['DATA_ACAO', 'TOTAL_POSTAGEM']
    data_filtred = pd.concat([data_filtred, df], axis=0, ignore_index=True)

#%%
data_filtred = data_filtred.drop_duplicates()

# %%
data_filtred.to_excel('data_filtred.xlsx')

#%%
data_filtred.to_parquet('data_filtred.parquet')