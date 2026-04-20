# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 2025

@author: luisa
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from scipy import stats
import joblib

df = pd.read_excel('Base Teste 7.xlsx')
#df.columns.tolist()
df.dropna(inplace=True)

df['FAT_z'] = stats.zscore(df['FAT'])
df = df[np.abs(df['FAT_z']) <= 2].drop(columns=['FAT_z'])

df['Share_z'] = stats.zscore(df['Share'])
df = df[np.abs(df['Share_z']) <= 2].drop(columns=['Share_z'])

df = pd.get_dummies(df, columns=df.select_dtypes(include='object').columns, drop_first=True)
df.dropna(inplace=True)

df_share = df.drop(columns=['FAT'])
train_share, test_share = train_test_split(df_share, test_size=0.2, random_state=0)

scaler_share = StandardScaler()
x_train_share = train_share.drop(columns=['Share'])
y_train_share = train_share['Share']
x_test_share = test_share.drop(columns=['Share'])
y_test_share = test_share['Share']

x_train_share_scaled = scaler_share.fit_transform(x_train_share)
x_test_share_scaled = scaler_share.transform(x_test_share)


best_mlp_model_share = MLPRegressor(
    activation='tanh',
    alpha=0.001,
    hidden_layer_sizes=(128, 64, 32),
    learning_rate='adaptive',
    learning_rate_init=0.01,
    max_iter=3000,
    solver='sgd',
    random_state=42,
    early_stopping=True
)

best_mlp_model_share.fit(x_train_share_scaled, y_train_share)

#MSE Share: 0.005816340885019058
#MSE FAT: 0.04744855400862348


y_pred_share = best_mlp_model_share.predict(x_test_share_scaled)
print("MSE Share:", mean_squared_error(y_test_share, y_pred_share))

df_fat = df.drop(columns=['FAT', 'Sobra_demanda'])
alvo_fat = df[['FAT']]

scaler_df = StandardScaler()
df_scaled = scaler_df.fit_transform(df_fat)

scaler_alvo = StandardScaler()
alvo_scaled = scaler_alvo.fit_transform(alvo_fat)

train_fat, test_fat = train_test_split(df_scaled, test_size=0.2, random_state=0)
train_target, test_target = train_test_split(alvo_scaled, test_size=0.2, random_state=0)

best_mlp_model_fat = MLPRegressor(
    activation='tanh',
    alpha=1e-06,
    hidden_layer_sizes=(128, 64),
    learning_rate='adaptive',
    learning_rate_init=0.01,
    max_iter=3000,
    solver='adam',
    random_state=42,
    early_stopping=True
)
best_mlp_model_fat.fit(train_fat, train_target.ravel())

y_pred_fat_scaled = best_mlp_model_fat.predict(test_fat).reshape(-1, 1)
y_pred_fat = scaler_alvo.inverse_transform(y_pred_fat_scaled)
print("MSE FAT:", mean_squared_error(test_target, y_pred_fat_scaled))

joblib.dump(best_mlp_model_share, 'best_mlp_model_share.pkl')
joblib.dump(best_mlp_model_fat, 'best_mlp_model_fat.pkl')
joblib.dump(scaler_share, 'scaler_share.pkl')
joblib.dump(scaler_df, 'scaler_df.pkl')
joblib.dump(scaler_alvo, 'scaler_alvo.pkl')
joblib.dump(df.columns.tolist(), 'colunas.pkl')

colunas_fat = df_fat.columns.tolist()

#%%
# ====================== ENTRADA MANUAL DE DADOS ======================
#novos_dados_dict = {}
#for col in x_train_share.columns:
#    valor = float(input(f"Digite o valor para {col}: "))
#    novos_dados_dict[col] = valor
#
#novos_dados_df = pd.DataFrame([novos_dados_dict])
#novos_dados_scaled = scaler_share.transform(novos_dados_df)

# Prever Share
#share_previsto = best_mlp_model_share.predict(novos_dados_scaled)[0]
#print(f"Share previsto: {share_previsto:.4f}")

# Prever FAT
#novos_dados_df['Share'] = share_previsto
#novos_dados_fat = novos_dados_df[df_fat.columns]
#novos_dados_fat_scaled = scaler_df.transform(novos_dados_fat)

#fat_previsto_scaled = best_mlp_model_fat.predict(novos_dados_fat_scaled).reshape(-1, 1)
#fat_previsto_original = scaler_alvo.inverse_transform(fat_previsto_scaled)[0][0]
#print(f"Faturamento previsto: {fat_previsto_original:.2f}")

