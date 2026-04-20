# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 09:36:05 2025

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

# ====================== CARREGAR E PREPARAR DADOS ======================
df = pd.read_excel('Base Teste 7.xlsx')
#df.columns.tolist()
df.dropna(inplace=True)

# Remover outliers de FAT
df['FAT_z'] = stats.zscore(df['FAT'])
df = df[np.abs(df['FAT_z']) <= 2].drop(columns=['FAT_z'])

# Remover outliers de Share
df['Share_z'] = stats.zscore(df['Share'])
df = df[np.abs(df['Share_z']) <= 2].drop(columns=['Share_z'])

# One-hot encoding
df = pd.get_dummies(df, columns=df.select_dtypes(include='object').columns, drop_first=True)
df.dropna(inplace=True)

# ====================== ETAPA 1: Prever SHARE ======================
df_share = df.drop(columns=['FAT'])
train_share, test_share = train_test_split(df_share, test_size=0.2, random_state=0)

scaler_share = StandardScaler()
x_train_share = train_share.drop(columns=['Share'])
y_train_share = train_share['Share']
x_test_share = test_share.drop(columns=['Share'])
y_test_share = test_share['Share']

x_train_share_scaled = scaler_share.fit_transform(x_train_share)
x_test_share_scaled = scaler_share.transform(x_test_share)


# Modelo com parâmetros ótimos para Share
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


# Avaliação Share
y_pred_share = best_mlp_model_share.predict(x_test_share_scaled)
print("MSE Share:", mean_squared_error(y_test_share, y_pred_share))

# ====================== ETAPA 2: Prever FAT ======================
df_fat = df.drop(columns=['FAT', 'Sobra_demanda'])
alvo_fat = df[['FAT']]

scaler_df = StandardScaler()
df_scaled = scaler_df.fit_transform(df_fat)

scaler_alvo = StandardScaler()
alvo_scaled = scaler_alvo.fit_transform(alvo_fat)

train_fat, test_fat = train_test_split(df_scaled, test_size=0.2, random_state=0)
train_target, test_target = train_test_split(alvo_scaled, test_size=0.2, random_state=0)

# Modelo com parâmetros ótimos para FAT
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

# Avaliação FAT
y_pred_fat_scaled = best_mlp_model_fat.predict(test_fat).reshape(-1, 1)
y_pred_fat = scaler_alvo.inverse_transform(y_pred_fat_scaled)
print("MSE FAT:", mean_squared_error(test_target, y_pred_fat_scaled))

# ====================== SALVAR MODELOS E SCALERS ======================
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

#%%
import matplotlib.pyplot as plt

# --- CONFIGURAÇÃO GLOBAL (Norma Tabela 8) ---
plt.rcParams['font.family'] = 'Arial'           # Fonte Arial
plt.rcParams['font.size'] = 11                  # Tamanho 11 (ou menor)
plt.rcParams['axes.titlesize'] = 0              # REMOVE TÍTULO (não deve conter)
plt.rcParams['axes.labelsize'] = 11             # Títulos dos eixos tamanho 11
plt.rcParams['axes.labelcolor'] = 'black'       # Cor da fonte preta
plt.rcParams['axes.linewidth'] = 1.5            # Largura dos eixos 1,5 pt
plt.rcParams['axes.edgecolor'] = 'black'        # Cor dos eixos preta
plt.rcParams['axes.grid'] = False               # REMOVE GRADES (não deve conter)
plt.rcParams['figure.facecolor'] = 'white'      # Fundo branco (sem preenchimento)
plt.rcParams['axes.facecolor'] = 'white'        # Área do gráfico branca
#%%
plt.figure(figsize=(7, 5))

# 1. Plot normal
plt.scatter(y_test_original, y_pred_original, alpha=0.6, color='teal', edgecolors='none') # Tirei a borda branca dos pontos
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1.5, label='Ideal')

# 2. AJUSTE DAS SPINES (O que a Tabela 8 pede)
ax = plt.gca()
ax.spines['top'].set_visible(False)    # Remove borda superior
ax.spines['right'].set_visible(False)  # Remove borda direita
ax.spines['left'].set_linewidth(1.5)   # Garante 1,5pt no Y
ax.spines['bottom'].set_linewidth(1.5) # Garante 1,5pt no X

# 3. TEXTOS (Tudo em preto e Arial 11 pelo rcParams)
plt.xlabel('Valor Real (R$)')
plt.ylabel('Valor Predito (R$)')
# plt.title(...)  <-- APAGUE ESSA LINHA (A norma proíbe título interno)

# 4. LEGENDA (Sem moldura/borda)
plt.legend(frameon=False)

plt.savefig('faturamento_real_vs_predito.png', dpi=300, bbox_inches='tight')
plt.show()
#%%
plt.figure(figsize=(7, 5))
residuos = y_test_original.values.flatten() - y_pred_original.flatten()

plt.scatter(y_pred_original, residuos, alpha=0.6, color='orange', edgecolors='none')
plt.axhline(y=0, color='red', linestyle='--', lw=1.5)

# Ajuste de escala para Milhões (10^6)
plt.ticklabel_format(style='sci', axis='y', scilimits=(6,6))

# Estilo em L
ax = plt.gca()
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['left', 'bottom']].set_linewidth(1.5)

plt.xlabel('Valores Preditos (R$)')
plt.ylabel('Resíduos (Real - Predito)')

plt.savefig('grafico_residuos.png', dpi=300, bbox_inches='tight')
plt.show()
#%%
plt.figure(figsize=(7, 5))
plt.plot(best_mlp_model_fat.loss_curve_, color='royalblue', lw=2)

# Estilo em L
ax = plt.gca()
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['left', 'bottom']].set_linewidth(1.5)

plt.xlabel('Iterações (Épocas)')
plt.ylabel('Perda (MSE)')

plt.savefig('curva_perda.png', dpi=300, bbox_inches='tight')
plt.show()
#%%
#plt.figure(figsize=(10, 5)) # Mais largo para as amostras respirarem
plt.plot(y_test_share.values, color='red', label='Real', lw=1.5)
plt.plot(nn_predict, color='royalblue', label='Predito', lw=1.5)

# Estilo em L
ax = plt.gca()
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['left', 'bottom']].set_linewidth(1.5)

plt.xlabel('Amostras (Teste)')
plt.ylabel('Participação de Mercado (Share)')
plt.legend(frameon=False)

plt.savefig('comparativo_share.png', dpi=300, bbox_inches='tight')
plt.show()
#%%
plt.plot(y_test_original.values, color='red', label='Real', lw=1.5)
plt.plot(y_pred_original, color='royalblue', label='Predito', lw=1.5)

plt.ticklabel_format(style='sci', axis='y', scilimits=(6,6))

# Estilo em L
ax = plt.gca()
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['left', 'bottom']].set_linewidth(1.5)

plt.xlabel('Amostras (Teste)')
plt.ylabel('Faturamento (R$)')
plt.legend(frameon=False)

plt.savefig('comparativo_fat.png', dpi=300, bbox_inches='tight')
plt.show()
#%%
# --- PRINT DE MÉTRICAS EXTRAS PARA O TEXTO ---
print("\n--- Métricas Finais (FAT) ---")
print(f"R2 Score: {r2_score(y_test_original, y_pred_original):.4f}")
print(f"MAE: {mean_absolute_error(y_test_original, y_pred_original):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(test_target, y_pred_fat_scaled)):.4f}")

#%%
from sklearn.metrics import r2_score

# --- Métricas para SHARE ---
r2_share = r2_score(y_test_share, y_pred_share)
# MAPE para Share (como os valores são pequenos, o MAPE dá uma visão de erro relativo)
mape_share = np.mean(np.abs((y_test_share - y_pred_share) / y_test_share)) * 100
precisao_share = 100 - mape_share

# --- Métricas para FAT (na escala original de Reais) ---
r2_fat = r2_score(y_test_original, y_pred_original)
# MAPE para FAT (fundamental para o negócio entender a margem de erro)
mape_fat = np.mean(np.abs((y_test_original.values.flatten() - y_pred_original.flatten()) / y_test_original.values.flatten())) * 100
precisao_fat = 100 - mape_fat

print(f"--- Desempenho SHARE ---")
print(f"R2: {r2_share:.4f}")
print(f"Precisão Preditiva (1-MAPE): {precisao_share:.2f}%")

print(f"\n--- Desempenho FAT ---")
print(f"R2: {r2_fat:.4f}")
print(f"Precisão Preditiva (1-MAPE): {precisao_fat:.2f}%")