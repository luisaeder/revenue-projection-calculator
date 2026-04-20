# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 2025

@author: luisa
"""

import joblib
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox

try:
    best_mlp_model_share = joblib.load('best_mlp_model_share.pkl')
    best_mlp_model_fat = joblib.load('best_mlp_model_fat.pkl')
    scaler_share = joblib.load('scaler_share.pkl')
    scaler_df = joblib.load('scaler_df.pkl')
    scaler_alvo = joblib.load('scaler_alvo.pkl')
    colunas_fat = joblib.load('colunas.pkl')
    
except FileNotFoundError:
    raise FileNotFoundError("Arquivos .pkl não encontrados! Rode primeiro o treino.py.")
#%%
nomes_exibicao = {
    'cont_rede_300': 'Qtd. Redes (300m)',
    'cont_ind_300': 'Qtd. Independentes (300m)',
    'cont_fsj_300': 'Qtd. Próprias (300m)',
    'cont_rede_600': 'Qtd. Redes (600m)',
    'cont_ind_600': 'Qtd. Independentes (600m)',
    'cont_fsj_600': 'Qtd. Próprias (600m)',
    'cont_rede_1000': 'Qtd. Redes (1km)',
    'cont_ind_1000': 'Qtd. Independentes (1km)',
    'cont_fsj_1000': 'Qtd. Próprias (1km)',
    'POPULAÇÃO 1KM': 'População em 1km',
    'Demanda 1km': 'Demanda farma em 1km',
    'Demanda Cidade': 'Demanda farma na cidade',
    'FAT_CID': 'Faturamento Próprio na cidade',
    'captura_cidade': 'Captura de mercado Própria na cidade',
    'Share': 'Participação de Mercado'   
}
#%%
root = tk.Tk()
root.title("Previsão de Share e Faturamento")

canvas = tk.Canvas(root)
canvas.pack(fill='both', expand=True, side='left')

frame = tk.Frame(canvas)
canvas.create_window((0, 0), window=frame, anchor='nw')

entries = {}
row = 0
#%%
tk.Label(frame, text='Posição de Loja').grid(row=row, column=0, padx=5, pady=2, sticky='w')
class_var = tk.StringVar()
ttk.Combobox(frame, textvariable=class_var, values=['Esquina', 'Meio de Quadra', 'Outro'], state="readonly").grid(row=row, column=1, padx=5, pady=2, sticky='ew')
class_var.set('Esquina')
entries['Class_Pos_Loja'] = class_var
row += 1


tk.Label(frame, text='Zona').grid(row=row, column=0, padx=5, pady=2, sticky='w')
zona_var = tk.StringVar()
ttk.Combobox(frame, textvariable=zona_var,
             values=['Zona Verde', 'Zona Amarela', 'Zona Vermelha'],
             state="readonly").grid(row=row, column=1, padx=5, pady=2, sticky='ew')
zona_var.set('Zona Verde')
entries['Zona'] = zona_var
row += 1

tk.Label(frame, text='UF').grid(row=row, column=0, padx=5, pady=2, sticky='w')
uf_var = tk.StringVar()
ttk.Combobox(frame, textvariable=uf_var, values=['RS', 'SC', 'PR'], state="readonly").grid(row=row, column=1, padx=5, pady=2, sticky='ew')
uf_var.set('RS')
entries['UF'] = uf_var
row += 1

tk.Label(frame, text='Bairro').grid(row=row, column=0, padx=5, pady=2, sticky='w')
bairro_var = tk.StringVar()
ttk.Combobox(frame, textvariable=bairro_var, values=['Bairro', 'Centro'], state="readonly").grid(row=row, column=1, padx=5, pady=2, sticky='ew')
bairro_var.set('Bairro')
entries['Bairro'] = bairro_var
row += 1
#%%
for col in colunas_fat:
    if col in ['Share', 'Estado_PR', 'Estado_RS', 'Estado_SC',
               'Posição Loja_Meio de Quadra', 'Posição Loja_Outro',
               'Zona','UF','Bairro','FAT','Sobra_demanda','captura_cidade']:
        continue
    label_text = nomes_exibicao.get(col, col)
    tk.Label(frame, text=label_text).grid(row=row, column=0, padx=5, pady=2, sticky='w')
    entry = tk.Entry(frame)
    entry.grid(row=row, column=1, padx=5, pady=2, sticky='ew')
    entries[col] = entry
    row += 1

#%%
def calcular_previsao():
    try:
        dados = {}
       
        pos_loja = entries['Class_Pos_Loja'].get()
        dados['Posição Loja_Meio de Quadra'] = 1.0 if pos_loja == 'Meio de Quadra' else 0.0
        dados['Posição Loja_Outro'] = 1.0 if pos_loja == 'Outro' else 0.0

        zona_map = {'Zona Verde': 0.0, 'Zona Amarela': 1.0, 'Zona Vermelha': 2.0}
        dados['Zona'] = zona_map[entries['Zona'].get()]
        
        bairro_map = {'Bairro': 0.0, 'Centro': 1.0}
        dados['Bairro'] = bairro_map[entries['Bairro'].get()]
        
        uf_map = {'RS':0.0, 'SC': 1.0, 'PR': 2.0}
        dados['UF'] = uf_map[entries['UF'].get()]
                
        fat_cid = entries['FAT_CID'].get().strip()
        fat_cid = float(fat_cid)
        demanda_cid = entries['Demanda Cidade'].get().strip()
        demanda_cid = float(demanda_cid)

        captura_cidade = fat_cid/demanda_cid
        captura_cidade = float(captura_cidade)
        dados['captura_cidade'] = captura_cidade
       

        for col in entries:
            if col not in ['Zona', 'UF', 'Class_Pos_Loja', 'Bairro', 'captura_cidade']:
                val = entries[col].get().strip()
                dados[col] = float(val)
        
        ordem_share = [
            'Zona',
            'UF',
            'Idade',
            'Bairro',
            'cont_rede_300',
            'cont_ind_300',
            'cont_fsj_300',
            'cont_rede_600',
            'cont_ind_600',
            'cont_fsj_600',
            'cont_rede_1000',
            'cont_ind_1000',
            'cont_fsj_1000',
            'POPULAÇÃO 1KM',
            'Demanda 1km',
            'Demanda Cidade',
            'FAT_CID',
            'captura_cidade',
            'Sobra_demanda',
            'Posição Loja_Meio de Quadra',
            'Posição Loja_Outro'
        ]
        
        ordem_fat = ['Zona',
                     'UF',
                     'Idade',
                     'Bairro',
                     'cont_rede_300',
                     'cont_ind_300',
                     'cont_fsj_300',
                     'cont_rede_600',
                     'cont_ind_600',
                     'cont_fsj_600',
                     'cont_rede_1000',
                     'cont_ind_1000',
                     'cont_fsj_1000',
                     'POPULAÇÃO 1KM',
                     'Demanda 1km',
                     'Demanda Cidade',
                     'FAT_CID',
                     'captura_cidade',
                     'Share',
                     'Posição Loja_Meio de Quadra',
                     'Posição Loja_Outro']
                
        dados_share = {k: v for k, v in dados.items() if k != 'Share'}
        df_share = pd.DataFrame([dados_share])
        df_share['Sobra_demanda'] = 1.0 - captura_cidade
        df_share = df_share[ordem_share]
        novos_dados_scaled = scaler_share.transform(df_share)
        share_previsto = best_mlp_model_share.predict(novos_dados_scaled)[0]

        dados['Share'] = share_previsto
        dados_fat = {k: v for k, v in dados.items() if k not in ('FAT', 'Sobra_demanda')}
        df_fat = pd.DataFrame([dados_fat])[ordem_fat]
        df_fat = df_fat[ordem_fat]
        novos_dados_fat_scaled = scaler_df.transform(df_fat)
        fat_previsto_scaled = best_mlp_model_fat.predict(novos_dados_fat_scaled).reshape(-1, 1)
        fat_previsto_original = scaler_alvo.inverse_transform(fat_previsto_scaled)[0][0]

        messagebox.showinfo("Resultado",
                            f"📊 Share previsto: {share_previsto*100:.2f}%\n"
                            f"💰 FAT previsto: R${fat_previsto_original:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                            )
        
    except Exception as e:
        messagebox.showerror("Erro", f"Erro na previsão: {e}")

def limpar_campos():
    for widget in entries.values():
        if isinstance(widget, tk.Entry):
            widget.delete(0, tk.END)
#%%
tk.Button(frame, text="Calcular Previsão", command=calcular_previsao, bg="#4CAF50", fg="white").grid(row=row, column=0, pady=10, padx=5)
tk.Button(frame, text="Limpar", command=limpar_campos, bg="#f44336", fg="white").grid(row=row, column=1, pady=10, padx=5)

frame.update_idletasks()
canvas.config(scrollregion=canvas.bbox("all"))
root.geometry("320x500")
root.mainloop()


