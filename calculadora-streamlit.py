import streamlit as st
import joblib
import pandas as pd

# Configuração da página
st.set_page_config(page_title="Previsão de Share e Faturamento", layout="centered")

# Cache para carregar os modelos apenas uma vez
@st.cache_resource
def load_models():
    try:
        models = {
            'model_share': joblib.load('best_mlp_model_share.pkl'),
            'model_fat': joblib.load('best_mlp_model_fat.pkl'),
            'scaler_share': joblib.load('scaler_share.pkl'),
            'scaler_df': joblib.load('scaler_df.pkl'),
            'scaler_alvo': joblib.load('scaler_alvo.pkl'),
            'colunas_fat': joblib.load('colunas.pkl')
        }
        return models
    except FileNotFoundError:
        st.error("Arquivos .pkl não encontrados! Certifique-se de que os modelos estão no repositório.")
        return None

models = load_models()

# Dicionário de tradução/exibição
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
    'Idade': 'Idade da Loja'
}

st.title("📊 Calculadora de Projeção")
st.markdown("Previsão de Participação de Mercado (Share) e Faturamento Mensal.")

if models:
    # --- FORMULÁRIO DE ENTRADA ---
    with st.form("previsao_form"):
        col1, col2 = st.columns(2)

        with col1:
            pos_loja = st.selectbox("Posição de Loja", ['Esquina', 'Meio de Quadra', 'Outro'])
            zona = st.selectbox("Zona", ['Zona Verde', 'Zona Amarela', 'Zona Vermelha'])
            uf = st.selectbox("UF", ['RS', 'SC', 'PR'])
            bairro = st.selectbox("Tipo de Localização", ['Bairro', 'Centro'])
            
        # Entradas numéricas dinâmicas com base nas colunas do modelo
        dados_input = {}
        
        # Mapeamento categórico para numérico (conforme seu código original)
        dados_input['Posição Loja_Meio de Quadra'] = 1.0 if pos_loja == 'Meio de Quadra' else 0.0
        dados_input['Posição Loja_Outro'] = 1.0 if pos_loja == 'Outro' else 0.0
        
        zona_map = {'Zona Verde': 0.0, 'Zona Amarela': 1.0, 'Zona Vermelha': 2.0}
        dados_input['Zona'] = zona_map[zona]
        
        bairro_map = {'Bairro': 0.0, 'Centro': 1.0}
        dados_input['Bairro'] = bairro_map[bairro]
        
        uf_map = {'RS': 0.0, 'SC': 1.0, 'PR': 2.0}
        dados_input['UF'] = uf_map[uf]

        st.subheader("Dados de Mercado e Concorrência")
        cols = st.columns(2)
        
        # Loop para criar os campos numéricos (filtrando o que já foi tratado)
        for i, col in enumerate(models['colunas_fat']):
            if col in ['Share', 'Estado_PR', 'Estado_RS', 'Estado_SC', 'FAT', 
                       'Posição Loja_Meio de Quadra', 'Posição Loja_Outro', 
                       'Zona', 'UF', 'Bairro', 'Sobra_demanda', 'captura_cidade']:
                continue
            
            label = nomes_exibicao.get(col, col)
            with cols[i % 2]:
                dados_input[col] = st.number_input(label, value=0.0, step=1.0)

        submit = st.form_submit_button("Calcular Previsão", use_container_width=True)

    # --- LÓGICA DE CÁLCULO ---
    if submit:
        try:
            # Cálculo da Captura e Sobra (conforme sua lógica)
            captura_cidade = dados_input['FAT_CID'] / dados_input['Demanda Cidade'] if dados_input['Demanda Cidade'] != 0 else 0
            dados_input['captura_cidade'] = captura_cidade
            sobra_demanda = 1.0 - captura_cidade

            # Organização dos dados para o Modelo de Share
            ordem_share = [
                'Zona', 'UF', 'Idade', 'Bairro', 'cont_rede_300', 'cont_ind_300', 
                'cont_fsj_300', 'cont_rede_600', 'cont_ind_600', 'cont_fsj_600', 
                'cont_rede_1000', 'cont_ind_1000', 'cont_fsj_1000', 'POPULAÇÃO 1KM', 
                'Demanda 1km', 'Demanda Cidade', 'FAT_CID', 'captura_cidade', 
                'Sobra_demanda', 'Posição Loja_Meio de Quadra', 'Posição Loja_Outro'
            ]
            
            df_share = pd.DataFrame([dados_input])
            df_share['Sobra_demanda'] = sobra_demanda
            df_share = df_share[ordem_share]
            
            # Predição do Share
            share_scaled = models['scaler_share'].transform(df_share)
            share_previsto = models['model_share'].predict(share_scaled)[0]

            # Predição do Faturamento (usando o share calculado)
            ordem_fat = [
                'Zona', 'UF', 'Idade', 'Bairro', 'cont_rede_300', 'cont_ind_300', 
                'cont_fsj_300', 'cont_rede_600', 'cont_ind_600', 'cont_fsj_600', 
                'cont_rede_1000', 'cont_ind_1000', 'cont_fsj_1000', 'POPULAÇÃO 1KM', 
                'Demanda 1km', 'Demanda Cidade', 'FAT_CID', 'captura_cidade', 
                'Share', 'Posição Loja_Meio de Quadra', 'Posição Loja_Outro'
            ]
            
            dados_input['Share'] = share_previsto
            df_fat = pd.DataFrame([dados_input])[ordem_fat]
            
            fat_scaled_input = models['scaler_df'].transform(df_fat)
            fat_previsto_scaled = models['model_fat'].predict(fat_scaled_input).reshape(-1, 1)
            fat_previsto_original = models['scaler_alvo'].inverse_transform(fat_previsto_scaled)[0][0]

            # --- RESULTADOS ---
            st.divider()
            res_col1, res_col2 = st.columns(2)
            res_col1.metric("Share Previsto", f"{share_previsto*100:.2f}%")
            res_col2.metric("Faturamento Mensal", f"R$ {fat_previsto_original:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
            
        except Exception as e:
            st.error(f"Erro no processamento: {e}")
