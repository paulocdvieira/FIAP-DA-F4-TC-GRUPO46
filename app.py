

# streamlit run app.py

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# --- 1. CLASSES CUSTOMIZADAS ---
class OrdinalFeature(BaseEstimator, TransformerMixin):
    def __init__(self, cols=['CAEC', 'CALC']):
        self.cols = cols
        self.mapping = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
    def fit(self, X, y=None): return self
    def transform(self, X):
        X_copy = X.copy()
        for col in self.cols:
            X_copy[col] = X_copy[col].map(self.mapping).fillna(0)
        return X_copy

class ObesityEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.binary_mapping = {'no': 0, 'yes': 1}
        self.gender_mapping = {'Female': 0, 'Male': 1}
        self.oh_cols = ['MTRANS']
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    def fit(self, X, y=None):
        self.encoder.fit(X[self.oh_cols])
        return self
    def transform(self, X):
        X_copy = X.copy()
        X_copy['Gender'] = X_copy['Gender'].map(self.gender_mapping)
        for col in ['family_history', 'FAVC', 'SMOKE', 'SCC']:
            X_copy[col] = X_copy[col].map(self.binary_mapping)
        encoded = self.encoder.transform(X_copy[self.oh_cols])
        names = self.encoder.get_feature_names_out(self.oh_cols)
        df_encoded = pd.DataFrame(encoded, columns=names, index=X_copy.index)
        return pd.concat([X_copy.drop(columns=self.oh_cols), df_encoded], axis=1)

class MinMaxTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols=['Age', 'Height', 'Weight', 'IMC']):
        self.cols = cols
        self.scaler = MinMaxScaler()
    def fit(self, X, y=None):
        self.scaler.fit(X[self.cols])
        return self
    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.cols] = self.scaler.transform(X_copy[self.cols])
        return X_copy

# --- 2. CONFIGURAÇÕES ---
st.set_page_config(page_title="Diagnóstico via Modelo Preditivo", page_icon="⚖️", layout="wide")

@st.cache_resource
def carregar_ativos():
    modelo = joblib.load('modelo_obesidade_rf.joblib')
    colunas = joblib.load('colunas_modelo.joblib')
    return modelo, colunas

pipeline, colunas_treino = carregar_ativos()

# --- 3. MAPEAMENTOS ---
map_genero = {"Masculino": "Male", "Feminino": "Female"}
map_sim_nao = {"Sim": "yes", "Não": "no"}
map_frequencia = {"Não consome": "no", "Às vezes": "Sometimes", "Frequentemente": "Frequently", "Sempre": "Always"}
map_transporte = {"Transporte Público": "Public_Transportation", "Automóvel": "Automobile", "Caminhada": "Walking", "Motocicleta": "Motorbike", "Bicicleta": "Bike"}

map_resultados = {
    'Insufficient_Weight': 'Peso Insuficiente', 'Normal_Weight': 'Peso Normal',
    'Overweight_Level_I': 'Sobrepeso Nível I', 'Overweight_Level_II': 'Sobrepeso Nível II',
    'Obesity_Type_I': 'Obesidade Grau I', 'Obesity_Type_II': 'Obesidade Grau II', 'Obesity_Type_III': 'Obesidade Grau III'
}

map_features_pt = {
    'Gender': 'Gênero', 'Age': 'Idade', 'Height': 'Altura', 'Weight': 'Peso',
    'family_history': 'Histórico Familiar', 'FAVC': 'Consumo Calórico Freq.',
    'FCVC': 'Consumo de Vegetais', 'NCP': 'Nº de Refeições', 'CAEC': 'Comer entre Refeições',
    'SMOKE': 'Fumante', 'CH2O': 'Consumo de Água', 'SCC': 'Monitoramento de Calorias',
    'FAF': 'Atividade Física', 'TUE': 'Uso de Eletrônicos', 'CALC': 'Consumo de Álcool',
    'IMC': 'IMC (Cálculo Biométrico)', 'MTRANS_Automobile': 'Transporte: Automóvel',
    'MTRANS_Bike': 'Transporte: Bicicleta', 'MTRANS_Motorbike': 'Transporte: Moto',
    'MTRANS_Public_Transportation': 'Transporte: Público', 'MTRANS_Walking': 'Transporte: Caminhada'
}

# --- 4. INTERFACE ---
st.title("⚖️ Diagnóstico via Modelo Preditivo")
st.markdown("""
Esta ferramenta utiliza um **Modelo Preditivo** treinado com dados de pessoas entre **14 e 61 anos**. 
Preencha os campos abaixo para obter uma análise baseada em hábitos e dados biométricos.
""")

with st.form("main_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("📋 Perfil")
        genero_pt = st.selectbox("Gênero", ["Selecione..."] + list(map_genero.keys()))
        age = st.number_input("Idade", min_value=0, max_value=120, value=0)
        height = st.number_input("Altura (m) - ex: 1.70", min_value=0.0, max_value=2.5, value=0.0, step=0.01)
        weight = st.number_input("Peso (kg)", min_value=0.0, max_value=300.0, value=0.0, step=0.1)
    
    with c2:
        st.subheader("🥗 Dieta e Hidratação")
        hist_familiar_pt = st.selectbox("Histórico familiar de sobrepeso?", ["Selecione..."] + list(map_sim_nao.keys()))
        favc_pt = st.selectbox("Consome alimentos calóricos com frequência?", ["Selecione..."] + list(map_sim_nao.keys()))
        fcvc = st.slider("Consumo de vegetais (1-3)", 1, 3, 1, help="1:Raramente, 2:Às vezes, 3:Sempre")
        ch2o = st.slider("Consumo de água (1-3)", 1, 3, 1, help="1:<1L, 2:1-2L, 3:>2L")
        caec_pt = st.selectbox("Lanches entre refeições?", ["Selecione..."] + list(map_frequencia.keys()))

    with c3:
        st.subheader("🏃 Estilo de Vida")
        faf = st.slider("Atividade física (0-3)", 0, 3, 0, help="0:Nenhuma, 1:1-2x, 2:3-4x, 3:5x ou +")
        tue = st.slider("Tempo de eletrônicos (0-2)", 0, 2, 0, help="0:0-2h, 1:3-5h, 2:>5h")
        smoke_pt = st.selectbox("Fumante?", ["Selecione..."] + list(map_sim_nao.keys()))
        calc_pt = st.selectbox("Consumo de álcool?", ["Selecione..."] + list(map_frequencia.keys()))
        mtrans_pt = st.selectbox("Meio de transporte", ["Selecione..."] + list(map_transporte.keys()))
        scc_pt = st.selectbox("Monitora calorias?", ["Selecione..."] + list(map_sim_nao.keys()))
        ncp = st.slider("Refeições principais por dia", 1, 4, 1)

    submit = st.form_submit_button("ANALISAR PERFIL")

# --- 5. RESULTADOS ---
if submit:
    if "Selecione..." in [genero_pt, hist_familiar_pt, smoke_pt, favc_pt, caec_pt, scc_pt, calc_pt, mtrans_pt] or age == 0 or height == 0 or weight == 0:
        st.error("⚠️ Por favor, preencha todos os campos corretamente.")
    else:
        imc = weight / (height ** 2)
        dados = {
            'Gender': map_genero[genero_pt], 'Age': age, 'Height': height, 'Weight': weight,
            'family_history': map_sim_nao[hist_familiar_pt], 'FAVC': map_sim_nao[favc_pt],
            'FCVC': fcvc, 'NCP': ncp, 'CAEC': map_frequencia[caec_pt], 'SMOKE': map_sim_nao[smoke_pt],
            'CH2O': ch2o, 'SCC': map_sim_nao[scc_pt], 'FAF': faf, 'TUE': tue, 'CALC': map_frequencia[calc_pt],
            'MTRANS': map_transporte[mtrans_pt], 'IMC': imc
        }
        
        X_user = pd.DataFrame([dados])[colunas_treino]
        pred_id = pipeline.predict(X_user)[0]
        
        res_en = ['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']
        res_final = map_resultados[res_en[pred_id]]
        
        st.success(f"### Resultado do Modelo Preditivo: {res_final}")
        st.info(f"IMC Calculado: {imc:.2f}")

        # Gráfico IMC
        st.write("### 📊 Posicionamento no IMC (Referência OMS)")
        faixas = {"Abaixo do Peso": 18.5, "Peso Normal": 24.9, "Sobrepeso": 29.9, "Obesidade": 40.0}
        df_ref = pd.DataFrame(list(faixas.items()), columns=['Status', 'Limite'])
        fig_ref, ax_ref = plt.subplots(figsize=(10, 3))
        sns.barplot(data=df_ref, x='Limite', y='Status', palette="coolwarm", ax=ax_ref)
        ax_ref.axvline(imc, color='black', linestyle='--', linewidth=2, label=f'Seu IMC ({imc:.1f})')
        plt.legend()
        st.pyplot(fig_ref)

        # Importância das Variáveis
        st.write("### 🔍 Fatores Determinantes para este Diagnóstico")
        rf_model = pipeline.named_steps['model']
        features_final = pipeline.named_steps['cat'].transform(pipeline.named_steps['ord'].transform(X_user)).columns
        importancias = pd.Series(rf_model.feature_importances_, index=features_final)
        importancias.index = [map_features_pt.get(col, col) for col in importancias.index]
        importancias = importancias.sort_values(ascending=False).head(7)
        
        fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
        importancias.plot(kind='barh', ax=ax_imp, color='skyblue')
        ax_imp.invert_yaxis()
        st.pyplot(fig_imp)