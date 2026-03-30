import streamlit as st
import joblib
import numpy as np

# Importação das bibliotecas de Machine Learning e métricas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

model = joblib.load('modelo_atendimento.pkl')

st.title("📊 Previsão de Falha no Atendimento")

clientes = st.slider("Clientes / Dia", 50, 200, 100)
guiches = st.slider("Guichês", 1, 15, 6)

tempo_autorizacao = st.slider("Tempo autorização min", 1, 30, 10)
erros = st.slider("Erros cadastro min", 0, 10, 3)

clientes_por_guiche = clientes / guiches

input_data = np.array([[
    clientes,
    guiches,
    tempo_autorizacao,
    erros,
    clientes_por_guiche
]])

if st.button("Prever"):
    resultado = model.predict(input_data)[0]

    if resultado == 1:
        st.error("⚠️ Falha prevista")
    else:
        st.success("✅ Atendimento OK")