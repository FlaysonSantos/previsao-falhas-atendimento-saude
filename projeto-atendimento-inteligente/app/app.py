import streamlit as st
import joblib
import pandas as pd
import os

# 1. Carregamento do modelo com caminho dinâmico
# Isso garante que ele funcione tanto no computador local quanto no Streamlit Cloud
diretorio_atual = os.path.dirname(__file__)
caminho_modelo = os.path.join(diretorio_atual, 'modelo_atendimento.pkl')

try:
    model = joblib.load(caminho_modelo)
except FileNotFoundError:
    st.error(f"❌ Erro: O ficheiro 'modelo_atendimento.pkl' não foi encontrado no caminho: {caminho_modelo}")
    st.stop()

# ==========================================
# 🎛️ BARRA LATERAL (SIDEBAR) - Setup Operacional
# ==========================================
with st.sidebar:
    st.header("⚙️ Setup Operacional")
    st.markdown("Ajuste os parâmetros para simular o cenário atual da operação.")
    st.markdown("---")

    st.subheader("Volume e Capacidade")
    clientes = st.slider("Clientes / Dia", 50, 200, 100)
    guiches = st.slider("Guichés Abertos", 1, 15, 6)

    st.subheader("Perfil do Atendimento")
    plano_saude = st.selectbox("Possui Plano de Saúde?", [1, 0], format_func=lambda x: "Sim" if x == 1 else "Não")
    documentos = st.slider("Documentos Pendentes", 0, 10, 0)

    st.subheader("Desempenho da Equipe")
    experiencia_operador = st.slider("Experiência (Anos/Nível)", 0, 20, 5)
    tempo_autorizacao = st.slider("Tempo Autorização (min)", 1, 30, 10)
    erros_cadastro = st.slider("Erros de Registo", 0, 10, 3)

# Cálculo da feature derivada
clientes_por_guiche = clientes / guiches

# ==========================================
# 🖥️ ECRÃ PRINCIPAL - DASHBOARD DE GESTÃO
# ==========================================
st.title("📊 Dashboard Inteligente de Atendimento")
st.markdown("Monitorização contínua e previsão de estrangulamentos suportada por Machine Learning. 👨🏻‍💻Desenvolvido por [Flayson Santos](https://github.com/FlaysonSantos/previsao-falhas-atendimento-saude)")
st.markdown("---")

# Linha de KPIs (Indicadores Chave)
st.subheader("📈 Indicadores do Cenário Atual")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric(label="Volume de Clientes", value=clientes)
kpi2.metric(label="Capacidade (Guichés)", value=guiches)
kpi3.metric(label="Carga (Clientes/Guiché)", value=f"{clientes_por_guiche:.1f}")
kpi4.metric(label="Risco por Erros", value=erros_cadastro, delta="Alerta" if erros_cadastro > 5 else "Controlado", delta_color="inverse")

st.markdown("---")

# Estruturação dos dados
dados_entrada = pd.DataFrame([[
    clientes, guiches, plano_saude, documentos,
    experiencia_operador, tempo_autorizacao, erros_cadastro, clientes_por_guiche
]], columns=[
    'clientes', 'guiches', 'plano_saude', 'documentos',
    'experiencia_operador', 'tempo_autorizacao', 'erros_cadastro', 'clientes_por_guiche'
])


# ==========================================
# ⚙️ PAINEL DE DECISÃO DINÂMICO
# ==========================================
st.subheader("⚙️ Centro de Decisão Estratégica")

if st.button("Executar Simulação do Sistema", type="primary", use_container_width=True):
    with st.spinner('O algoritmo está a calcular o impacto operacional...'):
        resultado = model.predict(dados_entrada.values)[0]
    
    st.markdown("---")
    
    # 1. Indicadores de Impacto Imediato
    res1, res2 = st.columns(2)
    
    if resultado == 1:
        with res1:
            st.error("### 🚨 STATUS: RISCO DE FALHA")
            st.write("O cenário atual não sustenta o Nível de Serviço (NS) de 75%.") [cite: 164, 232]
        
        with res2:
            st.warning("### 🛠️ PLANO DE CONTINGÊNCIA")
            # Motor Prescritivo para recomendação automática
            solucao = False
            for g in range(guiches + 1, 16):
                cpg_sim = clientes / g
                cen_sim = pd.DataFrame([[clientes, g, plano_saude, documentos, experiencia_operador, tempo_autorizacao, erros_cadastro, cpg_sim]], columns=dados_entrada.columns)
                if model.predict(cen_sim.values)[0] == 0:
                    st.success(f"**Ação:** Aumentar para **{g} guichés** resolve o problema.")
                    st.progress((g/15), text=f"Capacidade Utilizada: {g}/15")
                    solucao = True
                    break
            if not solucao:
                st.info("⚠️ Capacidade máxima atingida. Reduza Erros ou Tempos de Autorização.") [cite: 1039, 1040]

    else:
        with res1:
            st.success("### ✅ STATUS: OPERAÇÃO ESTÁVEL")
            st.write("A operação está a entregar o NS acima da meta.") [cite: 1399]
        
        with res2:
            # Análise de Ociosidade (Lean Thinking)
            st.info("### 🍃 OPORTUNIDADE LEAN")
            guiches_ideais = guiches
            for g in range(guiches - 1, 0, -1):
                cpg_sim = clientes / g
                cen_sim = pd.DataFrame([[clientes, g, plano_saude, documentos, experiencia_operador, tempo_autorizacao, erros_cadastro, cpg_sim]], columns=dados_entrada.columns)
                if model.predict(cen_sim.values)[0] == 0:
                    guiches_ideais = g
                else: break
            
            if guiches_ideais < guiches:
                st.metric("Guichés Excedentes", value=guiches - guiches_ideais, delta="- Custo Operacional", delta_color="normal")
                st.write(f"Podes operar com apenas **{guiches_ideais} guichés** com segurança.")
            else:
                st.write("Recursos perfeitamente equilibrados (Just-in-Time).")

    # 2. Gráfico Dinâmico de Causa Raiz (Feature Importance)
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🔍 Análise de Causa Raiz em Tempo Real")
    st.write("Identificação automática do fator que mais está a pressionar o atendimento neste momento:")
    
    try:
        importancias = model.feature_importances_
        df_imp = pd.DataFrame(importancias * 100, index=dados_entrada.columns, columns=['Impacto %']).sort_values(by='Impacto %', ascending=True)
        st.bar_chart(df_imp, color="#ff4b4b")
    except:
        st.info("Gráfico de Pareto dinâmico indisponível.") [cite: 689, 713]


# --- FEATURE IMPORTANCE DINÂMICO ---
st.markdown("---")
st.subheader("📊 Importância das Variáveis no Cenário Atual")
try:
    importancias = model.feature_importances_
    df_imp = pd.DataFrame(importancias * 100, index=dados_entrada.columns, columns=['Impacto (%)']).sort_values(by='Impacto (%)', ascending=True)
    st.bar_chart(df_imp)
except:
    st.info("Gráfico de impacto indisponível para este modelo.")

# ==========================================
# 📊 GRÁFICO DE IMPACTO DAS VARIÁVEIS
# ==========================================
st.markdown("---")
st.subheader("🔍 O que mais impacta as falhas no atendimento?")
st.write("O gráfico abaixo mostra o 'peso' (importância) que o algoritmo dá a cada variável para prever uma falha, ajudando a identificar a causa raiz.")

try:
    # Extrai a importância de cada variável do modelo
    importancias = model.feature_importances_
    
    # Cria a tabela já com as variáveis no índice (Isto resolve o KeyError!)
    df_importancias = pd.DataFrame(
        importancias * 100,
        index=dados_entrada.columns,
        columns=['Impacto (%)']
    ).sort_values(by='Impacto (%)', ascending=True)

    # Desenha o gráfico de barras passando a tabela diretamente
    st.bar_chart(df_importancias, height=350)
    
except AttributeError:
    st.info("⚠️ O seu modelo não tem o atributo 'feature_importances_'. Isto costuma acontecer se o modelo guardado não for um Random Forest (ou Decision Tree).")
