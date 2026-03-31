import streamlit as st
import joblib
import pandas as pd
import os

# 1. Configuração da Página e Carregamento do Modelo
st.set_page_config(page_title="Dashboard de Operações", page_icon="📊", layout="wide")

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
st.markdown("Monitorização contínua e previsão de estrangulamentos suportada por Machine Learning.")
st.markdown("---")

# Linha de KPIs (Indicadores Chave)
st.subheader("📈 Indicadores do Cenário Atual")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric(label="Volume de Clientes", value=clientes)
kpi2.metric(label="Capacidade (Guichés)", value=guiches)
kpi3.metric(label="Carga (Clientes/Guiché)", value=f"{clientes_por_guiche:.1f}")

# Alerta dinâmico baseado em Carta de Controle (Fase Control do Green Belt)
risco_cor = "inverse" if erros_cadastro > 5 else "normal"
kpi4.metric(label="Risco por Erros", value=erros_cadastro, delta="Alerta" if erros_cadastro > 5 else "Controlado", delta_color=risco_cor)

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
# 🔮 MOTOR DE PREVISÃO, PRESCRIÇÃO E IMPACTO FINANCEIRO
# ==========================================
st.subheader("⚙️ Centro de Decisão Estratégica e Impacto Financeiro")

if st.button("Executar Simulação do Sistema", type="primary", use_container_width=True):
    with st.spinner('O algoritmo está calculando o impacto operacional e financeiro...'):
        resultado = model.predict(dados_entrada.values)[0]
    
    st.markdown("<br>", unsafe_allow_html=True)
    res_col1, res_col2 = st.columns(2)

    # --- VARIÁVEIS FINANCEIRAS DO BUSINESS CASE ---
    # Baseado na regra do Grupo Vitta: (NS Atual - Meta) * 235 * 192 mil clientes
    ganho_anual_protegido = 378556.80 
    # Estimativa de custo de um operador por mês (salário + encargos) para cálculo Lean
    custo_estimado_posto_mes = 3500.00 

    if resultado == 1:
        # --- CENÁRIO: RISCO DE QUEBRA DE SLA ---
        with res_col1:
            st.error("### 🚨 STATUS: RISCO DE FALHA")
            st.write("Alta probabilidade de quebra de fluxo com o cenário atual.")
            
            # Métrica de Dor Financeira
            st.metric(
                label="💸 Risco Financeiro Anualizado", 
                value=f"- R$ {ganho_anual_protegido:,.2f}", 
                delta="Perda potencial de bônus por quebra de NS", 
                delta_color="inverse"
            )
            
        with res_col2:
            st.warning("### 🛠️ DECISÃO PRESCRITIVA")
            solucao_encontrada = False
            
            # Simulação para achar a solução
            for g in range(guiches + 1, 16):
                cpg_sim = clientes / g
                cen_sim = pd.DataFrame([[clientes, g, plano_saude, documentos, experiencia_operador, tempo_autorizacao, erros_cadastro, cpg_sim]], columns=dados_entrada.columns)
                
                if model.predict(cen_sim.values)[0] == 0:
                    st.success(f"💡 **Ação:** Abra mais **{g - guiches} guichê(s)** imediatamente para proteger a receita da unidade.")
                    st.progress(g/15, text=f"Capacidade Física Necessária: {g}/15 guichês")
                    solucao_encontrada = True
                    break 
            
            if not solucao_encontrada:
                st.error("🚨 **COLAPSO DE CAPACIDADE**")
                st.info("Aumentar guichês não resolverá. Atue na causa raiz: reduza erros ou o tempo de autorização do sistema.")
                
    else:
        # --- CENÁRIO: OPERAÇÃO ESTÁVEL E OTIMIZAÇÃO ---
        with res_col1:
            st.success("### ✅ STATUS: OPERAÇÃO ESTÁVEL")
            st.write("O cenário suporta a demanda e garante a meta de NS.")
            
            # Métrica de Ganho Garantido
            st.metric(
                label="💰 Receita de SLA Protegida", 
                value=f"R$ {ganho_anual_protegido:,.2f}", 
                delta="Ganho anual mantido", 
                delta_color="normal"
            )
            
        with res_col2:
            st.info("### 🔍 OPORTUNIDADE LEAN (OTIMIZAÇÃO)")
            guiches_ideais = guiches
            
            # Otimização reversa: fechando guichês
            for g in range(guiches - 1, 0, -1):
                cpg_sim = clientes / g
                cen_sim = pd.DataFrame([[clientes, g, plano_saude, documentos, experiencia_operador, tempo_autorizacao, erros_cadastro, cpg_sim]], columns=dados_entrada.columns)
                if model.predict(cen_sim.values)[0] == 0:
                    guiches_ideais = g
                else: 
                    break
            
            if guiches_ideais < guiches:
                postos_salvos = guiches - guiches_ideais
                economia_mes = postos_salvos * custo_estimado_posto_mes
                
                # Métrica de Economia Lean
                st.metric(
                    label="Ociosidade Identificada", 
                    value=f"- {postos_salvos} Guichê(s)", 
                    delta=f"Economia estimada: R$ {economia_mes:,.2f} / mês", 
                    delta_color="normal"
                )
                st.write(f"Você pode operar com segurança usando apenas **{guiches_ideais} guichês** e realocar a equipe.")
            else:
                st.success("💡 **Eficiência Máxima:** Sem ociosidade. A operação está 100% otimizada.")



# ==========================================
# 📊 ANÁLISE DE CAUSA RAIZ (Feature Importance)
# ==========================================
st.markdown("---")
st.subheader("🔍 O que mais impacta as falhas no atendimento?")
try:
    importancias = model.feature_importances_
    df_imp = pd.DataFrame(importancias * 100, index=dados_entrada.columns, columns=['Impacto %']).sort_values(by='Impacto %', ascending=True)
    st.bar_chart(df_imp, color="#ff4b4b", height=350)
except AttributeError:
    st.info("⚠️ O seu modelo não suporta visualização de importância de variáveis.")
