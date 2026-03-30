import streamlit as st
import joblib
import pandas as pd

# 1. Configuração da Página (Layout Wide para expandir no ecrã)
st.set_page_config(page_title="Dashboard de Operações", page_icon="📊", layout="wide")

# Tentar carregar o modelo
try:
    model = joblib.load('modelo_atendimento.pkl')
except FileNotFoundError:
    st.error("❌ Erro: O ficheiro 'modelo_atendimento.pkl' não foi encontrado.")
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
st.markdown("Monitorização contínua e previsão de estrangulamentos suportada por Machine Learning. 👨🏻‍💻Desenvolvido por [Flayson Santos](https://github.com/FlaysonSantos)")
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
# 🔮 MOTOR DE PREVISÃO E PRESCRIÇÃO
# ==========================================
st.subheader("⚙️ Análise Preditiva e Plano de Ação")

if st.button("Executar Simulação do Sistema", type="primary", use_container_width=True):
    with st.spinner('A processar o algoritmo e analisar variáveis...'):
        resultado = model.predict(dados_entrada.values)[0]
    
    st.markdown("<br>", unsafe_allow_html=True)

    col_alert, col_action = st.columns(2)

    if resultado == 1:
        # CENÁRIO 1: VAI DAR FALHA
        with col_alert:
            st.error("### ⚠️ ALERTA DE FALHA\nAlta probabilidade de quebra de fluxo com este cenário. Necessária intervenção.")
            
        with col_action:
            st.warning("### 🔍 AVALIANDO SOLUÇÕES...")
            solucao_encontrada = False
            
            # Simula ABRIR guichés até ao limite máximo de 15
            for guiches_simulados in range(guiches + 1, 16):
                cpg_simulado = clientes / guiches_simulados
                cenario_simulado = pd.DataFrame([[
                    clientes, guiches_simulados, plano_saude, documentos, 
                    experiencia_operador, tempo_autorizacao, erros_cadastro, cpg_simulado
                ]], columns=dados_entrada.columns)
                
                if model.predict(cenario_simulado.values)[0] == 0:
                    guiches_a_abrir = guiches_simulados - guiches
                    st.success(f"💡 **Plano de Ação:** Abra mais **{guiches_a_abrir} guiché(s)** (passando a operar com {guiches_simulados} no total) para estabilizar o processo.")
                    solucao_encontrada = True
                    break 
                    
            # SE CHEGOU A 15 GUICHÉS E CONTINUA A FALHAR (O SEU NOVO CENÁRIO)
            if not solucao_encontrada:
                st.error("🚨 **COLAPSO DE CAPACIDADE EMINENTE**")
                st.info(f"💡 **Diagnóstico:** O volume atual ({clientes} clientes) ultrapassa a capacidade máxima física da unidade (15 guichés). Aumentar a equipa já não é possível.")
                st.warning(
                    "🛠️ **Ativar Plano de Contingência:**\n"
                    "- **Desvio de Fluxo:** Direcionar clientes para o autoatendimento, aplicação ou triagem rápida.\n"
                    "- **Força-Tarefa:** Deslocar supervisores ou backoffice para o atendimento de linha da frente.\n"
                    "- **Modo Emergência:** Focar em zerar *Erros de Registo* e acelerar o *Tempo de Autorização* ao máximo."
                )
                
    else:
        # CENÁRIO 2: ATENDIMENTO OK (Verifica se há desperdício)
        with col_alert:
            st.success("### ✅ OPERAÇÃO ESTÁVEL\nO cenário atual suporta a procura sem estrangulamentos.")
            
        with col_action:
            st.info("### 🔍 PROCURANDO OPORTUNIDADES LEAN...")
            guiches_ideais = guiches
            
            # Simula FECHAR guichés do atual até 1
            for guiches_simulados in range(guiches - 1, 0, -1):
                cpg_simulado = clientes / guiches_simulados
                cenario_simulado = pd.DataFrame([[
                    clientes, guiches_simulados, plano_saude, documentos, 
                    experiencia_operador, tempo_autorizacao, erros_cadastro, cpg_simulado
                ]], columns=dados_entrada.columns)
                
                if model.predict(cenario_simulado.values)[0] == 1:
                    break
                else:
                    guiches_ideais = guiches_simulados
                    
            if guiches_ideais < guiches:
                guiches_a_fechar = guiches - guiches_ideais
                st.warning(f"💡 **Alerta de Ociosidade:** Está a utilizar recursos em excesso. Pode **fechar {guiches_a_fechar} guiché(s)** e operar apenas com **{guiches_ideais}**. O atendimento continuará estável e reduzirá os custos operacionais.")
            else:
                st.success("💡 **Eficiência Máxima:** Os seus recursos estão perfeitamente dimensionados para a procura atual. Não há ociosidade.")


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