import streamlit as st
import joblib
import pandas as pd
import os
import shap
import matplotlib.pyplot as plt
import numpy as np

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
st.markdown("Monitorização contínua e previsão de estrangulamentos suportada por Machine Learning.  DEV.[Flayson Santos](https://github.com/FlaysonSantos/previsao-falhas-atendimento-saude/blob/main/projeto-atendimento-inteligente/README.md)")
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
# 🔮 MOTOR DE PREVISÃO E PRESCRIÇÃO (Dinamizado)
# ==========================================
st.subheader("⚙️ Centro de Decisão Estratégica")

if st.button("Executar Simulação do Sistema", type="primary", use_container_width=True):
    with st.spinner('O algoritmo está a calcular o impacto operacional...'):
        resultado = model.predict(dados_entrada.values)[0]
    
    st.markdown("<br>", unsafe_allow_html=True)
    res_col1, res_col2 = st.columns(2)

    if resultado == 1:
        with res_col1:
            st.error("### 🚨 STATUS: RISCO DE FALHA\nAlta probabilidade de quebra de fluxo com este cenário.")
            
        with res_col2:
            st.warning("### 🛠️ DECISÃO PRESCRITIVA")
            solucao_encontrada = False
            
            # Simulação automática para encontrar a configuração ideal
            for g in range(guiches + 1, 16):
                cpg_sim = clientes / g
                cen_sim = pd.DataFrame([[clientes, g, plano_saude, documentos, experiencia_operador, tempo_autorizacao, erros_cadastro, cpg_sim]], columns=dados_entrada.columns)
                
                if model.predict(cen_sim.values)[0] == 0:
                    st.success(f"💡 **Plano de Ação:** Abra mais **{g - guiches} guiché(s)** para estabilizar o processo.")
                    st.progress(g/15, text=f"Capacidade Necessária: {g}/15 guichés")
                    solucao_encontrada = True
                    break 
            
            if not solucao_encontrada:
                st.error("🚨 **COLAPSO DE CAPACIDADE**")
                st.info("Aumentar guichés não é suficiente. Atue em: **Erros de Registo** ou **Tempo de Autorização**.")
                
    else:
        with res_col1:
            st.success("### ✅ STATUS: OPERAÇÃO ESTÁVEL\nO cenário atual suporta a procura sem estrangulamentos.")
            
        with res_col2:
            st.info("### 🔍 OPORTUNIDADE LEAN (OTIMIZAÇÃO)")
            guiches_ideais = guiches
            for g in range(guiches - 1, 0, -1):
                cpg_sim = clientes / g
                cen_sim = pd.DataFrame([[clientes, g, plano_saude, documentos, experiencia_operador, tempo_autorizacao, erros_cadastro, cpg_sim]], columns=dados_entrada.columns)
                if model.predict(cen_sim.values)[0] == 0:
                    guiches_ideais = g
                else: break
            
            if guiches_ideais < guiches:
                st.metric("Potencial de Redução", value=f"{guiches - guiches_ideais} Guiché(s)", delta="- Custos Operacionais")
                st.write(f"Pode operar com apenas **{guiches_ideais} guichés** mantendo a segurança do SLA.")
            else:
                st.success("💡 **Eficiência Máxima:** Recursos perfeitamente dimensionados.")



# ==========================================
# 🧠 EXPLICABILIDADE DO MODELO (SHAP) E CAUSA RAIZ
# ==========================================
st.markdown("---")
st.subheader("🧠 Por que o modelo tomou esta decisão?")
st.write("A análise **SHAP** abaixo não olha para a regra geral, mas sim exclusivamente para **este cenário exato** que você acabou de simular. Descubra quais variáveis empurram o risco para a **FALHA (Vermelho)** e quais seguram a operação no **SUCESSO (Azul)**.")

try:
    with st.spinner('A calcular os valores de Shapley (SHAP)...'):
        # 1. Cria o 'Explicador' baseado no seu Random Forest
        explainer = shap.TreeExplainer(model)
        
        # 2. Calcula os valores SHAP apenas para os dados atuais simulados
        shap_values = explainer.shap_values(dados_entrada)
        
        # 3. Tratamento robusto para extrair apenas um array 1D (Isso resolve o erro!)
        if isinstance(shap_values, list):
            shap_instance = shap_values[1] # Pega a classe 1 (Falha) para modelos binários listados
        else:
            shap_instance = shap_values
            
        # Força o array a ficar plano (1D), transformando [[x, y]] em [x, y]
        shap_instance_1d = np.array(shap_instance).flatten()
        
        # Fallback de segurança caso a API do SHAP retorne classes combinadas
        if len(shap_instance_1d) > len(dados_entrada.columns):
            shap_instance_1d = shap_instance_1d[-len(dados_entrada.columns):]
            
        # 4. Criamos uma tabela formatada para visualizar no Streamlit
        df_shap = pd.DataFrame({
            'Variável': dados_entrada.columns,
            'Força (Impacto SHAP)': shap_instance_1d
        })
        
        # Ordenamos do maior impacto para o menor
        df_shap['Impacto Absoluto'] = df_shap['Força (Impacto SHAP)'].abs()
        df_shap = df_shap.sort_values(by='Impacto Absoluto', ascending=False).drop(columns=['Impacto Absoluto'])
        
        # 5. Desenhamos um gráfico de barras horizontal (Vermelho para risco, Azul para proteção)
        fig, ax = plt.subplots(figsize=(10, 5))
        
        cores = ['#ff4b4b' if x > 0 else '#1f77b4' for x in df_shap['Força (Impacto SHAP)']]
        ax.barh(df_shap['Variável'], df_shap['Força (Impacto SHAP)'], color=cores)
        
        ax.set_xlabel('← Ajuda a Estabilizar (Azul) | Aumenta Risco de Falha (Vermelho) →')
        ax.set_title('Impacto de cada variável na decisão atual')
        plt.gca().invert_yaxis() # Inverte para o maior impacto ficar no topo
        
        # Removemos as bordas para um visual mais limpo
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Renderizamos no Streamlit
        st.pyplot(fig)
        
        # 6. Adicionamos um Insight Dinâmico em texto
        ofensor_principal = df_shap.iloc[0]
        if ofensor_principal['Força (Impacto SHAP)'] > 0:
            st.error(f"🎯 **Causa Raiz Deste Cenário:** A variável **'{ofensor_principal['Variável']}'** é a principal responsável por empurrar esta simulação para o colapso.")
        else:
            st.success(f"🛡️ **Principal Fortaleza:** A variável **'{ofensor_principal['Variável']}'** é o que mais está ajudando a manter a operação estável neste momento.")

except Exception as e:
    st.info(f"⚠️ Não foi possível gerar a análise SHAP para este modelo. Detalhes: {e}")
