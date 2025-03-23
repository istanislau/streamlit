import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error

st.set_page_config(page_title="Dashboard Polinomial", layout="wide")

# === Upload de CSV ===
st.markdown("## 📁 Importar Dataset")
with st.expander("⬆️ Carregar CSV personalizado"):
    st.markdown("O arquivo deve conter pelo menos as colunas:")
    st.markdown("- `date`: datas no formato dd/mm/yyyy")
    st.markdown("- `size_db`, `cpu`, `mem`: variáveis alvo")
    st.markdown(
        "Todas as colunas após `mem` serão consideradas variáveis explicativas.")
    uploaded_file = st.file_uploader("Escolha o CSV", type="csv")

if uploaded_file:
    st.session_state['uploaded'] = True
    df = pd.read_csv(uploaded_file, sep=',', decimal='.')

    required_cols = ['date', 'size_db', 'cpu', 'mem']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(
            f"❌ O CSV está faltando as colunas obrigatórias: {', '.join(missing_cols)}")
        st.stop()

    try:
        df['date'] = pd.to_datetime(df['date'], format="%d/%m/%Y")
    except Exception:
        st.error("❌ A coluna `date` deve estar no formato dd/mm/yyyy")
        st.stop()
else:
    st.session_state['uploaded'] = False
    st.warning(
        "⚠️ Nenhum arquivo CSV carregado. Por favor, carregue um arquivo para iniciar a análise.")
    st.stop()

# === Requer upload antes de continuar ===
if 'uploaded' not in st.session_state or not st.session_state['uploaded']:
    st.stop()

original_df = df.copy()
# === Sidebar reorganizada ===
filtrar_data = st.sidebar.checkbox(
    "Filtrar por intervalo de datas", value=True)
st.sidebar.header("Filtros de Data")
data_min = df['date'].min().date()
data_max = df['date'].max().date()
col1, col2 = st.sidebar.columns(2)
data_inicial = col1.date_input("Data inicial", value=data_min,
                               min_value=data_min, max_value=data_max, key="input_data_inicial")
data_final = col2.date_input("Data final", value=data_max,
                             min_value=data_min, max_value=data_max, key="input_data_final")
data_range = st.sidebar.slider("Intervalo de datas", min_value=data_min,
                               max_value=data_max, value=(data_min, data_max), key="slider_datas")


st.sidebar.markdown("---")
with st.sidebar.expander("⚙️ Parâmetros do Modelo"):
    grau_regressao = st.slider("Grau da Regressão Polinomial",
                               min_value=1, max_value=5, value=2, key="slider_grau")
    media_movel_janela = st.slider(
        "Janela da média móvel (dias)", min_value=3, max_value=30, value=7, step=1, key="slider_mm")

# === Parâmetros e variáveis explicativas ===
variaveis_exp = df.columns.tolist()
variaveis_exp = variaveis_exp[variaveis_exp.index('mem') + 1:]  # após memória
st.sidebar.markdown("---")
with st.sidebar.expander("📦 Valores Futuros Customizados"):
    usar_valores_custom = st.checkbox(
        "Usar valores futuros customizados", key="check_futuro")
    dias = st.slider("Dias de projeção futura", min_value=7,
                     max_value=180, value=90, step=1, key="dias_projecao")
    if usar_valores_custom:
        custom_inputs = {}
        for var in variaveis_exp:
            custom_inputs[var] = st.number_input(
                var, value=float(original_df[var].iloc[-1]))
        future_X = pd.DataFrame([custom_inputs] * dias)
    future_days = dias


if filtrar_data:
    df = df[(df['date'] >= pd.to_datetime(data_range[0])) &
            (df['date'] <= pd.to_datetime(data_range[1]))]


# === Layout de abas ===
tab1, tab2 = st.tabs(["Visualização", "Modelos Estatísticos"])

with tab1:
    st.header("Visualização de Dados")

    # Valores máximos e mínimos
    st.subheader("📌 Destaques de Consumo")
    col1, col2, col3 = st.columns(3)
    colunas = [('size_db', 'DB (TB)', col1),
               ('cpu', 'CPU (%)', col2), ('mem', 'Memória (%)', col3)]
    for col, label, bloco in colunas:
        max_val = df[col].max()
        min_val = df[col].min()
        max_date = df[df[col] == max_val]['date'].iloc[0].strftime('%d/%m/%Y')
        min_date = df[df[col] == min_val]['date'].iloc[0].strftime('%d/%m/%Y')
        with bloco:
            st.metric(label=f"{label} Máximo",
                      value=f"{max_val:.2f}", delta=f"em {max_date}")
            st.metric(label=f"{label} Mínimo",
                      value=f"{min_val:.2f}", delta=f"em {min_date}")

    # Novos destaques: totais e médias semanais
    st.markdown("---")
    st.subheader("📊 Métricas de Atividade")

    col_exp_sel, col_aggr_sel = st.columns(2)
    with col_exp_sel:
        exp_vars_selecionadas = st.multiselect(
            "Variáveis para análise", options=variaveis_exp, default=variaveis_exp)
    with col_aggr_sel:
        tipo_agregacao = st.selectbox("Tipo de agregação", options=[
                                      'média semanal', 'soma total'])

    # total_ativos = df['clientes_ativos'].max()
    semanas = (df['date'].max() - df['date'].min()).days / 7

    agregados = {}
    for var in exp_vars_selecionadas:
        if tipo_agregacao == 'média semanal':
            agregados[var] = df[var].sum() / semanas
        else:
            agregados[var] = df[var].sum()

        cols = st.columns(len(exp_vars_selecionadas))
    for i, var in enumerate(exp_vars_selecionadas):
        with cols[i]:
            label = var.replace('_', ' ').capitalize()
            val = agregados[var]
            unidade = "/semana" if tipo_agregacao == 'média semanal' else " total"
            st.metric(f"{label} ({unidade})", f"{val:.1f}")
        max_val = df[col].max()
        min_val = df[col].min()
        max_date = df[df[col] == max_val]['date'].iloc[0].strftime('%d/%m/%Y')
        min_date = df[df[col] == min_val]['date'].iloc[0].strftime('%d/%m/%Y')

    # === Gráficos suavizados ===
    X_poly = df[variaveis_exp]
    y_size = df['size_db']
    y_cpu = df['cpu']
    y_mem = df['mem']

    modelo_size = make_pipeline(PolynomialFeatures(
        grau_regressao), LinearRegression()).fit(X_poly, y_size)
    modelo_cpu = make_pipeline(PolynomialFeatures(
        grau_regressao), LinearRegression()).fit(X_poly, y_cpu)
    modelo_mem = make_pipeline(PolynomialFeatures(
        grau_regressao), LinearRegression()).fit(X_poly, y_mem)

    y_pred_size = modelo_size.predict(X_poly)
    y_pred_cpu = modelo_cpu.predict(X_poly)
    y_pred_mem = modelo_mem.predict(X_poly)

    fig_size = go.Figure()
    fig_size.add_trace(go.Scatter(
        x=df['date'], y=df['size_db'].rolling(window=media_movel_janela * 2, min_periods=1).mean(), mode='lines', name='Tamanho DB (TB) - Suavizado'))
    fig_size.update_layout(title='Tamanho da Base de Dados', yaxis_title='TB')

    fig_cpu_mem = go.Figure()
    fig_cpu_mem.add_trace(go.Scatter(
        x=df['date'], y=df['cpu'].rolling(window=media_movel_janela * 2, min_periods=1).mean(), mode='lines', name='CPU (%) - Suavizado'))
    fig_cpu_mem.add_trace(go.Scatter(
        x=df['date'], y=df['mem'].rolling(window=media_movel_janela * 2, min_periods=1).mean(), mode='lines', name='Memória (%) - Suavizado'))
    fig_cpu_mem.update_layout(
        title='Consumo de CPU e Memória', yaxis_title='%')

    # Tendência com média móvel
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Tamanho da Base de Dados")
        st.plotly_chart(fig_size, use_container_width=True)
    with col2:
        st.subheader("📊 Consumo de CPU e Memória")
        st.plotly_chart(fig_cpu_mem, use_container_width=True)

    # Gráfico de tendência após os principais
    st.subheader(f"📈 Tendência (Média Móvel {media_movel_janela} dias)")
    df_mm = df[['date', 'size_db', 'cpu', 'mem']].copy()
    df_mm['size_db_mm'] = df_mm['size_db'].rolling(
        window=media_movel_janela).mean()
    df_mm['cpu_mm'] = df_mm['cpu'].rolling(window=media_movel_janela).mean()
    df_mm['mem_mm'] = df_mm['mem'].rolling(window=media_movel_janela).mean()

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=df_mm['date'], y=df_mm['size_db_mm'], mode='lines', name='DB MM'))
    fig_trend.add_trace(go.Scatter(
        x=df_mm['date'], y=df_mm['cpu_mm'], mode='lines', name='CPU MM'))
    fig_trend.add_trace(go.Scatter(
        x=df_mm['date'], y=df_mm['mem_mm'], mode='lines', name='MEM MM'))
    fig_trend.update_layout(
        title='Tendência de Recursos (média móvel)', yaxis_title='Valor Médio')
    st.plotly_chart(fig_trend, use_container_width=True)

    with st.expander("🔍 Ver dados brutos"):
        st.dataframe(df, use_container_width=True)
        st.download_button("📥 Baixar CSV filtrado", data=df.to_csv(
            index=False), file_name="dados_filtrados.csv", mime="text/csv")

    last_date = df['date'].max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=future_days, freq='D')

    # Simulando valores futuros usando os últimos dias reais (ciclo)
    recent_data = original_df[variaveis_exp].tail(
        future_days).reset_index(drop=True)
    if len(recent_data) < future_days:
        # Repete se não tiver dias disponíveis
        repeat_count = (future_days // len(recent_data)) + 1
        future_X = pd.concat([recent_data] * repeat_count,
                             ignore_index=True).iloc[:future_days]
    else:
        future_X = recent_data

    future_size_pred = modelo_size.predict(future_X)
    future_cpu_pred = modelo_cpu.predict(future_X)
    future_mem_pred = modelo_mem.predict(future_X)

    # === Cenários otimistico/pessimista ===
    future_X_opt = future_X.copy()
    future_X_pess = future_X.copy()
    for col in future_X.columns:
        future_X_opt[col] = future_X[col] * 1.1  # +10%
        future_X_pess[col] = future_X[col] * 0.9  # -10%

    df_futuro = pd.DataFrame({
        'date': future_dates,
        'size_db_pred': future_size_pred,
        'cpu_pred': future_cpu_pred,
        'mem_pred': future_mem_pred,
        'size_db_opt': modelo_size.predict(future_X_opt),
        'cpu_opt': modelo_cpu.predict(future_X_opt),
        'mem_opt': modelo_mem.predict(future_X_opt),
        'size_db_pess': modelo_size.predict(future_X_pess),
        'cpu_pess': modelo_cpu.predict(future_X_pess),
        'mem_pess': modelo_mem.predict(future_X_pess)
    })

    # Projeções e erros
    df_pred = pd.DataFrame({
        'date': df['date'],
        'size_db_real': y_size,
        'size_db_pred': y_pred_size,
        'cpu_real': y_cpu,
        'cpu_pred': y_pred_cpu,
        'mem_real': y_mem,
        'mem_pred': y_pred_mem
    })

    df_pred['erro_size'] = df_pred['size_db_real'] - df_pred['size_db_pred']
    df_pred['erro_cpu'] = df_pred['cpu_real'] - df_pred['cpu_pred']
    df_pred['erro_mem'] = df_pred['mem_real'] - df_pred['mem_pred']

    rmse_size = mean_squared_error(
        df_pred['size_db_real'], df_pred['size_db_pred']) ** 0.5
    rmse_cpu = mean_squared_error(
        df_pred['cpu_real'], df_pred['cpu_pred']) ** 0.5
    rmse_mem = mean_squared_error(
        df_pred['mem_real'], df_pred['mem_pred']) ** 0.5

    mae_size = mean_absolute_error(
        df_pred['size_db_real'], df_pred['size_db_pred'])
    mae_cpu = mean_absolute_error(df_pred['cpu_real'], df_pred['cpu_pred'])
    mae_mem = mean_absolute_error(df_pred['mem_real'], df_pred['mem_pred'])

    df_pred['erro_total_abs'] = df_pred[[
        'erro_size', 'erro_cpu', 'erro_mem']].abs().sum(axis=1)
    maiores_erros = df_pred.nlargest(5, 'erro_total_abs')

with tab2:
    st.header("Modelagem Estatística")

    st.subheader("📈 Projeções com Regressão Polinomial")
    st.markdown(
        f"Projeção dos próximos {future_days} dias com base nas variáveis explicativas.")

    st.markdown("### 📊 Cenário Base")
    col_base1, col_base2 = st.columns(2)
    with col_base1:
        fig_base_db = go.Figure()
        fig_base_db.add_trace(go.Scatter(
            x=df_futuro['date'], y=df_futuro['size_db_pred'], mode='lines', name='Tamanho DB'))
        fig_base_db.update_layout(title='DB - Cenário Base', yaxis_title='TB')
        st.plotly_chart(fig_base_db, use_container_width=True)
    with col_base2:
        fig_base_cpu_mem = go.Figure()
        fig_base_cpu_mem.add_trace(go.Scatter(
            x=df_futuro['date'], y=df_futuro['cpu_pred'], mode='lines', name='CPU'))
        fig_base_cpu_mem.add_trace(go.Scatter(
            x=df_futuro['date'], y=df_futuro['mem_pred'], mode='lines', name='Memória'))
        fig_base_cpu_mem.update_layout(
            title='CPU / Memória - Cenário Base', yaxis_title='%')
        st.plotly_chart(fig_base_cpu_mem, use_container_width=True)

    st.markdown("### 🌟 Cenário Otimista vs Pessimista")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        fig_s_db = go.Figure()
        fig_s_db.add_trace(go.Scatter(
            x=df_futuro['date'], y=df_futuro['size_db_opt'], mode='lines', name='DB Otimista', line=dict(dash='dash')))
        fig_s_db.add_trace(go.Scatter(
            x=df_futuro['date'], y=df_futuro['size_db_pess'], mode='lines', name='DB Pessimista', line=dict(dash='dot')))
        fig_s_db.update_layout(
            title='DB - Cenários Alternativos', yaxis_title='TB')
        st.plotly_chart(fig_s_db, use_container_width=True)
    with col_s2:
        fig_s_cpu_mem = go.Figure()
        fig_s_cpu_mem.add_trace(go.Scatter(
            x=df_futuro['date'], y=df_futuro['cpu_opt'], mode='lines', name='CPU Otimista', line=dict(dash='dash')))
        fig_s_cpu_mem.add_trace(go.Scatter(
            x=df_futuro['date'], y=df_futuro['cpu_pess'], mode='lines', name='CPU Pessimista', line=dict(dash='dot')))
        fig_s_cpu_mem.add_trace(go.Scatter(
            x=df_futuro['date'], y=df_futuro['mem_opt'], mode='lines', name='Memória Otimista', line=dict(dash='dash')))
        fig_s_cpu_mem.add_trace(go.Scatter(
            x=df_futuro['date'], y=df_futuro['mem_pess'], mode='lines', name='Memória Pessimista', line=dict(dash='dot')))
        fig_s_cpu_mem.update_layout(
            title='CPU / Memória - Cenários Alternativos', yaxis_title='%')
        st.plotly_chart(fig_s_cpu_mem, use_container_width=True)

    with st.expander("📈 Ver/Exportar dados previstos"):
        def classifica_erro(valor):
            if valor < 1:
                return "🟢 Ótimo"
            elif valor < 3:
                return "🟡 Moderado"
            else:
                return "🔴 Ruim"

        col_e1, col_e2, col_e3 = st.columns(3)
        with col_e1:
            st.plotly_chart(px.histogram(df_pred, x='erro_size', nbins=30,
                            title='Erro Size TB'), use_container_width=True)
        with col_e2:
            st.plotly_chart(px.histogram(df_pred, x='erro_cpu',
                            nbins=30, title='Erro CPU'), use_container_width=True)
        with col_e3:
            st.plotly_chart(px.histogram(df_pred, x='erro_mem', nbins=30,
                            title='Erro Memória'), use_container_width=True)

        st.markdown("---")
        st.markdown("#### Erros das previsões (RMSE / MAE)")
        st.markdown(
            f"- Size TB: `RMSE={rmse_size:.4f}` ({classifica_erro(rmse_size)}), `MAE={mae_size:.4f}`")
        st.markdown(
            f"- CPU: `RMSE={rmse_cpu:.4f}` ({classifica_erro(rmse_cpu)}), `MAE={mae_cpu:.4f}`")
        st.markdown(
            f"- Memória: `RMSE={rmse_mem:.4f}` ({classifica_erro(rmse_mem)}), `MAE={mae_mem:.4f}`")

        st.markdown("##### Estatísticas adicionais dos erros")
        st.markdown(
            f"- Size TB → Max: `{df_pred['erro_size'].max():.4f}`, Min: `{df_pred['erro_size'].min():.4f}`, Std: `{df_pred['erro_size'].std():.4f}`")
        st.markdown(
            f"- CPU → Max: `{df_pred['erro_cpu'].max():.4f}`, Min: `{df_pred['erro_cpu'].min():.4f}`, Std: `{df_pred['erro_cpu'].std():.4f}`")
        st.markdown(
            f"- Memória → Max: `{df_pred['erro_mem'].max():.4f}`, Min: `{df_pred['erro_mem'].min():.4f}`, Std: `{df_pred['erro_mem'].std():.4f}`")

        st.markdown("---")
        st.markdown("### 🔎 Dias com maiores erros totais")
        st.markdown(
            "Esses dias representam os maiores desvios entre os valores reais e previstos. Eles podem indicar:")
        st.markdown(
            "- Atividades incomuns como picos de uso ou exclusões em massa")
        st.markdown(
            "- Mudanças comportamentais que o modelo ainda não capturou")
        st.markdown("- Necessidade de novos atributos explicativos")
        st.dataframe(maiores_erros[['date', 'erro_size', 'erro_cpu',
                     'erro_mem', 'erro_total_abs']], use_container_width=True)
        st.download_button("📥 Baixar dados previstos", data=df_pred.to_csv(
            index=False), file_name="projecoes.csv", mime="text/csv")

    def regressao_pol(x_vars, y_var, grau):
        X = original_df[x_vars]
        y = original_df[y_var].values
        modelo = make_pipeline(PolynomialFeatures(grau), LinearRegression())
        modelo.fit(X, y)
        r2 = modelo.score(X, y)
        coef = modelo.named_steps['linearregression'].coef_
        intercepto = modelo.named_steps['linearregression'].intercept_
        nomes_coef = list(PolynomialFeatures(grau).fit(
            df[x_vars]).get_feature_names_out(x_vars))
        return coef, r2, nomes_coef, intercepto

    st.subheader("Correlação entre variáveis")
    st.dataframe(df[['size_db', 'cpu', 'mem'] +
                 variaveis_exp].corr(), height=250)

    st.subheader(f"Regressão Polinomial (grau {grau_regressao})")
    for target in ['size_db', 'cpu', 'mem']:
        coef, r2, nomes, intercepto = regressao_pol(
            variaveis_exp, target, grau_regressao)
        df_coef = pd.DataFrame({
            'Variável': ['Intercepto'] + nomes,
            'Coeficiente': [intercepto] + coef.tolist()
        })
        st.markdown(f"**{target.upper()}** ~ Variáveis explicativas")
        st.markdown(f"R²: `{r2:.4f}`")
        st.dataframe(df_coef.sort_values(by='Coeficiente', key=abs,
                     ascending=False), use_container_width=True)
        st.download_button(f"📥 Baixar coeficientes de {target.upper()}", data=df_coef.to_csv(
            index=False), file_name=f"coef_{target}.csv", mime="text/csv")
        st.plotly_chart(go.Figure(go.Bar(
            x=df_coef['Coeficiente'],
            y=df_coef['Variável'],
            orientation='h'
        )).update_layout(title=f'Importância dos coeficientes para {target.upper()}', height=400), use_container_width=True)
        st.markdown("---")
