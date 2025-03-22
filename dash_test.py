import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
from sklearn.linear_model import LinearRegression
import numpy as np
# teste push


def resolve_path(path):
    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(sys.executable)
    elif '__file__' in globals():
        base_path = os.path.dirname(os.path.abspath(__file__))
    else:
        base_path = os.getcwd()

    return os.path.abspath(os.path.join(base_path, path))


st.set_page_config(
    page_title="Dashboard de Recursos",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Leitura e preparo do dataset
df = pd.read_csv(resolve_path('db_sim.csv'), sep=',', encoding='utf-8')


df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
df['size_tb'] = (pd.to_numeric(df['size'], errors='coerce') / 1024).round(3)
df['cpu'] = df['cpu'] / 100
df['mem'] = df['mem'] / 100

# Cópia dos dados originais para uso nas regressões
original_df = df.copy()

# Sidebar - Filtro por data
st.sidebar.header("Filtros")
usar_filtro = st.sidebar.checkbox("Filtrar por intervalo de datas")

if usar_filtro:
    data_min = df['date'].min().date()
    data_max = df['date'].max().date()

    col1, col2 = st.sidebar.columns(2)
    data_inicial = col1.date_input(
        "Data inicial", value=data_min, min_value=data_min, max_value=data_max)
    data_final = col2.date_input(
        "Data final", value=data_max, min_value=data_min, max_value=data_max)

    if data_inicial and data_final:
        df = df[(df['date'] >= pd.to_datetime(data_inicial))
                & (df['date'] <= pd.to_datetime(data_final))]

    data_range = st.sidebar.slider(
        "Intervalo de datas",
        min_value=data_min,
        max_value=data_max,
        value=(data_min, data_max)
    )

    df = df[(df['date'] >= pd.to_datetime(data_range[0])) &
            (df['date'] <= pd.to_datetime(data_range[1]))]

# Texto formatado para rótulos
df['cpu_str'] = df['cpu'].map(lambda x: f"{x * 100:.1f}%")
df['mem_str'] = df['mem'].map(lambda x: f"{x * 100:.1f}%")

# Suaviza as curvas com média móvel
df['cpu_smooth'] = df['cpu'].rolling(window=7, min_periods=1).mean()
df['mem_smooth'] = df['mem'].rolling(window=7, min_periods=1).mean()

# Métricas de destaque
col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)
max_cpu = df.loc[df['cpu'].idxmax()]
min_cpu = df.loc[df['cpu'].idxmin()]
max_mem = df.loc[df['mem'].idxmax()]
min_mem = df.loc[df['mem'].idxmin()]
max_db = df.loc[df['size_tb'].idxmax()]
min_db = df.loc[df['size_tb'].idxmin()]

col1.metric("CPU Máxima", f"{max_cpu['cpu'] * 100:.1f}%",
            help=f"Data: {max_cpu['date'].date()}")
col4.metric("CPU Mínima", f"{min_cpu['cpu'] * 100:.1f}%",
            help=f"Data: {min_cpu['date'].date()}")
col2.metric("Memória Máxima",
            f"{max_mem['mem'] * 100:.1f}%", help=f"Data: {max_mem['date'].date()}")
col5.metric("Memória Mínima",
            f"{min_mem['mem'] * 100:.1f}%", help=f"Data: {min_mem['date'].date()}")
col3.metric("Tamanho Máximo DB",
            f"{max_db['size_tb']:.3f} TB", help=f"Data: {max_db['date'].date()}")
col6.metric("Tamanho Mínimo DB",
            f"{min_db['size_tb']:.3f} TB", help=f"Data: {min_db['date'].date()}")

# Regressão linear para projeção do tamanho da base

# Dados e modelo para tamanho da base
df_reg = original_df.dropna(subset=['date', 'size_tb']).copy().copy()
df_reg['timestamp'] = df_reg['date'].astype('int64') // 10**9
X_db = df_reg['timestamp'].values.reshape(-1, 1)
y_db = df_reg['size_tb'].values
if len(np.unique(y_db)) > 1:
    model_db = LinearRegression().fit(X_db, y_db)
else:
    model_db = LinearRegression()
    st.sidebar.warning("Regressão DB: dados sem variação suficiente.")

# Dados e modelo para CPU
df_reg_cpu = original_df.dropna(subset=['date', 'cpu']).copy().copy()
df_reg_cpu['timestamp'] = df_reg_cpu['date'].astype('int64') // 10**9
X_cpu = df_reg_cpu['timestamp'].values.reshape(-1, 1)
y_cpu = df_reg_cpu['cpu'].values
if len(np.unique(y_cpu)) > 1:
    model_cpu = LinearRegression().fit(X_cpu, y_cpu)
else:
    model_cpu = LinearRegression()
    st.sidebar.warning("Regressão CPU: dados sem variação suficiente.")

# Dados e modelo para Memória
df_reg_mem = original_df.dropna(subset=['date', 'mem']).copy().copy()
df_reg_mem['timestamp'] = df_reg_mem['date'].astype('int64') // 10**9
X_mem = df_reg_mem['timestamp'].values.reshape(-1, 1)
y_mem = df_reg_mem['mem'].values
if len(np.unique(y_mem)) > 1:
    model_mem = LinearRegression().fit(X_mem, y_mem)
else:
    model_mem = LinearRegression()
    st.sidebar.warning("Regressão MEM: dados sem variação suficiente.")
st.sidebar.divider()
# Exibe os coeficientes dos modelos
# Correlação simples
st.sidebar.markdown("### Correlação entre Variáveis")
correlacoes = original_df[['size_tb', 'cpu', 'mem']].corr()
st.sidebar.dataframe(correlacoes, height=150)

# Regressão múltipla: CPU ~ size_tb
X_cpu_db = original_df[['size_tb']]
y_cpu_db = original_df['cpu']
model_cpu_db = LinearRegression().fit(X_cpu_db, y_cpu_db)

# Regressão múltipla: MEM ~ size_tb
X_mem_db = original_df[['size_tb']]
y_mem_db = original_df['mem']
model_mem_db = LinearRegression().fit(X_mem_db, y_mem_db)

st.sidebar.markdown("### Regressão CPU e MEM com base no Tamanho da DB")
st.sidebar.markdown(f"**CPU ~ DB coef:** {model_cpu_db.coef_[0]:.6f}")
st.sidebar.markdown(
    f"**R² CPU~DB:** {model_cpu_db.score(X_cpu_db, y_cpu_db):.4f}")
st.sidebar.markdown(f"**MEM ~ DB coef:** {model_mem_db.coef_[0]:.6f}")
st.sidebar.markdown(
    f"**R² MEM~DB:** {model_mem_db.score(X_mem_db, y_mem_db):.4f}")

st.sidebar.divider()
st.sidebar.markdown("### Estatísticas da Regressão")

if hasattr(model_db, 'coef_') and model_db.coef_[0] != 0:
    st.sidebar.markdown(
        f"**Crescimento DB:** {model_db.score(X_db, y_db):.4f}")
else:
    st.sidebar.markdown("**Crescimento DB:** Dados insuficientes")

if hasattr(model_cpu, 'coef_') and model_cpu.coef_[0] != 0:
    st.sidebar.markdown(
        f"**Crescimento CPU:** {model_cpu.score(X_cpu, y_cpu):.4f}")
else:
    st.sidebar.markdown("**Crescimento CPU:** Dados insuficientes")

if hasattr(model_mem, 'coef_') and model_mem.coef_[0] != 0:
    st.sidebar.markdown(
        f"**Crescimento MEM:** {model_mem.score(X_mem, y_mem):.4f}")
else:
    st.sidebar.markdown("**Crescimento MEM:** Dados insuficientes")

# Gerar datas futuras para projeção
future_days = st.sidebar.slider("Projeção em dias", 30, 365, 90)
future_dates = pd.date_range(
    df['date'].max() + pd.Timedelta(days=1), periods=future_days)
future_timestamps = (future_dates.astype('int64') // 10**9).to_numpy()
predicted_sizes = model_db.predict(future_timestamps.reshape(-1, 1))

# Gráfico de Tamanho da Base com projeção
fig_size = go.Figure()
fig_size.add_trace(go.Scatter(
    x=df['date'], y=df['size_tb'], mode='lines+markers', name='Tamanho Real'))
fig_size.add_trace(go.Scatter(
    x=future_dates, y=predicted_sizes, mode='lines', name='Projeção'))
fig_size.update_layout(
    title='Tamanho da Base de Dados com Projeção (TB)', yaxis_title='Tamanho (TB)')

# Projeções futuras para CPU e MEM
predicted_cpu = model_cpu.predict(future_timestamps.reshape(-1, 1))
predicted_mem = model_mem.predict(future_timestamps.reshape(-1, 1))

# Gráfico de CPU e Memória
fig_usage = go.Figure()
fig_usage.add_trace(go.Scatter(
    x=df['date'], y=df['cpu_smooth'], mode='lines', name='CPU Real'))
fig_usage.add_trace(go.Scatter(
    x=df['date'], y=df['mem_smooth'], mode='lines', name='Memória Real'))
fig_usage.add_trace(go.Scatter(x=future_dates, y=predicted_cpu,
                    mode='lines', name='CPU Projeção', line=dict(dash='dot')))
fig_usage.add_trace(go.Scatter(x=future_dates, y=predicted_mem,
                    mode='lines', name='Memória Projeção', line=dict(dash='dot')))
fig_usage.update_layout(title='Uso de CPU e Memória com Projeção',
                        yaxis_tickformat=".0%", yaxis_title="Uso (%)")
fig_usage.update_layout(yaxis_tickformat=".0%", yaxis_title="Uso (%)")

# Layout com colunas para os gráficos
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig_usage, use_container_width=True)
with col2:
    st.plotly_chart(fig_size, use_container_width=True)

# Exibição da tabela de dados
st.markdown("### Dados filtrados")
st.dataframe(df[['date', 'size_tb', 'cpu_str', 'mem_str']],
             use_container_width=True)
