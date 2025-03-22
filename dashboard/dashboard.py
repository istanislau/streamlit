import streamlit as st
import pandas as pd
import plotly.express as px


st.set_page_config(layout="wide")

df = pd.read_csv("supermarket_sales.csv", sep=";", decimal=",")

# Perguntas
# média de preço por distrito, cidade
# preço mais caro em cada distrito
# preço mais barato em cada distrito

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

df["Month"] = df["Date"].apply(lambda x: str(x.year) + "-" + str(x.month))

month = st.sidebar.selectbox("Mes", df["Month"].unique())

df_filtred = df[df["Month"] == month]
df_filtred

col1, col2 = st.columns(2)
col3, col4, col5 = st.columns(3)

fig_date = px.bar(df_filtred, x="Date", y="Total",
                  title="Faturamento por dia", color="City")
fig_prod = px.bar(df_filtred, x="Date", y="Product line",
                  title="Faturamento por Produto", color="City", orientation="h")

city_total = df_filtred.groupby("City")[["Total"]].sum().reset_index()
fig_city = px.bar(df_filtred, x="City", y="Total",
                  title="Faturamento por filial")


col1.plotly_chart(fig_date)
col2.plotly_chart(fig_prod)
col3.plotly_chart(fig_city)
