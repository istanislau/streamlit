import streamlit as st
import pandas as pd
import plotly.express as px

# ler o arquivo csv encoding='ISO-8859-1'
df = pd.read_csv('db_sim.csv', sep=',')

print(df.columns)
