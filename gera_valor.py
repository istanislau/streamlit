'''' 
import random
import csv

# Lista de números permitidos
numeros = [1, 15, 30, 90]

# Gerando a lista aleatória com 654 valores
lista_aleatoria = [random.choice(numeros) for _ in range(654)]

# Salvando a lista em um arquivo CSV
with open('lista_aleatoria.csv', 'w', newline='') as arquivo_csv:
    escritor = csv.writer(arquivo_csv)
    escritor.writerow(['Valores'])  # Cabeçalho da coluna
    for valor in lista_aleatoria:
        escritor.writerow([valor])

print("Arquivo CSV gerado com sucesso: lista_aleatoria.csv")
'''
import random
import csv
from datetime import datetime, timedelta

# Função para gerar valores aleatórios com base no tipo de dado


def gerar_valor(tipo):
    if tipo == 'int':
        return random.randint(0, 100)  # Números inteiros entre 0 e 100
    elif tipo == 'float':
        # Números decimais entre 0 e 100
        return round(random.uniform(0, 100), 2)
    elif tipo == 'date':
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2023, 12, 31)
        random_date = start_date + \
            timedelta(days=random.randint(0, (end_date - start_date).days))
        return random_date.strftime('%d/%m/%Y')  # Datas no formato YYYY-MM-DD
    elif tipo == 'string':
        # Strings aleatórias
        return ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=5))
    else:
        raise ValueError(f"Tipo de dado não suportado: {tipo}")

# Função principal para gerar o CSV


def gerar_csv(nome_arquivo, linhas, colunas, nomes_colunas, tipos_colunas):
    # Verifica se o número de colunas e tipos corresponde aos nomes das colunas
    if len(nomes_colunas) != len(tipos_colunas):
        raise ValueError(
            "O número de nomes de colunas deve ser igual ao número de tipos de dados.")

    # Gera os dados
    dados = []
    for _ in range(linhas):
        linha = [gerar_valor(tipo) for tipo in tipos_colunas]
        dados.append(linha)

    # Salva os dados em um arquivo CSV
    with open(nome_arquivo, 'w', newline='', encoding='utf-8') as arquivo_csv:
        escritor = csv.writer(arquivo_csv)
        escritor.writerow(nomes_colunas)  # Escreve o cabeçalho
        escritor.writerows(dados)  # Escreve as linhas de dados

    print(f"Arquivo CSV gerado com sucesso: {nome_arquivo}")


# Exemplo de uso
if __name__ == "__main__":
    # Defina os parâmetros
    nome_arquivo = 'dados_gerados.csv'
    linhas = 400  # Quantidade de linhas
    colunas = 5  # Quantidade de colunas
    nomes_colunas = ['date', 'size_db', 'cpu',
                     'mem', "fundos"]  # Nomes das colunas
    # Tipos de dados para cada coluna
    tipos_colunas = ['date', 'int', 'float', 'float', "int"]

    # Gera o CSV
    gerar_csv(nome_arquivo, linhas, colunas, nomes_colunas, tipos_colunas)
