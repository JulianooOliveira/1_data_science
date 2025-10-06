# === estatisticas_descritivas.py ===
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# === LEITURA DOS DADOS ===
df = pd.read_csv('./HousingData.csv')

# === TRATAMENTO DE VALORES AUSENTES ===
df = df.fillna(df.mean(numeric_only=True))

# === SELECIONANDO APENAS AS COLUNAS NUMÉRICAS ===
numeric_df = df.select_dtypes(include=[np.number])

# === CÁLCULO DAS MEDIDAS DESCRITIVAS ===
print("=== MEDIDAS DESCRITIVAS ===\n")

# Média
mean_values = numeric_df.mean()
print("Média:\n", mean_values, "\n")

# Mediana
median_values = numeric_df.median()
print("Mediana:\n", median_values, "\n")

# Moda (pode haver múltiplas)
mode_values = numeric_df.mode().iloc[0]
print("Moda:\n", mode_values, "\n")

# Desvio padrão
std_values = numeric_df.std()
print("Desvio Padrão:\n", std_values, "\n")

# === TABELA RESUMIDA ===
summary = pd.DataFrame({
    'Média': mean_values,
    'Mediana': median_values,
    'Moda': mode_values,
    'Desvio Padrão': std_values
})

print("=== RESUMO GERAL ===")
print(summary.round(3))

# === SALVAR RESULTADOS EM CSV ===
summary.to_csv('estatisticas_descritivas.csv', index=True, sep=';', decimal=',')
print("\nArquivo 'estatisticas_descritivas.csv' gerado com sucesso!")
