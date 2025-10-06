import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configurações iniciais
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# === Leitura e Preparação dos Dados ===
df = pd.read_csv('./HousingData.csv')
# Tratando valores ausentes com a média
df = df.fillna(df.mean(numeric_only=True))

# Lista de colunas numéricas (excluindo MEDV para análise de concentração)
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if 'MEDV' in numeric_cols:
    numeric_cols.remove('MEDV')


# === 1. Análise de Concentração e Distribuição (Numéricas) ===

print("=== 1. CONCENTRAÇÃO: ASSIMETRIA (SKEWNESS) E CURTOSE (KURTOSIS) ===")
print("Assimetria:\n", df.skew(numeric_only=True), "\n")
print("Curtose:\n", df.kurt(numeric_only=True), "\n")

# Gráficos de Quartis (Box Plots)
plt.figure(figsize=(18, 12))
plt.suptitle('Gráficos de Quartis (Box Plots) para Colunas Numéricas', fontsize=16)

num_rows = int(np.ceil(len(numeric_cols) / 4))
for i, col in enumerate(numeric_cols):
    plt.subplot(num_rows, 4, i + 1)
    sns.boxplot(y=df[col])
    plt.title(col, fontsize=10)
    plt.ylabel('')
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig('quartile_boxplots.png')
plt.close()


# === 2. Análise da Moda (Categóricas/Discretas) ===

print("=== 2. MODA DAS COLUNAS CATEGÓRICAS/DISCRETAS ===")
# CHAS e RAD são as colunas discretas/categóricas mais relevantes para a moda.
discrete_cols = ['CHAS', 'RAD']
for col in discrete_cols:
    if col in df.columns:
        moda_val = df[col].mode().iloc[0]
        frequencia = df[col].value_counts().max()
        print(f"Moda de {col}: {moda_val} (Frequência: {frequencia})")
print("\n")


# === 3. Análise e Teste de Hipótese da Correlação (TODOS os Pares vs. MEDV) ===

print("=== 3. CORRELAÇÃO COMPLETA E TESTE DE HIPÓTESE (vs. MEDV) ===")
correlations = df.corr()['MEDV'].sort_values(ascending=False).drop('MEDV')
alpha = 0.05
top_cols_for_plot = ['RM', 'LSTAT', 'PTRATIO', 'INDUS'] 

# Relatório Numérico Detalhado (Valor, Direção, Força, P-Valor, Confirmação)
print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<30}".format("Variável", "R (Valor)", "Direção", "Força", "P-Valor", "Confirmação"))
print("-" * 75)

for col in correlations.index:
    r, p_value = stats.pearsonr(df['MEDV'], df[col])
    
    # Classificação da Força
    abs_r = abs(r)
    if abs_r >= 0.7: strength = "Forte"
    elif abs_r >= 0.5: strength = "Moderada"
    elif abs_r >= 0.3: strength = "Fraca"
    else: strength = "Muito Fraca"
    
    direction = "Positiva" if r > 0 else "Negativa"
    confirmation = f"Rejeita H0 (Correlacionado)" if p_value < alpha else "Não Rejeita H0 (Não Correlacionado)"
    
    print("{:<10} {:<10.4f} {:<10} {:<10} {:<10.6f} {:<30}".format(col, r, direction, strength, p_value, confirmation))

# Gráficos de dispersão para as correlações mais fortes/relevantes
plt.figure(figsize=(15, 10))
plt.suptitle('Gráficos de Dispersão das Principais Correlações com MEDV', fontsize=16)

for i, col in enumerate(top_cols_for_plot):
    plt.subplot(2, 2, i + 1)
    sns.regplot(x=df[col], y=df['MEDV'], scatter_kws={'alpha':0.6})
    r_val = correlations.loc[col]
    plt.title(f'{col} vs MEDV (r = {r_val:.2f})', fontsize=12)
    plt.xlabel(col)
    plt.ylabel('Valor Médio das Casas (MEDV)')
    
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig('correlation_regplots.png')
plt.close()


# === 4. Hipótese Comparativa (MEDV vs. CHAS) ===

print("\n=== 4. HIPÓTESE COMPARATIVA: MEDV vs. CHAS ===")
# Demonstração e teste: H0: MEDV_CHAS_1 = MEDV_CHAS_0 (Não há diferença de médias)

grupo_rio = df[df['CHAS'] == 1]['MEDV']
grupo_nao_rio = df[df['CHAS'] == 0]['MEDV']

# Teste T de Student
t_stat, p_hipotese = stats.ttest_ind(grupo_rio, grupo_nao_rio, equal_var=False)

print(f"Média Imóveis Rio (CHAS=1): ${grupo_rio.mean():.2f}")
print(f"Média Imóveis Não-Rio (CHAS=0): ${grupo_nao_rio.mean():.2f}")
print(f"P-Valor: {p_hipotese:.4f}")

if p_hipotese < alpha:
    conclusion = "Rejeita H₀: Há diferença significativa no valor médio dos imóveis."
else:
    conclusion = "Não Rejeita H₀: Não há evidência de diferença significativa no valor médio dos imóveis."
print(f"Conclusão (α={alpha}): {conclusion}\n")

# Visualização da Hipótese Comparativa
plt.figure(figsize=(6, 5))
sns.boxplot(x='CHAS', y='MEDV', data=df)
plt.title('Comparação de MEDV por Proximidade do Rio Charles (CHAS)', fontsize=12)
plt.xlabel('Faz Limite com Rio Charles (0=Não, 1=Sim)')
plt.ylabel('Valor Médio das Casas (MEDV)')
plt.savefig('comparative_hypothesis_boxplot.png')
plt.close()