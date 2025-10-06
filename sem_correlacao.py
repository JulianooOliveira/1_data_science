import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# === LEITURA DOS DADOS ===
df = pd.read_csv('./HousingData.csv')

print("=== VISUALIZAÇÃO INICIAL ===")
print(df.head(), "\n")

print("=== INFORMAÇÕES DO DATAFRAME ===")
print(df.info(), "\n")

print("=== ESTATÍSTICAS DESCRITIVAS ===")
print(df.describe(), "\n")

print("=== VALORES NULOS ===")
print(df.isnull().sum(), "\n")

# === ANÁLISE DE TIPOS ===
print("=== TIPOS DE DADOS ===")
print(df.dtypes, "\n")

# === VERIFICAÇÃO DE COLUNAS CATEGÓRICAS ===
cat_cols = df.select_dtypes(exclude=[np.number]).columns
if len(cat_cols) == 0:
    print("Não há colunas categóricas no dataset. Todas as variáveis são numéricas.\n")
else:
    print("Colunas categóricas encontradas:", list(cat_cols))
    for col in cat_cols:
        print(f"Moda de {col}: {df[col].mode(dropna=True).values[0]}")
    print()

# === TRATAMENTO DE VALORES AUSENTES ===
df = df.fillna(df.mean(numeric_only=True))

# === MATRIZ DE CORRELAÇÃO ===
print("=== MATRIZ DE CORRELAÇÃO (Pearson) ===")
corr = df.corr(numeric_only=True)
print(corr, "\n")

# === MAPA DE CALOR ===
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", center=0)
plt.title("Mapa de Correlação entre Variáveis Numéricas (Pearson)")
plt.tight_layout()
plt.show()

# === GRÁFICOS DE QUARTIS (BOXPLOTS) ===
print("=== ANÁLISE DE QUARTIS (BOXPLOTS) ===")
for col in df.select_dtypes(include=[np.number]).columns:
    plt.figure(figsize=(5,3))
    sns.boxplot(x=df[col], color='skyblue')
    plt.title(f'Boxplot — {col}')
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()
