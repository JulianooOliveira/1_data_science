import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

# === Leitura do arquivo ===
df = pd.read_csv('./HousingData.csv')

# Tratando valores ausentes
df = df.fillna(df.mean(numeric_only=True))

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Variáveis para o estudo
x = df['RM']      # Número médio de quartos
y = df['MEDV']    # Valor médio das casas

print("=== CORRELAÇÃO POSITIVA ENTRE RM E MEDV ===")
print(df[['RM', 'MEDV']].head(), "\n")

# === Coeficiente de Pearson ===
corr_pearson, p_value = stats.pearsonr(x, y)
r2 = corr_pearson ** 2

print(f"Coeficiente de Pearson (r): {corr_pearson:.4f}")
print(f"Coeficiente de Determinação (R²): {r2:.4f}")
print(f"Valor-p: {p_value:.10f}")
print(f"Direção: {'Positiva' if corr_pearson > 0 else 'Negativa'}")
print(f"Força: {'Forte' if abs(corr_pearson) > 0.7 else 'Moderada' if abs(corr_pearson) > 0.3 else 'Fraca'}")

# === Visualização ===
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(x, y, alpha=0.7, color='green')
plt.xlabel('Número Médio de Quartos (RM)')
plt.ylabel('Valor Médio das Casas (MEDV)')
plt.title(f'Dispersão — Correlação Positiva (r={corr_pearson:.3f})')

plt.subplot(1, 2, 2)
sns.regplot(x=x, y=y, ci=None, color='green', line_kws={'color': 'black'})
plt.title(f'Regressão Linear — r={corr_pearson:.2f}, p={p_value:.4f}')

plt.tight_layout()
plt.show()

# === Teste de Hipótese ===
alpha = 0.05
print(f"\n=== TESTE DE HIPÓTESE (α = {alpha}) ===")
print("H₀: Não há correlação linear entre RM e MEDV (ρ = 0)")
print("H₁: Há correlação linear entre RM e MEDV (ρ ≠ 0)")
print(f"Valor-p: {p_value:.6f}")

if p_value < alpha:
    print("DECISÃO: Rejeitamos H₀ — Há evidências de correlação linear positiva significativa.")
else:
    print("DECISÃO: Não rejeitamos H₀ — Não há evidências suficientes de correlação linear.")
