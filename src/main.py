import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuração de estilo para os gráficos
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

df_prices = pd.read_csv('chip_prices.csv')
df_controls = pd.read_csv('export_controls.csv')
df_fab = pd.read_csv('fab_capacity.csv')
df_ai_market = pd.read_csv('ai_chip_market.csv')
df_fin = pd.read_csv('chip_companies_financials.csv')

df_ai_market.dropna(subset=['estimated_revenue_usd_m', 'estimated_shipments_units'], inplace=True)

# Agrupando a receita anual por empresa (Vendor)
ai_revenue_by_vendor = df_ai_market.groupby(['year', 'vendor'])['estimated_revenue_usd_m'].sum().reset_index()

df_gpu_prices = df_prices[df_prices['product'].str.contains('NVIDIA|AMD|GPU', case=False, na=False)].copy()
df_gpu_prices['year_month'] = pd.to_datetime(df_gpu_prices['year_month'])

target_companies = ['NVIDIA', 'AMD', 'TSMC']
df_fin_gpus = df_fin[df_fin['company_name'].isin(target_companies)].copy()
df_fin_gpus.fillna(0, inplace=True)

df_fab_leading = df_fab[(df_fab['company'] == 'TSMC') & (df_fab['fab_type'] == 'logic_leading')]
fab_capacity_yearly = df_fab_leading.groupby('year')['monthly_wafer_capacity'].sum().reset_index()

df_controls_high_severity = df_controls[df_controls['severity_score'] >= 7]
sactions_per_year = df_controls_high_severity.groupby('year')['control_id'].count().reset_index()
sactions_per_year.rename(columns={'control_id': 'num_sancoes_graves'}, inplace=True)

fig, axs = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Análise Exploratória: Mercado Global de GPUs e Chips de IA', fontsize=18, fontweight='bold')

# Gráfico 1: Evolução da Receita (Mercado de Chips de IA) por Empresa
for vendor in ai_revenue_by_vendor['vendor'].unique():
    subset = ai_revenue_by_vendor[ai_revenue_by_vendor['vendor'] == vendor]
    axs[0, 0].plot(subset['year'], subset['estimated_revenue_usd_m'], marker='o', label=vendor)
axs[0, 0].set_title('Receita Estimada de Chips de IA por Empresa (Milhões USD)')
axs[0, 0].set_ylabel('Receita (USD m)')
axs[0, 0].legend()

# Gráfico 2: Receita Total vs. Gastos em P&D (NVIDIA e AMD)
sns.scatterplot(data=df_fin_gpus[df_fin_gpus['company_name'].isin(['NVIDIA', 'AMD'])], 
                x='rd_spend_usd_bn', y='revenue_usd_bn', hue='company_name', size='year', sizes=(50, 200), ax=axs[0, 1])
axs[0, 1].set_title('Correlação: Gastos em P&D vs Receita Total (Bilhões USD)')
axs[0, 1].set_xlabel('Gastos em P&D (USD bn)')
axs[0, 1].set_ylabel('Receita Total (USD bn)')

# Gráfico 3: Preço Histórico de uma GPU Específica (ex: NVIDIA H100)
h100_prices = df_gpu_prices[df_gpu_prices['product'].str.contains('H100', na=False)]
if not h100_prices.empty:
    axs[1, 0].plot(h100_prices['year_month'], h100_prices['price'], color='purple', linewidth=2)
    axs[1, 0].set_title('Evolução do Preço Unitário: NVIDIA H100')
    axs[1, 0].set_ylabel('Preço (USD)')
    axs[1, 0].tick_params(axis='x', rotation=45)
else:
    axs[1, 0].text(0.5, 0.5, 'Dados da H100 não encontrados', ha='center', va='center')

# Gráfico 4: Impacto da Geopolítica vs Capacidade de Produção (TSMC)
ax2 = axs[1, 1].twinx()
axs[1, 1].bar(fab_capacity_yearly['year'], fab_capacity_yearly['monthly_wafer_capacity'], color='skyblue', label='Capacidade TSMC (Wafers)', alpha=0.7)
ax2.plot(sactions_per_year['year'], sactions_per_year['num_sancoes_graves'], color='red', marker='x', linewidth=2, label='Sanções Graves')
axs[1, 1].set_title('Capacidade de Produção (TSMC) vs Sanções Geopolíticas')
axs[1, 1].set_ylabel('Capacidade Mensal de Wafers (Nós de Ponta)')
ax2.set_ylabel('Número de Sanções (Score >= 7)')

# Ajustar layout para não sobrepor textos
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("--- Estatísticas Descritivas: Chips de IA ---")
print(df_ai_market[['estimated_asp_usd', 'estimated_revenue_usd_m']].describe())

print("\n--- Receita Média (Financeiro) por Empresa ---")
print(df_fin_gpus.groupby('company_name')['revenue_usd_bn'].mean())