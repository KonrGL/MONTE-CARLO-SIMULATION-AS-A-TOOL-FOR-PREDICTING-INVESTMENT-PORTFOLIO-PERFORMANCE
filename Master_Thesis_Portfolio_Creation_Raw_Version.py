import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from pypfopt import EfficientFrontier, risk_models, expected_returns


#Loading the data and cleaning it
file_path = "C:/Users/lucif/Desktop/Studia/Freewheeling/Dane_Do_Pythona.xlsx"
df = pd.read_excel(file_path, header=1)
df.columns = df.columns.str.strip()

#########################################################################################
log_ror_columns = [
    "Log_RoR_TXN", "Log_RoR_RTX", "Log_RoR_RGR", "Log_RoR_LHX",
    "Log_RoR_BALL", "Log_RoR_HON", "Log_RoR_AXON", "Log_RoR_HII", "Log_RoR_LMT"
]

df_cleaned = df.dropna(subset=log_ror_columns)
log_rors = df_cleaned[log_ror_columns]
#########################################################################################

# Printing the Log RoR values
Log_RoR_TXN = df_cleaned["Log_RoR_TXN"].values
Log_RoR_RTX = df_cleaned["Log_RoR_RTX"].values
Log_RoR_RGR = df_cleaned["Log_RoR_RGR"].values
Log_RoR_LHX = df_cleaned["Log_RoR_LHX"].values
Log_RoR_BALL = df_cleaned["Log_RoR_BALL"].values
Log_RoR_HON = df_cleaned["Log_RoR_HON"].values
Log_RoR_AXON = df_cleaned["Log_RoR_AXON"].values
Log_RoR_HII = df_cleaned["Log_RoR_HII"].values
Log_RoR_LMT = df_cleaned["Log_RoR_LMT"].values

print(Log_RoR_TXN)
print(Log_RoR_RTX)
print(Log_RoR_RGR)
print(Log_RoR_LHX)
print(Log_RoR_BALL)
print(Log_RoR_HON)
print(Log_RoR_AXON)
print(Log_RoR_HII)
print(Log_RoR_LMT)

#Plotting Logarythmic Returns
plt.figure(figsize=(14,8))
for column in log_ror_columns:
    plt.plot(df_cleaned.index, df_cleaned[column], label=column, linewidth=1)
plt.title('Daily Logarithmic Returns of 9 Stocks (5 Years)', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Log Return', fontsize=12)
plt.legend(title='Stocks', loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

ann_factor = 252
expected_returns = log_rors.mean()*ann_factor
covar_matrix = log_rors.cov() * ann_factor
print("Annualized Covariance Matrix:")
print(covar_matrix.round(2))
plt.figure(figsize=(12,10))
sns.heatmap(covar_matrix,annot=True,cmap="coolwarm", fmt=".4f", linewidths=0.6)
plt.title("Annualized Covariance Matrix Heatmap")
plt.show()


##############################################################################################
#Splitting the portfolio and creating scenarios

#1. Defining returns for portfolio
def portfolio_returns(weights,covar_matrix):
    return sum(weights*covar_matrix)
#2. Defining portfolio voatility
def portfolio_volatility(weights,covar_matrix):
    variance = weights @ covar_matrix @ weights 
    return np.sqrt(variance)
#3. Defining sharpe ratio
def sharpe_ratio(weights,expected_returns,covar_matrix,risk_free_rate):
    return (portfolio_returns(weights,expected_returns)-risk_free_rate)/portfolio_volatility(weights,covar_matrix)
#Sharpe Ratio = Portfolio Return - Risk-Free Rate/Portfolio Volatility

#Defining Parameters and stuff
rfr = 0.0455 #Risk-Free Rate na rynek Amerykański (bo stocki są z USA to obecnie 4.55%)
number_of_assets = len(log_ror_columns)
bounds = [(0,0.3)]*number_of_assets



constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}] #suma akcji w portfolio = 1
initial_weights = np.array([1/number_of_assets]*number_of_assets)
#ustawiam punkt startowy dla kodu, mianowice przy ilości 9 to 11.11(1)% dla każdej spółki
#dzięki temu kod ma punt odniesienia do dalszych przekształceń

#1. Scenariusz Maksymalne Ryzyko - Maksymalne Zwroty

def negative_portfolio_return(weights, expected_returns):
    return -portfolio_returns(weights,expected_returns)

max_risk_result = minimize(
    negative_portfolio_return, initial_weights, args=(expected_returns,),
    method="SLSQP", bounds = bounds, constraints=constraints
)
max_risk_return_weights = max_risk_result.x

#2. Optymalne Portfolio (Balans pomiędzy ryzkiem a zwrotem, dla tych co się boją)

def neg_sharpe_ratio(weights, expected_returns, covar_matrix, rfr):
    return -sharpe_ratio(weights, expected_returns, covar_matrix, rfr)

optimal_result = minimize(
    neg_sharpe_ratio, initial_weights, args=(expected_returns, covar_matrix, rfr),
    method='SLSQP', bounds=bounds, constraints=constraints
)
optimal_weights = optimal_result.x

#3. Minimalne Ryzyko i Minimalna Nagroda (Dla tych co nigdy w pokera nie grali)

min_risk_result = minimize(
    portfolio_volatility, initial_weights, args=(covar_matrix,),
    method='SLSQP', bounds=bounds, constraints=constraints
)
min_risk_weights = min_risk_result.x

#Wyniki 3 scenariuszy dla danych akcji

scenarios = [
    ("Maximum Risk & Maximum Return",max_risk_return_weights),("Optimal Risk & Optimal Returns",optimal_weights),("Minimal Risk & Minimal Returns",min_risk_weights)
]

for name, weights in scenarios:
    ret = portfolio_returns(weights, expected_returns)
    vol = portfolio_volatility(weights, covar_matrix)
    sr = sharpe_ratio(weights, expected_returns, covar_matrix, rfr)
    print(f"\n{name} Portfolio:")
    for ticker, weight in zip(log_ror_columns, weights):
        print(f"{ticker}: {weight:.4f}")
    print(f"Expected Return: {ret:.4f}")
    print(f"Volatility: {vol:.4f}")
    print(f"Sharpe Ratio: {sr:.4f}")

# Visualization
plt.figure(figsize=(14, 8))
bar_width = 0.25
x = np.arange(number_of_assets)

plt.bar(x - bar_width, max_risk_return_weights, bar_width, label='Max Risk & Return', color='red')
plt.bar(x, optimal_weights, bar_width, label='Optimal (Balanced)', color='purple')
plt.bar(x + bar_width, min_risk_weights, bar_width, label='Min Risk & Return', color='blue')

plt.xticks(x, log_ror_columns, rotation=45)
plt.xlabel('Stocks')
plt.ylabel('Weights')
plt.title('Portfolio Weights Across Scenarios (Max 30% per Stock)')
plt.legend()
plt.tight_layout()
plt.show()
