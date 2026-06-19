import pandas as pd
import numpy as np
from scipy.optimize import minimize

#file path

file_path = input("Podaj pełną ścieżkę do pliku Excel (z \\ zamiast /): ")
sheet_name = "LOGRORS12"  # log rors


def load_data(path, sheet):
    df = pd.read_excel(path, sheet_name=sheet)
    df = df.dropna()  # usuń wiersze z brakami
    return df

#MPT

def portfolio_performance(weights, mean_returns, cov_matrix):
    port_return = np.dot(weights, mean_returns)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return port_return, port_volatility

def negative_sharpe_ratio(weights, mean_returns, cov_matrix):
    ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
    return -ret / vol  # risk-free rate = 0

#Portfolio Optimalization

def optimize_portfolio(mean_returns, cov_matrix, method):
    num_assets = len(mean_returns)
    initial_weights = np.ones(num_assets) / num_assets
    bounds = tuple((0, 1) for _ in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    if method == "min_var":
        objective = lambda w: portfolio_performance(w, mean_returns, cov_matrix)[1]
    elif method == "max_ret":
        objective = lambda w: -np.dot(w, mean_returns)
    elif method == "max_sharpe":
        objective = lambda w: negative_sharpe_ratio(w, mean_returns, cov_matrix)
    else:
        raise ValueError("Nieznana metoda optymalizacji")

    result = minimize(objective, initial_weights, method='SLSQP',
                      bounds=bounds, constraints=constraints)

    if not result.success:
        raise Exception("Błąd optymalizacji:", result.message)

    return result.x

#Printing

def print_portfolio(title, tickers, weights, mean_returns, cov_matrix):
    ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
    sharpe = ret / vol if vol != 0 else 0
    print(f"\n📊 {title}")
    print("-" * 40)
    for t, w in zip(tickers, weights):
        if w > 0.01:
            print(f"{t}: {w:.2%}")
    print(f"✅ Oczekiwany zwrot: {ret:.4f}")
    print(f"📈 Ryzyko (odchylenie): {vol:.4f}")
    print(f"📊 Wskaźnik Sharpe'a: {sharpe:.4f}")


def main():
    data = load_data(file_path, sheet_name)
    tickers = data.columns
    mean_returns = data.mean()
    cov_matrix = data.cov()

    # Min
    w_min_var = optimize_portfolio(mean_returns, cov_matrix, method="min_var")
    print_portfolio("Minimum Variance Portfolio", tickers, w_min_var, mean_returns, cov_matrix)

    # Max
    w_max_ret = optimize_portfolio(mean_returns, cov_matrix, method="max_ret")
    print_portfolio("Maximum Return Portfolio", tickers, w_max_ret, mean_returns, cov_matrix)

    # Opt "Sharpe"
    w_sharpe = optimize_portfolio(mean_returns, cov_matrix, method="max_sharpe")
    print_portfolio("Optimal Sharpe Portfolio", tickers, w_sharpe, mean_returns, cov_matrix)

if __name__ == "__main__":
    main()
