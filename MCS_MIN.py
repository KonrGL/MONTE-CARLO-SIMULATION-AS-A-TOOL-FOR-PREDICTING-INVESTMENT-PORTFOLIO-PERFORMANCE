import pandas as pd

# Dane dla oryginalnego scenariusza
original_data = {
    "Metric": [
        "EPS", "BVPS", "ROE", "DPR", "Retention Ratio (b)", "Growth Rate (g)", "Cost of Equity (r_e)",
        "Forecasted Dividend (D1)", "Stock Price (P0)", "Market Equity (E)", "Market Debt (Bonds)",
        "Cost of Debt (Bonds) after tax", "Cost of Debt (Bank) after tax", "Total Debt (D)",
        "Weight of Equity (w_E)", "Weight of Debt (w_D)", "Average Cost of Debt (k_d_avg)",
        "WACC", "Debt/Equity Ratio (D/E)", "NOPAT", "EVA"
    ],
    "Value": [
        4.8, 20.0, 0.24, 0.40, 0.60, 0.144, 0.1725,
        2.197, 77.1, 771, 196, 0.1008, 0.12, 296,
        0.7225, 0.2775, 0.1068, 0.1543, 0.384, 64, -13.15
    ]
}

# Dane dla scenariusza z DPR = 90%
new_dpr_data = {
    "Metric": [
        "DPR (new)", "Retention Ratio (b) new",
        "Growth Rate (g) new", "Forecasted Dividend (D1) new",
        "Stock Price (P0) new"
    ],
    "Value": [0.90, 0.10, 0.024, 4.32, 29.1]
}

# Tworzenie DataFrame’ów
df_original = pd.DataFrame(original_data)
df_new_dpr = pd.DataFrame(new_dpr_data)

# Wydruk w formacie CSV (kopiuj do Excela)
print("Original Scenario Metrics")
print(df_original.to_csv(index=False))
print("\nNew DPR = 90% Scenario Metrics")
print(df_new_dpr.to_csv(index=False))
