import numpy as np
import matplotlib.pyplot as plt
import random

# parameters
mu = 0.0020259
sigma = 0.01585
n_days = 252
n_simulations = 10000
S_0 = 27779.2094
n_paths_to_plot = 1000

# real trajectory
actual_trajectory = [
    27761.3484,
    28356.2654,
    28421.5932,
    28943.4121,
    29064.1269,
    29481.9553,
    28732.8182,
    28873.0684,
    28170.7682,
    28467.2796,
    28579.5233,
    28721.6559,
    28620.5663,
    28898.3896,
    28521.6552,
    28358.6203,
    28207.2236,
    28211.9699,
    30123.6187,
    30497.1672,
    31380.3617,
    31924.5638,
    31446.157,
    31693.9929,
    31738.2726,
    31490.9145,
    32231.5406,
    31983.9715,
    33078.5437,
    33204.85,
    33651.1481,
    33646.2392,
    33486.0772,
    32935.5061,
    33666.4744,
    34682.2204,
    35790.9976,
    35406.652,
    35285.3761,
    37584.094,
    38741.931,
    38199.8966,
    38856.3451,
    39499.4499,
    39492.1672,
    39285.9449,
    40393.7216,
    42461.9541,
    42078.8251,
    44284.6766,
    44137.9482,
    43123.7611,
    43101.136,
    43438.3874,
    44198.3854,
    43444.5277,
    43079.1337,
    42897.4855,
    44480.285,
    43815.4413,
    43226.5706,
    42165.1948,
    42751.2986,
    42530.9328,
    43110.5825,
    42446.6345,
    42755.9731,
    40814.5792,
    41414.4301,
    41427.2084,
    39533.0841,
    40079.2258,
    40949.1482,
    40483.6006,
    41395.0269,
    42364.9356,
    43002.0452,
    42383.1542,
    44423.4597,
    44515.0617,
    44414.7746,
    43292.686,
    45849.7045,
    45625.1863,
    45928.3339,
    47156.5342,
    46877.6306,
    48493.1236,
    48446.402,
    49929.563,
    50543.0586,
    50968.8242,
    51651.1572,
    50423.0704,
    52026.3794,
    52137.8156,
    51751.342,
    52150.173,
    52006.2132,
    50492.5175,
    50300.4965,
    50892.387,
    49939.575,
    50453.035,
    50641.6985,
    50808.7625,
    50457.159,
    49935.8785,
    52687.926,
    51732.427,
    51586.901,
    52877.636,
    53105.836,
    53897.399,
    53096.032,
    52902.618,
    52699.056,
    52175.932,
    51722.376,
    53654.878,
    55139.193,
    54824.744,
    53573.303,
    53786.262,
    53393.251,
    53084.853,
    53312.697,
    53741.997,
    53916.023,
    53630.66,
    54655.493,
    51710.21,
    50211.786,
    49069.81,
    49720.878,
    49110.461,
    50140.857,
    49628.523,
    50434.177,
    50488.715,
    51598.269,
    49204.679,
    49417.729,
    50288.487,
    49766.789,
    48466.4295,
    46660.0215,
    46922.9295,
    46413.7925,
    46304.1535,
    46793.4695,
    47749.8905,
    48507.05,
    48794.675,
    50200.592,
    50133.515,
    49426.014,
    49478.373,
    50170.007,
    51062.7945,
    50295.357,
    51262.373,
    51795.087,
    52094.178,
    52511.11,
    54630.734,
    54041.857,
    52701.924,
    51308.2365,
    51212.4,
    51861.6535,
    50995.3465,
    50719.904,
    52002.3715,
    52266.863,
    51585.2715,
    53815.731,
    54353.438,
    54179.225,
    54115.587,
    54533.6895,
    54106.084,
    55867.202,
    56232.232,
    56717.031,
    56871.805,
    56300.959,
    55785.145,
    56123.367,
    55136.842,
    53470.331,
    54795.407,
    56578.446,
    57966.982,
    57803.634,
    57977.292,
    57529.783,
    57067.273,
    55603.037,
    56450.316,
    57505.842,
    56973.287,
    57087.719,
    57985.683,
    57385.564,
    57743.597,
    57556.339,
    56541.188,
    56552.263,
    56734.536,
    56834.51,
    56994.962,
    58452.632,
    58858.972,
    58744.502,
    58930.957,
    59498.772,
    58636.159,
    59255.311,
    57482.178,
    58874.955,
    59600.105,
    59202.746,
    58020.166,
    57845.28,
    57978.07,
    57158.734,
    58084.662,
    59304.592,
    60871.832,
    61461.732,
    62215.826,
    63298.806,
    63320.082,
    63268.816,
    63473.014,
    63592.99,
    62618.68,
    63855.892,
    61901.172,
    61732.976,
    62522.16,
    62557.526,
    63226.462,
    64205.876,
    64446.898,
    66470.03,
    66258.522,
    65268.71,
    64704.04,
    65025.952,
    65776.6
]


# simulation
dt = 1
simulated_paths = np.zeros((n_days + 1, n_simulations))
simulated_paths[0] = S_0

for t in range(1, n_days + 1):
    Z = np.random.normal(0, 1, n_simulations)
    simulated_paths[t] = simulated_paths[t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

# stats
mean_path = simulated_paths.mean(axis=1)
percentile_5 = np.percentile(simulated_paths, 5, axis=1)
percentile_95 = np.percentile(simulated_paths, 95, axis=1)


sampled_indices = random.sample(range(n_simulations), n_paths_to_plot)
colors = [np.random.rand(3,) for _ in range(n_paths_to_plot)]

plt.figure(figsize=(12, 6))

# 1. random traj
for color, i in zip(colors, sampled_indices):
    plt.plot(simulated_paths[:, i], color=color, alpha=0.5, linewidth=0.6)

# 2. avg traj
plt.plot(mean_path, color='blue', label='Average Trajectory', linewidth=2)

# 3. confidence int
plt.fill_between(range(n_days + 1), percentile_5, percentile_95, color='blue', alpha=0.5, label='90% Confidence Interval')

# 4.real traj
if actual_trajectory is not None:
    plt.plot(actual_trajectory, color='red', linestyle='--', linewidth=2, label='Real Trajectory')

final_value = mean_path[-1]
plt.text(n_days, final_value, f'# {final_value:,.2f} PLN',
         fontsize=9, ha='left', va='center', color='blue', fontweight='bold')

plt.title('Monte Carlo Simulation of Minimal Risk Portfolio with Confidence Intervals')
plt.xlabel('Trading Days')
plt.ylabel('Portfolio Value [PLN]')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 2.5))

final_values = simulated_paths[-1]

min_val = np.min(final_values)
q1 = np.percentile(final_values, 25)
median = np.median(final_values)
q3 = np.percentile(final_values, 75)
max_val = np.max(final_values)

import scipy.stats as stats

mean = np.mean(final_values)
std = np.std(final_values)
iqr = q3 - q1
skewness = stats.skew(final_values)
kurt = stats.kurtosis(final_values)
cv = std / mean

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = final_values[(final_values < lower_bound) | (final_values > upper_bound)]

# print
print("=== Boxplot Statistics for Final Portfolio Values ===")
print(f"Minimum:         {min_val:,.2f} PLN")
print(f"Q1 (25%):        {q1:,.2f} PLN")
print(f"Median:          {median:,.2f} PLN")
print(f"Q3 (75%):        {q3:,.2f} PLN")
print(f"Maximum:         {max_val:,.2f} PLN")
print(f"Mean:            {mean:,.2f} PLN")
print(f"Standard Deviation: {std:,.2f} PLN")
print(f"IQR:             {iqr:,.2f} PLN")
print(f"Coefficient of Variation: {cv:.4f}")
print(f"Skewness:        {skewness:.4f}")
print(f"Kurtosis:        {kurt:.4f}")
print(f"Number of Outliers: {len(outliers)}")


box = plt.boxplot(final_values, vert=False, patch_artist=True,
                  boxprops=dict(facecolor='lightblue', color='blue'),
                  medianprops=dict(color='red'),
                  flierprops=dict(marker='o', markerfacecolor='orange', markersize=4, linestyle='none'))

plt.legend([box['boxes'][0], box['medians'][0], box['fliers'][0]],
           ['Residual Value Distribution', 'Median', 'Outliers'],
           loc='upper right')

plt.xlabel('Final Portfolio Value [PLN]')
plt.title('Residual Value Distribution for Minimal Risk Portfolio (Boxplot)')
plt.grid(True)
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import scipy.stats as stats

actual_array = np.array(actual_trajectory[:n_days])    # 252 dni
predicted_mean = mean_path[1:]                          # 252 dni pomijamy dzień 0

absolute_errors = np.abs(actual_array - predicted_mean)
ape = absolute_errors / np.abs(actual_array) * 100
mape_mean = np.mean(ape)

df_mape_daily = pd.DataFrame({
    'Day': np.arange(1, n_days + 1),
    'Actual': actual_array,
    'Predicted (Mean)': predicted_mean,
    'Absolute Error': absolute_errors,
    'APE [%]': ape
})

summary_row = pd.DataFrame({
    'Day': ['AVG MAPE'],
    'Actual': [np.nan],
    'Predicted (Mean)': [np.nan],
    'Absolute Error': [np.nan],
    'APE [%]': [mape_mean]
})

df_mape_daily = pd.concat([df_mape_daily, summary_row], ignore_index=True)

final_values = simulated_paths[-1]

min_val = np.min(final_values)
q1 = np.percentile(final_values, 25)
median = np.median(final_values)
q3 = np.percentile(final_values, 75)
max_val = np.max(final_values)

mean_val = np.mean(final_values)
std_val = np.std(final_values)
iqr_val = q3 - q1
cv_val = std_val / mean_val
skewness = stats.skew(final_values)
kurt = stats.kurtosis(final_values)
lower_bound = q1 - 1.5 * iqr_val
upper_bound = q3 + 1.5 * iqr_val
outliers = final_values[(final_values < lower_bound) | (final_values > upper_bound)]

boxplot_stats = pd.DataFrame({
    "Statistic": ["Minimum", "Q1 (25%)", "Median", "Q3 (75%)", "Maximum",
                  "Mean", "Std Dev", "IQR", "Coeff Variation", "Skewness", "Kurtosis", "Outliers Count"],
    "Value": [min_val, q1, median, q3, max_val,
              mean_val, std_val, iqr_val, cv_val, skewness, kurt, len(outliers)]
})

# mape
mape_all = []
for i in range(n_simulations):
    sim = simulated_paths[1:, i]
    mape_i = np.mean(np.abs((actual_array - sim) / actual_array)) * 100
    mape_all.append(mape_i)
mape_avg_all = np.mean(mape_all)

df_mape_all = pd.DataFrame({
    "Simulation Index": np.arange(1, n_simulations + 1),
    "MAPE [%]": mape_all
})

mape_summary = pd.DataFrame({
    "Metric": ["MAPE dla średniej trajektorii", "Średnia MAPE ze wszystkich symulacji"],
    "Value": [mape_mean, mape_avg_all]
})

output_path = r"C:\Users\lucif\Desktop\Studia\Masters Thesis\Masterka For Real\MSC\mape_kompletne_wyniki.xlsx"

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    df_mape_daily.to_excel(writer, sheet_name="MAPE dzienne", index=False)
    boxplot_stats.to_excel(writer, sheet_name="Statystyki końcowe", index=False)
    df_mape_all.to_excel(writer, sheet_name="MAPE symulacje", index=False)
    mape_summary.to_excel(writer, sheet_name="Podsumowanie MAPE", index=False)

print(f"✅ Zapisano plik Excel z pełnymi wynikami do: {output_path}")
print(f"MAPE średniej trajektorii: {mape_mean:.2f}%")
print(f"Średnia MAPE wszystkich symulacji: {mape_avg_all:.2f}%")

