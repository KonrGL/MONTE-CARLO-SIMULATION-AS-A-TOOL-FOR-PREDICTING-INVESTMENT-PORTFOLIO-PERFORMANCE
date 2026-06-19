import numpy as np
import matplotlib.pyplot as plt
import random

# PARAMETRY
mu = 0.00201789
sigma = 0.01575
n_days = 252
n_simulations = 10000
S_0 = 28272.0079
n_paths_to_plot = 1000


actual_trajectory = [
    27633.78,
    27635.5321,
    28272.0079,
    28294.7369,
    28739.7389,
    28819.6048,
    29206.3011,
    28560.7295,
    28717.8156,
    28006.169,
    28335.7533,
    28416.2059,
    28567.3082,
    28485.833,
    28782.2127,
    28435.8192,
    28270.657,
    28119.1559,
    28105.2555,
    30234.348,
    30508.8165,
    31292.9899,
    31887.6276,
    31139.1947,
    31361.3616,
    31434.0851,
    31188.1926,
    31920.1794,
    31634.8693,
    32563.3188,
    32843.6511,
    33296.9379,
    33305.3601,
    33207.4372,
    32614.5583,
    33276.3104,
    34091.9745,
    35081.9579,
    34793.9381,
    34761.6372,
    36735.9768,
    37898.597,
    37476.9786,
    38211.0446,
    38944.4759,
    38965.315,
    38697.0759,
    39483.6837,
    41472.6069,
    41135.1303,
    43132.3157,
    43101.7104,
    42178.7422,
    42139.2963,
    42272.6251,
    43050.7298,
    42435.3061,
    42129.4,
    42032.5284,
    43512.4979,
    42802.0157,
    42396.053,
    41423.6723,
    42174.8975,
    41897.052,
    42373.8826,
    41756.3792,
    42106.3657,
    40390.3183,
    41058.0266,
    41188.9376,
    39438.3693,
    40054.9032,
    40816.6385,
    40282.1128,
    41053.5704,
    42110.9959,
    42749.7045,
    42263.4684,
    44220.3894,
    44346.7798,
    44327.7806,
    43142.1505,
    45811.361,
    45631.5559,
    46005.9113,
    47129.4957,
    47010.739,
    48718.5511,
    48742.667,
    50297.847,
    50953.3106,
    51201.9102,
    51809.1872,
    50621.0907,
    51992.1329,
    52312.1601,
    51950.084,
    52181.3645,
    51865.9212,
    50459.9197,
    50361.1963,
    50873.603,
    49927.687,
    50457.3734,
    50507.7031,
    50711.5155,
    50442.4614,
    50063.5547,
    52705.116,
    51815.427,
    51669.389,
    53061.578,
    53434.724,
    54351.871,
    53591.512,
    53353.574,
    53220.51,
    52706.988,
    52147.254,
    54090.068,
    55410.363,
    55020.82,
    53853.647,
    53874.208,
    53555.269,
    53332.159,
    53583.267,
    54028.927,
    54261.167,
    53941.296,
    55002.823,
    51752.532,
    50203.036,
    48957.46,
    49568.872,
    48980.987,
    50057.639,
    49526.961,
    50262.083,
    50282.413,
    51355.009,
    49165.759,
    49325.507,
    50183.037,
    49641.487,
    48272.1665,
    46477.3233,
    46887.2873,
    46424.8927,
    46398.6169,
    46802.1089,
    47753.8027,
    48550.148,
    48851.2718,
    50335.416,
    50290.031,
    49612.1268,
    49658.9886,
    50258.8158,
    51013.7407,
    50239.3114,
    51300.955,
    51775.735,
    52072.904,
    52474.346,
    54335.8,
    53540.335,
    51887.2476,
    50613.0539,
    50471.5828,
    50978.7657,
    50168.7251,
    49965.2308,
    51213.8341,
    51468.7686,
    50850.6121,
    52775.4646,
    53202.3548,
    53039.2526,
    52884.915,
    53207.7865,
    52962.4276,
    54546.936,
    55041.512,
    55439.739,
    55455.949,
    54939.635,
    54494.727,
    55016.611,
    54020.37,
    52459.0594,
    53792.957,
    55394.214,
    56665.668,
    56589.986,
    56628.388,
    56324.041,
    55823.621,
    54458.427,
    55288.276,
    56503.876,
    56048.601,
    56080.571,
    57019.647,
    56340.566,
    56716.861,
    56628.019,
    55558.926,
    55646.707,
    55957.134,
    56106.946,
    56292.716,
    57799.064,
    58094.458,
    58097.186,
    58369.621,
    58910.868,
    58196.951,
    58740.375,
    57012.318,
    58412.017,
    59101.771,
    58643.022,
    57567.138,
    57433.56,
    57521.76,
    56824.928,
    57795.584,
    58833.884,
    60220.504,
    60965.584,
    61796.232,
    62848.832,
    62746.184,
    62679.672,
    62833.308,
    62779.18,
    61781.06,
    62893.804,
    61016.204,
    60954.232,
    61570.54,
    61641.632,
    62362.824,
    63313.632,
    63432.476,
    65293.14,
    65076.464,
    64407.14,
    63825.56,
    64123.924,
    64704.6
]



# SYMULACJA
dt = 1
simulated_paths = np.zeros((n_days + 1, n_simulations))
simulated_paths[0] = S_0

for t in range(1, n_days + 1):
    Z = np.random.normal(0, 1, n_simulations)
    simulated_paths[t] = simulated_paths[t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)


mean_path = simulated_paths.mean(axis=1)
percentile_5 = np.percentile(simulated_paths, 5, axis=1)
percentile_95 = np.percentile(simulated_paths, 95, axis=1)


sampled_indices = random.sample(range(n_simulations), n_paths_to_plot)
colors = [np.random.rand(3,) for _ in range(n_paths_to_plot)]


plt.figure(figsize=(12, 6))

# 1. Losowe trajektorie
for color, i in zip(colors, sampled_indices):
    plt.plot(simulated_paths[:, i], color=color, alpha=0.5, linewidth=0.6)

# 2. Średnia trajektoria
plt.plot(mean_path, color='blue', label='Average Trajectory', linewidth=2)

# 3. Przedział ufności
plt.fill_between(range(n_days + 1), percentile_5, percentile_95, color='blue', alpha=0.5, label='90% Confidence Interval')

# 4. Trajektoria rzeczywista
if actual_trajectory is not None:
    plt.plot(actual_trajectory, color='red', linestyle='--', linewidth=2, label='Real Trajectory')

# 5. Tekst z końcową wartością
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

# Wartości odstające (poza 1.5 * IQR)
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = final_values[(final_values < lower_bound) | (final_values > upper_bound)]

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

#MAPE
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

