import numpy as np
import matplotlib.pyplot as plt
import random

# parameters
mu = 0.00076847
sigma = 0.01241
n_days = 252
n_simulations = 10000
S_0 = 14783.4633
n_paths_to_plot = 1000

# opt traj
actual_trajectory = [
    14721.5743, 14902.898, 14883.0372, 15097.73, 15048.0485, 15131.1466, 14897.0225, 14887.25, 14407.512, 14606.8683,
    14645.685, 14758.8101, 14780.4885, 14899.9579, 14830.9985, 14750.0719, 14616.8523, 14656.7999, 15315.5062,
    15453.6785, 15706.6041, 15931.5153, 15702.3, 15764.494, 15795.3497, 15797.8749, 16108.8785, 15816.5416,
    16109.5306, 16061.731, 16133.3728, 16244.3969, 16204.4496, 15828.152, 16074.3606, 16250.9401, 16528.9201,
    16416.5639, 16303.0248, 16965.7443, 17255.3155, 16955.085, 17145.4672, 17404.1451, 17408.0513, 17232.087,
    17389.7265, 18045.5662, 17960.3842, 18548.8883, 18459.8697, 18179.2666, 18160.5078, 18196.3098, 18537.5908,
    18307.0429, 18238.3319, 18165.1174, 18454.4222, 18414.1859, 18170.0218, 17904.3936, 18361.7578, 18225.5209,
    18330.7839, 18126.1797, 18207.3578, 17719.7779, 17931.4717, 17893.6059, 17191.129, 17445.8806, 17646.6294,
    17616.9621, 17748.396, 18106.2827, 18300.0747, 18103.0054, 18909.9118, 19097.7408, 19008.7045, 18729.407,
    19671.3751, 19652.925, 19828.1082, 20325.0423, 20497.742, 21281.5034, 21411.7985, 21796.8262, 22279.5614,
    22135.1494, 22426.1754, 21910.1253, 22260.3226, 22358.0764, 22308.642, 22405.6322, 22268.7779, 21846.1662,
    21906.4662, 22151.3081, 21788.1422, 21932.5051, 22023.2986, 22066.0928, 21957.7912, 21566.2259, 22563.6487,
    22153.4464, 22216.6087, 22806.127, 22810.422, 23235.7825, 22969.3989, 22808.21, 22877.1358, 22634.3133,
    22367.7498, 22999.7899, 23459.5689, 23238.3055, 22766.3066, 23011.9064, 22950.6773, 22838.5205, 22931.6959,
    22977.2699, 23051.7622, 22910.2418, 23377.6581, 22561.7472, 21999.9715, 21533.8852, 21801.1011, 21512.8676,
    21751.9979, 21439.5513, 21680.977, 21711.1336, 22049.9145, 21386.7949, 21502.697, 21816.1158, 21508.3472,
    20987.3581, 20145.247, 20240.4351, 20149.0702, 20174.9721, 20268.7414, 20676.0485, 20878.5503, 21123.2061,
    21658.0697, 21624.9246, 21380.6001, 21339.9424, 21532.2541, 21741.3392, 21465.0503, 21825.6937, 21841.8549,
    22041.8365, 22137.8277, 22680.7496, 22268.1556, 21840.062, 21301.2558, 21155.7542, 21221.6768, 21060.8724,
    20905.9313, 21488.8603, 21544.9395, 21293.4209, 21890.5493, 21923.7803, 21730.4848, 21728.0757, 21309.0392,
    21302.6878, 21787.5248, 22068.423, 22298.9815, 22321.8801, 22008.8779, 21864.2589, 21862.2821, 21434.0844,
    20961.3045, 21305.8634, 21856.4753, 22390.1506, 22360.9673, 22408.706, 22338.0595, 22227.5832, 21743.3801,
    22059.6898, 22337.4202, 22153.1966, 22213.0821, 22476.3502, 22368.2868, 22525.0343, 22535.359, 22169.437,
    22177.3255, 22161.3978, 22287.5796, 22293.2276, 22797.2431, 22940.3616, 22882.5034, 22991.7787, 23193.9239,
    22900.801, 23103.3242, 22452.9412, 22913.8315, 23274.2087, 23050.3224, 22646.2129, 22648.9345, 22657.7494,
    22540.8661, 22970.0219, 23383.2837, 23777.5853, 23993.9063, 24382.7456, 24851.8779, 24750.463, 24929.2975,
    25026.6333, 24729.4589, 24242.9386, 24843.1811, 24295.8447, 24251.3714, 24425.2624, 24433.779, 24593.6301,
    24922.1177, 24935.7644, 25542.9106, 25542.1868, 24921.5217, 24687.5141, 24791.3523, 24918.9341
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

# random traj
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

# 4. real traj
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

simulated_log_returns = np.diff(np.log(simulated_paths), axis=0)

plt.figure(figsize=(10, 6))
plt.hist(simulated_log_returns.flatten(), bins=50, alpha=0.7, color='blue', density=True, label='Simulated Log-Returns')
plt.axvline(x=simulated_log_returns.mean(), color='red', linestyle='--', label='Mean')
plt.title('Histogram of Simulated Log-Returns')
plt.xlabel('Log-Return')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()