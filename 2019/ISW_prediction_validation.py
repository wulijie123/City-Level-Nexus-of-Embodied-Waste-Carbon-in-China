# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 15:46:34 2026

@author: DELL
"""
import pandas as pd
import numpy as np

sw_pre = pd.read_csv("./result/inventory/2019_waste_data.csv")
sw_city = pd.read_excel("./data/city_hw_isw.xlsx", sheet_name='city')
sw_city["ISW/t"] = pd.to_numeric(sw_city["ISW/t"], errors="coerce")
sw_city = sw_city.replace(0, np.nan)  # Treat 0 as no data
hy = pd.read_excel("./data/city_hw_isw.xlsx", sheet_name="sector_isw", usecols=range(7), nrows=20)  # Proportion of specific waste categories assigned to sectors

# Save the original city count
original_city_count = len(sw_city)
print(f"Original number of cities: {original_city_count}")

# Prepare allocation proportion matrix
hy = hy.set_index(hy.columns[0])
waste_types = hy.columns.tolist()

# Keep cities with data for 4 or more waste types
valid_cities = []
city_data_count = {}  # Count of waste types with data for each city

for _, city_row in sw_city.iterrows():
    city = city_row['city']
    if pd.isna(city):
        continue
    
    # Count how many waste types have data for this city
    data_count = 0
    for waste_type in waste_types:
        if not pd.isna(city_row[waste_type]):
            data_count += 1
    
    city_data_count[city] = data_count
    
    # Keep cities with data for 4 or more waste types
    if data_count >= 4:
        valid_cities.append(city)

# Filter cities based on the condition
sw_city_filtered = sw_city[sw_city['city'].isin(valid_cities)].copy()

print(f"Cities with data for 4 or more waste types: {len(valid_cities)}")
print(f"Number of excluded cities: {original_city_count - len(valid_cities)}")

# Use the filtered data for further calculation
sw_city = sw_city_filtered

# Calculate solid waste allocation for each city and sector
sw_actual = []

for _, city_row in sw_city.iterrows():
    city = city_row['city']
    
    for sector20 in hy.index:
        sw_allocation = 0
        for waste_type in waste_types:
            waste_amount = city_row[waste_type]
            if pd.isna(waste_amount):
                continue  # Skip waste types with no data
            sw_allocation += float(waste_amount) * hy.loc[sector20, waste_type]
        
        sw_actual.append({
            'city': city,
            'sector20': sector20,
            'sw_act': sw_allocation })

sw_actual = pd.DataFrame(sw_actual)

# Rename the prediction column to 'sw_pre' to align with 'sw_act'
sw_pre = sw_pre.rename(columns={'ISW': 'sw_pre'})

# Merge for comparison (only compare common cities and sectors)
comparison = pd.merge(
    sw_actual[['city', 'sector20', 'sw_act']],  # Keep only necessary columns
    sw_pre[['city', 'sector20', 'sw_pre']],     # Keep only necessary columns
    on=['city', 'sector20'], 
    how='inner'
)

# Exclude records where sw_act is 0 (0 means no proportion data, do not include in comparison)
comparison = comparison[comparison['sw_act'] != 0]

# Calculate the differences
comparison['difference'] = comparison['sw_pre'] - comparison['sw_act']
comparison['relative_difference%'] = (comparison['difference'] / comparison['sw_act'] * 100).round(2)

# Output results
print(f"Statistical check:")
print(f"Number of common cities: {comparison['city'].nunique()}")
print(f"Total records: {len(comparison)}")
print(f"Average predicted value: {comparison['sw_pre'].mean():.2f}")
print(f"Average actual value: {comparison['sw_act'].mean():.2f}")
print(f"Average difference: {comparison['difference'].mean():.2f}")
print(f"Total predicted value: {comparison['sw_pre'].sum():.2f}")
print(f"Total actual value: {comparison['sw_act'].sum():.2f}")
print(f"Total difference: {comparison['difference'].sum():.2f}")
print(f"Average relative difference: {comparison['relative_difference%'].mean():.2f}%")

# Show top 10 rows of comparison results
print("\nTop 10 comparison results:")
print(comparison[['city', 'sector20', 'sw_pre', 'sw_act', 'difference', 'relative_difference%']].head(10))

# Summarize by city
city_summary = comparison.groupby('city').agg({
    'sw_pre': 'sum',
    'sw_act': 'sum'
}).reset_index()
city_summary['difference'] = city_summary['sw_pre'] - city_summary['sw_act']
city_summary['relative_difference%'] = (city_summary['difference'] / city_summary['sw_pre'] * 100).round(2)

print("\nCity-level summary (Top 10 cities):")
print(city_summary.head(10))

# Calculate fitting metrics
y_true = comparison['sw_act'].values
y_pred = comparison['sw_pre'].values

# 1. Root Mean Squared Error (RMSE)
rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

# 2. Mean Absolute Error (MAE)
mae = np.mean(np.abs(y_true - y_pred))

# 3. Coefficient of Determination (R²)
ss_res = np.sum((y_true - y_pred) ** 2)
ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
r2 = 1 - (ss_res / ss_tot)

# 4. Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print("Fitting metrics results:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.4f}")
print(f"MAPE: {mape:.2f}%")

# Plotting
import matplotlib.pyplot as plt

y_true = np.array(comparison['sw_act']).ravel()
y_pred = np.array(comparison['sw_pre']).ravel()
N = len(y_true)

AX_MAX = max(y_true.max(), y_pred.max()) + 1000000
bins = 42

# 2D histogram to calculate density
H, xedges, yedges = np.histogram2d(y_true, y_pred, bins=bins)

# Find the bin index for each point
x_idx = np.clip(np.digitize(y_true, xedges) - 1, 0, H.shape[0] - 1)
y_idx = np.clip(np.digitize(y_pred, yedges) - 1, 0, H.shape[1] - 1)
point_counts = H[x_idx, y_idx]

# Adjust color scale
vmin = 1
vmax = np.percentile(point_counts, 99)

# Plotting
fig, ax = plt.subplots(figsize=(7, 7), dpi=200)

sc = ax.scatter(
    y_true, y_pred,
    c=point_counts,
    s=40,
    cmap="jet",
    alpha=0.9,
    edgecolors="none"
)
sc.set_clim(vmin, vmax)

# 1:1 reference line (dark red)
ax.plot(
    [0, AX_MAX], [0, AX_MAX],
    color="#8B0000",
    linewidth=4
)

# Fitted line (gray)
coef = np.polyfit(y_true, y_pred, 1)
x_fit = np.linspace(0, AX_MAX, 400)
ax.plot(
    x_fit,
    coef[0] * x_fit + coef[1],
    color="#4f4f4f",
    linewidth=4
)

# Axis labels & font size
ax.set_xlim(0, AX_MAX)
ax.set_ylim(0, AX_MAX)

ax.set_xlabel("Declared generation quantity of ISW (t)", fontsize=21)
ax.set_ylabel("Predicted generation quantity of ISW (t)", fontsize=21)

ax.ticklabel_format(style="sci", axis="both", scilimits=(6, 6))
ax.tick_params(labelsize=21)
ax.xaxis.get_offset_text().set_fontsize(18)
ax.yaxis.get_offset_text().set_fontsize(18)

# Statistics annotation
ax.text(
    0.10, 0.95,
    f"R² = {r2:.2f}\nRMSE = {rmse:.2f}\nN = {N}",
    transform=ax.transAxes,
    va="top",
    fontsize=21
)

# Color bar
cax = fig.add_axes([0.78, 0.15, 0.03, 0.30])
cbar = plt.colorbar(sc, cax=cax)

desired_ticks = np.array([50, 150, 250])
valid_ticks = desired_ticks[
    (desired_ticks >= vmin) & (desired_ticks <= vmax)
]

if len(valid_ticks) > 0:
    cbar.set_ticks(valid_ticks)
    cbar.set_ticklabels([str(t) for t in valid_ticks])
else:
    ticks = np.linspace(vmin, vmax, 4, dtype=int)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([str(t) for t in ticks])

cbar.set_label("Count", fontsize=18)
cbar.ax.tick_params(labelsize=18)

# Save & show plot
plt.tight_layout()
plt.show()
