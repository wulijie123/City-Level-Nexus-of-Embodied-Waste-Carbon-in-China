# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 15:48:30 2026

@author: 26496
"""

import pandas as pd
import numpy as np

full_df = pd.read_csv("./result/inventory/2019HW_all.csv")
sw_df = pd.read_csv("./result/inventory/2019_newdata_combined_predictions.csv")  # Solid Waste prediction data
sw_df = sw_df.rename(columns={'predicted_sw_quantity': "sw_pred"})

# Merge sw_pred by city + sector20
full_df = full_df.merge(sw_df[["city", "sector20", "sw_pred"]], on=["city", "sector20"], how="left")
full_df["sw_pred"] = full_df["sw_pred"].fillna(0)

# Read the total solid waste for each city (0 values are considered NaN)
city_sw = pd.read_excel("./data/city_hw_isw.xlsx", sheet_name='city')
city_sw["ISW/t"] = pd.to_numeric(city_sw["ISW/t"], errors="coerce")
city_sw = city_sw.replace(0, np.nan)

# Merge with city total waste
full_df = full_df.merge(city_sw[["city", "ISW/t"]], on="city", how="left")
full_df = full_df.rename(columns={"ISW/t": "city_sw"})

# Province-level data
prov_sw = pd.read_excel("./data/city_hw_isw.xlsx", sheet_name='province')
prov_sw = prov_sw.rename(columns={"ISW/t": "prov_sw"})
prov_sw["prov_sw"] = pd.to_numeric(prov_sw["prov_sw"], errors="coerce").fillna(0)

full_df["province"] = full_df["province"].astype(str)

# Check for cities that have solid waste data
full_df["has_city_sw"] = full_df["city_sw"].notna()  # Whether there is city-level data
no_city_data = full_df[~full_df["has_city_sw"]]["city"].unique()
print(f"\nCities without city-level data (total {len(no_city_data)} cities):")
for i, city in enumerate(sorted(no_city_data), 1):
    print(f"{i:3d}. {city}")

# Sum of predictions by city
city_pred_sum = full_df.groupby("city")["sw_pred"].transform("sum")
full_df["sw"] = 0.0

# —— Cities with data —— 
mask_city = full_df["has_city_sw"] & (city_pred_sum > 0)
full_df.loc[mask_city, "sw"] = (full_df.loc[mask_city, "sw_pred"] * full_df.loc[mask_city, "city_sw"] / city_pred_sum[mask_city])

# —— Cities without data: Fill using province-level data —— 
# Solid waste amounts for cities in each province
city_unique = full_df[full_df["has_city_sw"]].drop_duplicates(subset=["city", "city_sw"])[["province", "city", "city_sw"]]  # Remove duplicates by city
prov_city_sum = city_unique.groupby("province")["city_sw"].sum().reset_index(name="city_sw_sum")

# Province-level remaining waste
prov_balance = prov_sw.merge(prov_city_sum, on="province", how="left")
prov_balance["city_sw_sum"] = prov_balance["city_sw_sum"].fillna(0)
prov_balance["prov_remain"] = (prov_balance["prov_sw"] - prov_balance["city_sw_sum"]).clip(lower=0)

# Total predicted waste for provinces without city data
prov_pred_missing = (full_df[~full_df["has_city_sw"]].groupby("province")["sw_pred"].sum().reset_index(name="pred_sum"))

# Scaling factors by province
prov_scale = prov_balance.merge(prov_pred_missing, on="province", how="left")
prov_scale["pred_sum"] = prov_scale["pred_sum"].fillna(0)

prov_scale["scale"] = np.where(prov_scale["pred_sum"] > 0, prov_scale["prov_remain"] / prov_scale["pred_sum"], 0)

# Apply scaling factor to missing cities
full_df = full_df.merge( prov_scale[["province", "scale"]],on="province",how="left")
mask_prov = (~full_df["has_city_sw"]) & (full_df["scale"] > 0)

full_df.loc[mask_prov, "sw"] = ( full_df.loc[mask_prov, "sw_pred"]* full_df.loc[mask_prov, "scale"])

# Consistency Check =================
# City-level consistency
city_check = full_df.groupby("city")["sw"].sum()
print("Number of cities with zero solid waste:", (city_check == 0).sum())  # The four provinces

# Compare provincial totals with the adjusted solid waste values
prov_sw_sum = full_df.groupby("province")["sw"].sum()
for province in prov_sw["province"]:
    # Calculate the adjustment factor
    total_prov_sw = prov_sw.loc[prov_sw["province"] == province, "prov_sw"].values[0]
    total_city_sw = prov_sw_sum.loc[province] if province in prov_sw_sum else 0
    if total_prov_sw != total_city_sw:
        # Adjustment factor
        adjustment_factor = total_prov_sw / total_city_sw
        mask = full_df["province"] == province
        full_df.loc[mask, "sw"] *= adjustment_factor

full_df['city'] = full_df['city'].replace('Hainan Tibetan', 'Hainan')
# Even distribution of waste among industries for selected provinces
selected_provinces = ['Yunnan', 'Qinghai', 'Hainan', 'Tibet']
mask_selected_provinces = full_df["city"].isin(selected_provinces)
df_selected_provinces = full_df[mask_selected_provinces]
prov_balance = prov_sw[prov_sw["province"].isin(selected_provinces)]
prov_balance["sw_per_industry"] = prov_balance["prov_sw"] / 20 
for province in selected_provinces:
    sw_per_industry = prov_balance.loc[prov_balance["province"] == province, "sw_per_industry"].values[0]
    # 直接筛选该省的所有行业并均匀分配固废量
    full_df.loc[(full_df["city"] == province) , "sw"] = sw_per_industry
    full_df.loc[(full_df["city"] == province) , "province"] = province  # Assign province value

# Final check
prov_check = full_df.groupby("province")["sw"].sum()
print("Province-level totals comparison:")
print(pd.concat([prov_check, prov_sw.set_index("province")["prov_sw"]], axis=1, keys=["calc", "stat"]))

#full_df.to_excel('./check.xlsx')

# ===================== RAS Adjustment (Province-level) ====================
hy_hwsw = pd.read_excel("./data/city_hw_isw.xlsx", sheet_name='sector')
hy_hwsw = hy_hwsw.groupby("sector20", as_index=False).sum(numeric_only=True)
hy_hwsw['sector20'] = hy_hwsw['sector20'].astype(int)
hy_hwsw = hy_hwsw.sort_values(by="sector20")
hy_hwsw = hy_hwsw.drop(columns="sector42")
complete_sectors = pd.DataFrame({"sector20": range(1, 21)})
hy_hwsw = pd.merge(complete_sectors, hy_hwsw, on="sector20", how="left").fillna(0)

# Industry target values
industry_target = hy_hwsw[['sector20', 'ISW/t']]
full_df['sector20'] = full_df['sector20'].astype(str).str.strip().str.upper()
industry_target['sector20'] = industry_target['sector20'].astype(str).str.strip().str.upper()

# Province target values (using prov_sw as provincial targets)
province_target = prov_sw.copy().rename(columns={"prov_sw": "ISW/t"})
province_target['province'] = province_target['province'].astype(str).str.strip()

# Ensure consistent province names
full_df['province'] = full_df['province'].astype(str).str.strip()

# Create city-industry matrix
adjusted_df = full_df[['city', 'sector20', 'sw']].pivot(index='city', columns='sector20', values='sw')

# Create city-to-province mapping (ensure matching order)
city_province_map = full_df[['city', 'province']].drop_duplicates().set_index('city')['province']
city_province_map = city_province_map.reindex(adjusted_df.index)  # Ensure order matches adjusted_df

# Set max iterations and convergence tolerance
max_iterations = 1000
tolerance = 1e-6

# RAS Iteration process
for iteration in range(max_iterations):
    
    # **Row adjustment**: Adjust based on provincial targets
    for province_name in province_target['province']:
        # Get current provincial total
        province_total_value = province_target[province_target['province'] == province_name]['ISW/t'].values[0]
        
        # Get all cities for the current province (based on mapping)
        province_cities = city_province_map[city_province_map == province_name].index
        
        if len(province_cities) > 0:
            # Sum current waste in province
            current_sum1 = adjusted_df.loc[province_cities].sum().sum()
            
            if current_sum1 > 0:
                # Calculate adjustment scale factor (province target / current total)
                scale_factor1 = province_total_value / current_sum1
                
                # Adjust all cities within province
                adjusted_df.loc[province_cities] = adjusted_df.loc[province_cities] * scale_factor1

    # **Column adjustment**: Adjust based on industry targets
    for j in adjusted_df.columns:
        industry_total = industry_target[industry_target['sector20'] == j]['ISW/t'].values[0]
        current_sum2 = adjusted_df[j].sum()

        if current_sum2 > 0:
            scale_factor2 = industry_total / current_sum2
            adjusted_df[j] = adjusted_df[j] * scale_factor2  # Adjust according to industry target

    # Convergence check
    # Industry difference
    total_industry_diff = np.sum(np.abs(adjusted_df.sum(axis=0) - industry_target.set_index('sector20')['ISW/t'].reindex(adjusted_df.columns).values))
    
    # Province difference: Use city_province_map for grouping
    adjusted_province_sum = adjusted_df.groupby(city_province_map).sum().sum(axis=1)
    
    # Get target provincial totals
    province_target_set = province_target.set_index('province')['ISW/t']
    
    # Calculate total provincial difference
    total_province_diff = np.sum(np.abs(adjusted_province_sum - province_target_set.reindex(adjusted_province_sum.index).values))
    
    print(f"Iteration {iteration + 1}:")
    print(f"Total industry difference: {total_industry_diff}")
    print(f"Total province difference: {total_province_diff}")

    # If difference is below tolerance, stop iteration
    if total_industry_diff < tolerance and total_province_diff < tolerance:
        print(f"RAS adjustment converged after {iteration + 1} iterations.")
        break

# Final adjusted totals comparison
adjusted_industry_sum = adjusted_df.sum(axis=0)  # Sum of adjusted values by industry
adjusted_province_sum = adjusted_df.groupby(city_province_map).sum().sum(axis=1)  # Sum of adjusted values by province

# Print final adjusted industry totals vs targets
print("\nFinal adjusted industry totals vs targets:")
for sector in adjusted_industry_sum.index:
    adjusted_value = adjusted_industry_sum[sector]
    target_value = industry_target[industry_target['sector20'] == sector]['ISW/t'].values[0]
    print(f"Sector {sector}: Adjusted = {adjusted_value}, Target = {target_value}, Difference = {adjusted_value - target_value}")

# Print final adjusted provincial totals vs targets
print("\nFinal adjusted provincial totals vs targets:")
for province in adjusted_province_sum.index:
    adjusted_value = adjusted_province_sum[province]
    target_value = province_target[province_target['province'] == province]['ISW/t'].values[0]
    print(f"Province {province}: Adjusted = {adjusted_value}, Target = {target_value}, Difference = {adjusted_value - target_value}")

# Print final total values comparison
final_industry_total = adjusted_industry_sum.sum()
final_province_total = adjusted_province_sum.sum()
final_industry_target_total = industry_target['ISW/t'].sum()
final_province_target_total = province_target['ISW/t'].sum()

print("\nFinal total comparison:")
print(f"Total industry adjusted = {final_industry_total}, Total industry target = {final_industry_target_total}, Difference = {final_industry_total - final_industry_target_total}")
print(f"Total province adjusted = {final_province_total}, Total province target = {final_province_target_total}, Difference = {final_province_total - final_province_target_total}")

# Reset data format
adjusted_df_reset = adjusted_df.reset_index().melt(id_vars=['city'], var_name='sector20', value_name='adjusted_ISW')

# Merge adjusted results back into original data
full_df = full_df.merge(adjusted_df_reset[['city', 'sector20', 'adjusted_ISW']], on=['city', 'sector20'], how='left')

# ===================== Merge Carbon Emission Data ====================
c_data = pd.read_excel('./data/2019_Carbon.xlsx')
full_df = full_df.astype({'city': str, 'sector20': str}).merge(c_data.astype({'city': str, 'sector20': str}), on=['city', 'sector20'], how='left')
full_df['Carbon/Mt'] = full_df['Carbon/Mt'].fillna(0)

# Save the final results
full_df = full_df.rename(columns={'adjusted_ISW': 'ISW', 'Carbon/Mt': 'Carbon'})
full_df = full_df[['city', 'sector20', 'HW', 'ISW', 'Carbon', 'total_output', 'GDP', 'pop', 'growth', 'secondaryrate', 'tertiaryrate']]
full_df.to_csv('./result/inventory/2019_waste_data.csv', index=False)


