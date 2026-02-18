# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 20:01:42 2026

@author: DELL
"""

import pandas as pd
import numpy as np

full_df = pd.read_excel("./data/2019_HW.xlsx")
df1 = pd.read_excel("./data/city_hw_isw.xlsx", sheet_name='province')

# Merge provincial totals into the original data
full_df = pd.merge(full_df, df1[['province', 'HW/t']], on='province', how='left')

# Sector matching
hy_hwsw = pd.read_excel("./data/city_hw_isw.xlsx", sheet_name='sector')
hy_hwsw = hy_hwsw.groupby("sector20", as_index=False).sum(numeric_only=True)
hy_hwsw['sector20'] = hy_hwsw['sector20'].astype(int)
hy_hwsw = hy_hwsw.sort_values(by="sector20")
hy_hwsw = hy_hwsw.drop(columns="sector42")
complete_sectors = pd.DataFrame({"sector20": range(1, 21)})
hy_hwsw = pd.merge(complete_sectors, hy_hwsw, on="sector20", how="left").fillna(0)

# ===================== RAS Adjustment =================
# Industry target values
industry_target = hy_hwsw[['sector20', 'HW/t']]
full_df['sector20'] = full_df['sector20'].astype(str).str.strip().str.upper()
industry_target['sector20'] = industry_target['sector20'].astype(str).str.strip().str.upper()

# Province target values
province_target = df1[['province', 'HW/t']]

# Create city-industry matrix
adjusted_df = full_df[['city', 'sector20', 'HW_generation']].pivot(index='city', columns='sector20', values='HW_generation')

# Create city-to-province mapping (ensure order consistency)
# Use mapping from full_df but make sure city order matches adjusted_df
city_province_map = full_df[['city', 'province']].drop_duplicates().set_index('city')['province']
city_province_map = city_province_map.reindex(adjusted_df.index)  # Ensure order matches adjusted_df

# Set maximum iterations and convergence tolerance
max_iterations = 1000
tolerance = 1e-6

# RAS iterative process
for iteration in range(max_iterations):
    
    # **Row adjustment**: adjust according to province target
    for province_name in province_target['province']:
        # Get the current province's target total
        province_total_value = province_target[province_target['province'] == province_name]['HW/t'].values[0]
        
        # Get all cities in the current province (based on mapping)
        province_cities = city_province_map[city_province_map == province_name].index
        
        if len(province_cities) > 0:
            # Get the total HW in all cities of the province
            current_sum1 = adjusted_df.loc[province_cities].sum().sum()
            
            if current_sum1 > 0:
                # Calculate adjustment factor (province target / current total)
                scale_factor1 = province_total_value / current_sum1
                
                # Adjust all cities within the province
                adjusted_df.loc[province_cities] = adjusted_df.loc[province_cities] * scale_factor1

    # **Column adjustment**: adjust according to industry target
    for j in adjusted_df.columns:
        industry_total = industry_target[industry_target['sector20'] == j]['HW/t'].values[0]
        current_sum2 = adjusted_df[j].sum()

        if current_sum2 > 0:
            scale_factor2 = industry_total / current_sum2
            adjusted_df[j] = adjusted_df[j] * scale_factor2  # Adjust according to industry target

    # Convergence check
    # Industry difference
    total_industry_diff = np.sum(np.abs(adjusted_df.sum(axis=0) - industry_target.set_index('sector20')['HW/t'].reindex(adjusted_df.columns).values))
    
    # Province difference: group by city_province_map
    adjusted_province_sum = adjusted_df.groupby(city_province_map).sum().sum(axis=1)
    
    # Get target province totals
    province_target_set = province_target.set_index('province')['HW/t']
    
    # Calculate total province difference
    total_province_diff = np.sum(np.abs(adjusted_province_sum - province_target_set.reindex(adjusted_province_sum.index).values))
    
    print(f"Iteration {iteration + 1}:")
    print(f"Total industry difference: {total_industry_diff}")
    print(f"Total province difference: {total_province_diff}")

    # Stop iteration if differences are below tolerance
    if total_industry_diff < tolerance and total_province_diff < tolerance:
        print(f"RAS adjustment converged after {iteration + 1} iterations")
        break

# Calculate final adjusted results
adjusted_industry_sum = adjusted_df.sum(axis=0)  # Summarize adjusted data by industry
adjusted_province_sum = adjusted_df.groupby(city_province_map).sum().sum(axis=1)  # Summarize adjusted data by province

# Print comparison of final adjusted industry totals with targets
print("\nComparison of final adjusted industry totals vs targets:")
for sector in adjusted_industry_sum.index:
    adjusted_value = adjusted_industry_sum[sector]
    target_value = industry_target[industry_target['sector20'] == sector]['HW/t'].values[0]
    print(f"Industry {sector}: Adjusted = {adjusted_value}, Target = {target_value}, Difference = {adjusted_value - target_value}")

# Print comparison of final adjusted province totals with targets
print("\nComparison of final adjusted province totals vs targets:")
for province in adjusted_province_sum.index:
    adjusted_value = adjusted_province_sum[province]
    target_value = province_target[province_target['province'] == province]['HW/t'].values[0]
    print(f"Province {province}: Adjusted = {adjusted_value}, Target = {target_value}, Difference = {adjusted_value - target_value}")

# Print comparison of final total values
final_industry_total = adjusted_industry_sum.sum()
final_province_total = adjusted_province_sum.sum()
final_industry_target_total = industry_target['HW/t'].sum()
final_province_target_total = province_target['HW/t'].sum()

print("\nComparison of final total values:")
print(f"Total industry adjusted = {final_industry_total}, Total industry target = {final_industry_target_total}, Difference = {final_industry_total - final_industry_target_total}")
print(f"Total province adjusted = {final_province_total}, Total province target = {final_province_target_total}, Difference = {final_province_total - final_province_target_total}")

# Restore to original data format
adjusted_df_reset = adjusted_df.reset_index().melt(id_vars=['city'], var_name='sector20', value_name='adjusted_HW')

# Merge adjusted results into original data
full_df = full_df.merge(adjusted_df_reset[['city', 'sector20', 'adjusted_HW']], on=['city', 'sector20'], how='left')


#========================= Match industry total output and city variables
x2019 = pd.read_csv("./data/2019MRIO/4.2 X2019.csv")
hy = pd.read_csv('./data/io42_sector20_mapping.csv', encoding='gbk')

total_output = np.zeros(len(full_df))

# Process each city
num_cities = len(full_df['city'].unique())
for i in range(num_cities):
    start42 = i*42
    end42 = start42 + 42
    city_values42 = x2019['total output_2019'].values[start42:end42]
    # Map 42 → 20 industries
    for j in range(42):
        sector20 = int(hy.loc[j, 'ID1'])
        idx20 = i*20 + (sector20-1)  
        total_output[idx20] += city_values42[j]

full_df['total_output'] = total_output

# --------------------
city_var = pd.read_excel("./data/2019_city_variate.xlsx")
# Merge into full_df
full_df = full_df.merge(city_var,on='city',how='left')

full_df = full_df[['city','province','sector20','adjusted_HW','total_output','GDP','pop',
                   'growth','secondaryrate','tertiaryrate']]
full_df = full_df.rename(columns={'adjusted_HW':'HW'})

# --------------------
full_df.to_csv("./result/inventory/2019HW_all.csv",index=False)
