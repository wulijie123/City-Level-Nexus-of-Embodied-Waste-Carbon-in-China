# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 21:20:51 2026

@author: DELL
"""

import pandas as pd 
import numpy as np
import copy
import os

os.chdir('../2019')

city_info = pd.read_csv('data/City_Name.csv', encoding='gbk')
city_info = city_info[city_info['City'] != 'Laiwu']  # In 2019, Laiwu was merged into Jinan
cities = city_info.loc[:, ['City']]

sector_info = pd.read_excel('data/sector20.xlsx')
provinces = ['Yunnan', 'Qinghai', 'Hainan Tibetan', 'Tibet']

# ==================== Find cities with zero emissions ====================
em = pd.read_csv('result/inventory/2019_waste_data.csv')

city_sums = em.groupby('city')[['HW', 'ISW', 'Carbon']].sum()

hw_zero_cities = city_sums[city_sums['HW'] == 0].index.tolist()
isw_zero_cities = city_sums[city_sums['ISW'] == 0].index.tolist()
c_zero_cities = city_sums[city_sums['Carbon'] == 0].index.tolist()

# ==================== Carbon Data ====================
cpbe = pd.read_excel('result/result_c/Production_by_sector.xlsx').iloc[:, :-1]
ccbe = pd.read_excel('result/result_c/Consumption_by_sector_with_export.xlsx')
ccbe = ccbe[ccbe['city'] != 'Export']

# ==================== Core Calculation Function ====================
def calculate_ccd(waste_type, waste_prefix):

    print(f"Calculating {waste_prefix} - Carbon Coupling Degree...")

    # ---------- Production Side ----------
    waste_pbe = pd.read_excel(f'result/result_{waste_type}/Production_by_sector.xlsx').iloc[:, :-1]
    pbe = pd.merge(waste_pbe, cpbe, on=['city', 'sector'])
    pbe.columns = ['city', 'sector', f'pbe_{waste_type}', 'pbe_c']

    # ---------- Consumption Side ----------
    waste_cbe = pd.read_excel(f'result/result_{waste_type}/Consumption_by_sector_with_export.xlsx')
    waste_cbe = waste_cbe[waste_cbe['city'] != 'Export']
    cbe = pd.merge(waste_cbe, ccbe, on=['city', 'sector'])
    cbe.columns = ['city', 'sector', f'cbe_{waste_type}', 'cbe_c']

    # ---------- Remove provinces ----------
    for p in provinces:
        pbe = pbe[pbe['city'] != p]
        cbe = cbe[cbe['city'] != p]

    # ---------- Remove cities with zero emissions ----------
    if waste_type == 'isw':
        drop_cities = set(isw_zero_cities + c_zero_cities)
    else:
        drop_cities = set(hw_zero_cities + c_zero_cities)

    pbe = pbe[~pbe['city'].isin(drop_cities)]

    # ---------- Find Maximum Values ----------
    max_waste = max(pbe[f'pbe_{waste_type}'].max(), cbe[f'cbe_{waste_type}'].max())
    max_c = max(pbe['pbe_c'].max(), cbe['cbe_c'].max())

    # ==================== Production Side ====================
    pbe[f'pbe_{waste_type}_nor'] = np.log(pbe[f'pbe_{waste_type}'] + 1) / np.log(max_waste + 1)
    pbe['pbe_c_nor'] = np.log(pbe['pbe_c'] + 1) / np.log(max_c + 1)

    pbe[f'car_{waste_type}_C'] = (pbe['pbe_c_nor'] * pbe[f'pbe_{waste_type}_nor'] /
                                  ((pbe['pbe_c_nor'] + pbe[f'pbe_{waste_type}_nor']) * 0.5) ** 2) ** 0.5
    pbe[f'car_{waste_type}_T'] = 0.5 * pbe['pbe_c_nor'] + 0.5 * pbe[f'pbe_{waste_type}_nor']
    pbe[f'car_{waste_type}_D'] = (pbe[f'car_{waste_type}_C'] * pbe[f'car_{waste_type}_T']) ** 0.5

    # ==================== Consumption Side ====================
    cbe[f'cbe_{waste_type}_nor'] = np.log(cbe[f'cbe_{waste_type}'] + 1) / np.log(max_waste + 1)
    cbe['cbe_c_nor'] = np.log(cbe['cbe_c'] + 1) / np.log(max_c + 1)

    cbe[f'car_{waste_type}_C'] = (cbe['cbe_c_nor'] * cbe[f'cbe_{waste_type}_nor'] /
                                  ((cbe['cbe_c_nor'] + cbe[f'cbe_{waste_type}_nor']) * 0.5) ** 2) ** 0.5
    cbe[f'car_{waste_type}_T'] = 0.5 * cbe['cbe_c_nor'] + 0.5 * cbe[f'cbe_{waste_type}_nor']
    cbe[f'car_{waste_type}_D'] = (cbe[f'car_{waste_type}_C'] * cbe[f'car_{waste_type}_T']) ** 0.5

    pbe.fillna(0, inplace=True)
    cbe.fillna(0, inplace=True)

    # Sort and take top 10 industries for each city
    sorted_pbe = pbe.groupby('city', group_keys=False).apply(
        lambda x: x.sort_values(f'car_{waste_type}_D', ascending=False).reset_index(drop=True)
    )

    sorted_cbe = cbe.groupby('city', group_keys=False).apply(
        lambda x: x.sort_values(f'car_{waste_type}_D', ascending=False).reset_index(drop=True)
    )

    return sorted_pbe, sorted_cbe

# ==================== City Summary ====================
def dofcity(df0, col, waste_type):
    df1 = pd.DataFrame(columns=[col], index=list(set(df0['city'])))
    for k in set(df0['city']):
        df = df0[df0['city'] == k].iloc[:10]
        df1.loc[k, col] = df[f'car_{waste_type}_D'].sum()
    df1.reset_index(inplace=True)
    df1.columns = ['city', col]
    return df1

# ==================== Sector Averages ====================
def mean_by_sector(df0, waste_type):
    # Sort by sector20 order
    df_mean = (
        df0.groupby('sector')[f'car_{waste_type}_D']
           .mean()
           .reset_index()
           .rename(columns={f'car_{waste_type}_D': f'{waste_type}_mean'})
    )
    # Use sector20.xlsx order
    df_mean['sector'] = pd.Categorical(df_mean['sector'], categories=sector_info['Sector1'], ordered=True)
    df_mean = df_mean.sort_values('sector').reset_index(drop=True)
    return df_mean

# ==================== Calculations ====================
isw_pbe_sorted, isw_cbe_sorted = calculate_ccd('isw', 'Industry Solid Waste')
hw_pbe_sorted, hw_cbe_sorted = calculate_ccd('hw', 'Hazardous Waste')

c_iswpbe_city = dofcity(isw_pbe_sorted, "c_isw_pbe", 'isw')
c_iswcbe_city = dofcity(isw_cbe_sorted, "c_isw_cbe", 'isw')
c_hwpbe_city = dofcity(hw_pbe_sorted, "c_hw_pbe", 'hw')
c_hwcbe_city = dofcity(hw_cbe_sorted, "c_hw_cbe", 'hw')

# Sector averages
isw_pbe_sector_mean = mean_by_sector(isw_pbe_sorted, 'isw')
isw_cbe_sector_mean = mean_by_sector(isw_cbe_sorted, 'isw')
hw_pbe_sector_mean = mean_by_sector(hw_pbe_sorted, 'hw')
hw_cbe_sector_mean = mean_by_sector(hw_cbe_sorted, 'hw')

# ==================== Merge City Results ====================
result = copy.deepcopy(cities)
result = result.rename(columns={'City': 'city'})
for df in [c_iswpbe_city, c_iswcbe_city, c_hwpbe_city, c_hwcbe_city]:
    result = pd.merge(result, df, on='city', how='left')

# ==================== Merge Sector Averages ====================
sector_mean = copy.deepcopy(sector_info[['Sector1']])
sector_mean = sector_mean.rename(columns={'Sector1': 'sector'})

sector_mean = sector_mean.merge(
    isw_pbe_sector_mean.rename(columns={'isw_mean': 'c_isw_pbe'}), on='sector', how='left'
).merge(
    isw_cbe_sector_mean.rename(columns={'isw_mean': 'c_isw_cbe'}), on='sector', how='left'
).merge(
    hw_pbe_sector_mean.rename(columns={'hw_mean': 'c_hw_pbe'}), on='sector', how='left'
).merge(
    hw_cbe_sector_mean.rename(columns={'hw_mean': 'c_hw_cbe'}), on='sector', how='left'
)

# ==================== Save Results ====================
output_file = 'result/CCD_result.xlsx'

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # City Summary
    result.to_excel(writer, sheet_name='city', index=False)

    # PBE/CBE for each city
    isw_pbe_sorted.to_excel(writer, sheet_name='c_isw_pbe', index=False)
    isw_cbe_sorted.to_excel(writer, sheet_name='c_isw_cbe', index=False)
    hw_pbe_sorted.to_excel(writer, sheet_name='c_hw_pbe', index=False)
    hw_cbe_sorted.to_excel(writer, sheet_name='c_hw_cbe', index=False)

    # Merged Sector Averages
    sector_mean.to_excel(writer, sheet_name='sector_mean', index=False)

print("Calculation complete! results saved to:", output_file)
