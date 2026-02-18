# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 21:24:28 2026

@author: DELL
"""

import pandas as pd
import numpy as np
import copy
import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

os.chdir('../2019')
Z = pd.read_csv('data/2019MRIO/MRIO_sector20/Z.csv', header=None).values  # 6240×6240
F = pd.read_csv('data/2019MRIO/MRIO_sector20/F.csv', header=None).values  # 6240×1560
F_city = pd.read_csv('data/2019MRIO/MRIO_sector20/F_city.csv', header=None).values  # 6240×312
Y = pd.read_csv('data/2019MRIO/MRIO_sector20/Y.csv', header=None).values.flatten()  # 6240
VA = pd.read_csv('data/2019MRIO/MRIO_sector20/VA.csv', header=None).values.flatten()  # 6240
EX = pd.read_csv('data/2019MRIO/MRIO_sector20/ex.csv', header=None).values.flatten()  # 6240
A = pd.read_csv('data/2019MRIO/MRIO_sector20/A.csv', header=None).values  # 6240×6240
B = pd.read_csv('data/2019MRIO/MRIO_sector20/B.csv', header=None).values  # 6240×6240
df_city = pd.read_excel('data/2019_city_variate.xlsx')

# Calculate actual Y2 = B * (F + EX)
F_total = F_city.sum(axis=1) + EX  # F (including export), 6240×1
Y2 = B @ F_total  # 6240×6240 @ 6240×1 = 6240×1
# pd.DataFrame(Y2).to_excel('Y2.xlsx', index=False, header=False)

waste_data = pd.read_csv('result/inventory/2019_waste_data.csv')  # Carbon in million tons, others in tons

hw = waste_data['HW'].values  # Hazardous Waste (6240,)
sw = waste_data['ISW'].values  # Industrial Solid Waste (6240,)
c = waste_data['Carbon'].values  # Carbon Emissions (6240,)

# Set Y2 as 0 where corresponding waste data is 0
hw[Y2 == 0] = 0
sw[Y2 == 0] = 0
c[Y2 == 0] = 0

city_info = pd.read_csv('data/City_Name.csv', encoding='gbk')
city_info = city_info[city_info['City'] != 'Laiwu']  # In 2019, Laiwu was merged into Jinan
city_names = city_info['City'].tolist()[:312]  

hy1 = pd.read_excel('data/sector20.xlsx')  
sector_names = hy1['Sector1'].tolist()  

#%%

def calculate_waste_flow(waste_amount, save_path, waste_type='hw'):
    if waste_type == 'hw':
        prefix = 'HW'
    elif waste_type == 'isw':
        prefix = 'ISW'
    else:
        prefix = 'Carbon' 
        
    os.makedirs(save_path, exist_ok=True)
    
    # Calculate waste intensity
    ht_i = waste_amount / Y2
    ht_i = np.nan_to_num(ht_i, nan=0.0, posinf=0.0, neginf=0.0)
    ht_i = ht_i.reshape(1, 6240)
    i_diag = np.diagflat(ht_i)
    
    # Calculate emission coefficient matrix
    emission_coeff = i_diag @ B
    
    # Calculate final demand diagonal matrix
    f_diag = np.diag(F_total)
    
    # Calculate emission matrix
    emissions = emission_coeff @ f_diag
    
    # Save emission coefficients and emission matrix
    pd.DataFrame(emission_coeff).to_csv(os.path.join(save_path, 'emission_coeff.csv'), index=None, header=None)
    pd.DataFrame(emissions).to_csv(os.path.join(save_path, 'emissions.csv'), index=None, header=None)
    
    #========================================================== Calculate emissions by city
    emissions_by_city = emission_coeff @ F_city  # 6240×312
    pd.DataFrame(emissions_by_city).to_csv(os.path.join(save_path, 'city_emissions.csv'), index=None, header=None)

    # Calculate city-to-city emission transfer (aggregate the industry info of emissions_by_city)
    emissions_by_city_merged = np.zeros([312, 312])
    for k in range(312):
        emissions_by_city_merged[k] = emissions_by_city[k*20:(k+1)*20, :].sum(axis=0)
    
    em_f_city_merge_df = pd.DataFrame(emissions_by_city_merged)
    em_f_city_merge_df.columns = city_names
    em_f_city_merge_df['city'] = city_names
    em_f_city_merge_df.set_index('city', inplace=True)
    
    em_f_city_merge_df.to_excel(os.path.join(save_path, f'{prefix}_city_emission_transfer.xlsx'))
    
    # Calculate city-to-city transfer matrix
    trans = pd.DataFrame(columns=range(312), index=range(312))
    for i in range(312): # Consumer cities
        for u in range(312): # Producer cities
            trans.iloc[u, i] = emissions_by_city[(20*u):(20+20*u), i].sum()  # Emission from i to u (i→u)
    
    # Calculate net transfer matrix: net[u,i] = trans[u,i] - trans[i,u]  (net emission transfer from u to i)
    net = trans - trans.transpose()
    net.columns = city_names
    net['city'] = city_names
    net.set_index('city', inplace=True)
    net['sum'] = net.sum(axis=1)  ## Positive sum means emissions are transferred to you
    net.to_excel(os.path.join(save_path, f'{prefix}_net_transfer_between_cities.xlsx'))
    
    citynet=pd.DataFrame(columns=['city','netflow']) # Net transfer dataframe
    citynet['city']=[str(i) for i in city_names]
    citynet['netflow']= net['sum'].values 
    
    # ============================================== Calculate production and consumption emissions
    # Production emissions
    production_by_city = np.zeros(312)
    for i in range(312):
        production_by_city[i] = waste_amount[i*20:(i+1)*20].sum()
    
    # Consumption emissions
    consumption_by_city = np.zeros(312)
    for i in range(312):
        consumption_by_city[i] = emissions_by_city[:, i].sum()  # Sum across columns
    
    # Production and consumption summary
    production_consumption = pd.DataFrame({
        'city': city_names,
        'pbe': production_by_city,
        'cbe': consumption_by_city
    })
    
    #============================================================ Calculate emission intensity (per GDP and per capita)
    inten_city = df_city.copy
    
    # Merge emission intensity data
    inten_city = pd.merge(df_city, citynet, on='city') 
    inten_city = pd.merge(inten_city, production_consumption, on='city')  # Add net transfer and production-consumption variables
    
    inten_city['pbeintensity'] = inten_city['pbe'] / inten_city['GDP']
    inten_city['cbeintensity'] = inten_city['cbe'] / inten_city['GDP']
    inten_city['pbeperca'] = inten_city['pbe'] / inten_city['pop']
    inten_city['cbeperca'] = inten_city['cbe'] / inten_city['pop']
    
    inten_city.to_excel(os.path.join(save_path, 'city_emissions_and_intensity.xlsx'), index=None)
    
    #============================================================================ Breakdown emissions by sector
    # Production side breakdown by sector
    pbe_sector = pd.DataFrame(columns=['city', 'sector', 'pbe', 'sum'])
    for i in range(312):
        city_total = production_by_city[i]
        for j in range(20):
            a = i * 20 + j
            pbe_sector.loc[a, 'sector'] = sector_names[j]
            pbe_sector.loc[a, 'pbe'] = waste_amount[a]
            pbe_sector.loc[a, 'city'] = city_names[i]
            pbe_sector.loc[a, 'sum'] = city_total
    
    pbe_sector.to_excel(os.path.join(save_path, 'Production_by_sector.xlsx'), index=None)
    
    # Consumption side breakdown by sector, including "Exports"
    F_city_with_export = np.hstack([F_city, EX.reshape(-1, 1)])  # Add export column
    f_city_se_with_export = pd.DataFrame(columns=range(6240 + 20))  # Update columns to include "Exports"
    
    for j in range(313):  # Including exports
        fcity = F_city_with_export[:, j]  # Get demand for the j-th city (including export)
        for l in range(20):  # Decompose by each sector
            lst = [0] * 6240  # Create a zero array for each city's demand in each sector
            indexlst = [a * 20 + l for a in range(312)]  # Index for the l-th sector of all cities (including export)
            for k in indexlst:
                lst[k] = fcity[k]  # Keep only the demand for the l-th sector
            f_city_se_with_export[j * 20 + l] = lst  # Add the demand for the l-th sector
    
    # Calculate consumption side sector emissions
    f_city_se_array_with_export = f_city_se_with_export.values
    cbe_se_with_export = emission_coeff @ f_city_se_array_with_export  # Emissions for each city and sector
    cbe_sum_with_export = cbe_se_with_export.sum(axis=0)  # Sum across columns, giving total emissions per city and sector (6260, 1)
    
    # Aggregate by city level, including "Exports"
    cbe_ci_with_export = []
    for k in range(313):  # Including export as a new city
        indlst = [k * 20 + u for u in range(20)]
        cbe_ci_with_export.append(cbe_sum_with_export[indlst].sum())  # Sum of emissions for each city
    
    # Save consumption side sector breakdown, including "Exports"
    cbe_se_df_with_export = pd.DataFrame(columns=['city', 'sector', 'cbe'])
    city_list_with_export = []
    for k in range(313):  # Including export as a new city
        city_list_with_export.extend([city_names[k]] * 20 if k < 312 else ['Export'] * 20)  # Export as a city
    
    cbe_se_df_with_export['city'] = city_list_with_export
    cbe_se_df_with_export['sector'] = [sector_names[i % 20] for i in range(6260)]  # Total 6260 rows
    cbe_se_df_with_export['cbe'] = cbe_sum_with_export
    
    cbe_se_df_with_export.to_excel(os.path.join(save_path, 'Consumption_by_sector_with_export.xlsx'), index=None)
    
    #================================================================== Industry transfer to cities
    em_f_city_ex = emission_coeff @ F_city_with_export  # Emission matrix (including export)
    em_f_city_ex_merged = np.zeros((20, 313))  # 20 sectors, 313 cities (including export)

    # Two-level loop: outer loop for cities, inner loop for sectors
    for j in range(312):  # 312 cities
        for i in range(20):  # 20 sectors
            em_f_city_ex_merged[i, :] += em_f_city_ex[j * 20 + i, :]  # Aggregate emissions for each sector across cities

    # Column names: city names + "Export"
    column_names = city_names + ["Export"]
    wide_df = pd.DataFrame(em_f_city_ex_merged, index=sector_names, columns=column_names)  # Wide format (sector × cities + export)

    # Convert to long format
    wide_df_reset = wide_df.reset_index()
    wide_df_reset = wide_df_reset.rename(columns={'index': 'sector'})
    long_df = pd.melt(wide_df_reset, id_vars=['sector'], value_vars=column_names, var_name='city', value_name='cbe')  # New columns
    long_df = long_df[['sector', 'city', 'cbe']]

    long_df.to_excel(os.path.join(save_path, 'Industry_flow_to_cities_with_export.xlsx'), index=False)
    
    #================================================================== Net transfer by industry decomposition
    # Create city pair labels
    lstcs = []
    for k in range(len(city_names)):  # Starting city
        for j in range(k + 1, len(city_names)):  # Target city
            astr = city_names[k] + '→' + city_names[j]
            lstcs.append(astr)
    
    # Calculate industry net transfer for each city pair
    trans_se = pd.DataFrame()  
    cnt = 0  # Counter for the processed city pairs
    for s in range(312):  # Starting city
        for l in range(s + 1, 312):  # Target city
            s_l_20 = emissions_by_city[(20*s):(20+20*s), l] - emissions_by_city[(20*l):(20+20*l), s]  # Net transfer vector (20 sectors)
            trans_se[cnt] = s_l_20  # Store in the corresponding column
            cnt += 1
    
    trans_se.columns = lstcs
    trans_se1 = trans_se.transpose()
    trans_se1.columns = sector_names
    trans_se1['sum'] = trans_se1.sum(axis=1)
    
    # Aggregate net transfer by industry for each city
    cshz = pd.DataFrame()
    for s in range(312):
        s_20 = np.zeros(20)  # Aggregate net transfer vector for each city (20 sectors)
        for l in range(312):
            s_20 += emissions_by_city[(20*s):(20+20*s), l] - emissions_by_city[(20*l):(20+20*l), s]
        cshz[s] = s_20
    
    cshz.columns = city_names
    cshz1 = cshz.transpose()
    cshz1.columns = sector_names
    cshz1['Netoutflow'] = cshz1.sum(axis=1)
    
    # Calculate industry contribution percentages
    cshz2 = pd.DataFrame()
    for g in range(20):
        cshz2[cshz1.columns.tolist()[g]] = cshz1[cshz1.columns.tolist()[g]] / cshz1['Netoutflow'] * 100
    cshz2['Netoutflow'] = cshz1['Netoutflow']
    
    trans_se1.to_excel(os.path.join(save_path, f'CityPair_{prefix}_netTransfer_industry_contribution.xlsx'))
  
    excel_filename = os.path.join(save_path, f'{prefix}_CityTransfer_by_industry_summary.xlsx')
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        cshz1.to_excel(writer, sheet_name=f'{prefix}Transfer')
        cshz2.to_excel(writer, sheet_name=f'{prefix}Transfer_percentage')
    
    print(f"Results saved to {save_path}")
    
    return {'emission_coeff': emission_coeff, 'emissions': emissions, 'emissions_by_city': emissions_by_city, 'trans': trans, 'net': net, 'citynet': citynet}

#%%
# Calculate hazardous waste
print("Calculating hazardous waste flow...")
hw_results = calculate_waste_flow(hw, 'result/result_hw/', 'hw')

# Calculate industrial solid waste
print("Calculating industrial solid waste flow...")
sw_results = calculate_waste_flow(sw, 'result/result_isw/', 'isw')

# Calculate carbon emissions
print("Calculating carbon flow...")
carbon_results = calculate_waste_flow(c, 'result/result_c/', 'c')

print("\nAll calculations are complete!")
