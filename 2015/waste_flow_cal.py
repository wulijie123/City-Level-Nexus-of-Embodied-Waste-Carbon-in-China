# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 22:40:19 2026

@author: 26496
"""

import pandas as pd
import numpy as np
import copy
import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

os.chdir('../2015')
Z = pd.read_csv('data/MRIO_sector20/Z.csv', header=None).values  # 6260×6260
F = pd.read_csv('data/MRIO_sector20/F.csv', header=None).values  # 6260×1565
F_city = pd.read_csv('data/MRIO_sector20/F_city.csv', header=None).values  # 6260×313
Y = pd.read_csv('data/MRIO_sector20/Y.csv', header=None).values.flatten()  # 6260
VA = pd.read_csv('data/MRIO_sector20/VA.csv', header=None).values.flatten()  # 6260
EX = pd.read_csv('data/MRIO_sector20/ex.csv', header=None).values.flatten()  # 6260
A = pd.read_csv('data/MRIO_sector20/A.csv', header=None).values  # 6260×6260
B = pd.read_csv('data/MRIO_sector20/B.csv', header=None).values  # 6260×6260
df_city = pd.read_excel('data/2015_city_variate.xlsx')

# Compute actual Y2 = B * (F + EX)
F_total = F_city.sum(axis=1) + EX  # F (including exports), 6260×1
Y2 = B @ F_total 

waste_data = pd.read_excel('data/2015_waste_data.xlsx')  # Carbon in million tons, others in tons

hazardous_waste = waste_data['HW'].values  # Hazardous waste amount (6260,)
solid_waste = waste_data['ISW'].values  # Solid waste amount (6260,)
carbon_emissions = waste_data['Carbon'].values  # Carbon emissions (6260,)

# Set waste data corresponding to Y2 = 0 to 0
hazardous_waste[Y2 == 0] = 0
solid_waste[Y2 == 0] = 0
carbon_emissions[Y2 == 0] = 0

city_info = pd.read_csv('data/City_Name.csv', encoding='gbk')
city_names = city_info['City'].tolist()[:313]  

sector_info = pd.read_excel('data/sector20.xlsx')  # Sector name mapping
sector_names = sector_info['Sector1'].tolist()  

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
    intensity = waste_amount / Y2
    intensity = np.nan_to_num(intensity, nan=0.0, posinf=0.0, neginf=0.0)
    intensity = intensity.reshape(1, 6260)
    i_diag = np.diagflat(intensity)
    
    # Calculate emission coefficient matrix
    emission_coeff = i_diag @ B
    
    # Calculate final demand diagonal matrix
    f_diag = np.diag(F_total)
    
    # Calculate emission matrix
    emissions = emission_coeff @ f_diag
    
    # Save emission coefficients and emission matrix
    pd.DataFrame(emission_coeff).to_csv(os.path.join(save_path, 'emission_coeff.csv'), index=None, header=None)
    pd.DataFrame(emissions).to_csv(os.path.join(save_path, 'emissions.csv'), index=None, header=None)
    
    #========================================================== Calculate emissions by city demand
    emissions_by_city = emission_coeff @ F_city  # 6260×313
    pd.DataFrame(emissions_by_city).to_csv(os.path.join(save_path, 'city_emissions.csv'), index=None, header=None)

    # Calculate inter-city emission transfer summary (i.e., merge industry info from emissions_by_city)
    emissions_by_city_merged = np.zeros([313, 313])
    for k in range(313):
        emissions_by_city_merged[k] = emissions_by_city[k*20:(k+1)*20, :].sum(axis=0)
    
    emissions_by_city_merged_df = pd.DataFrame(emissions_by_city_merged)
    emissions_by_city_merged_df.columns = city_names
    emissions_by_city_merged_df['city'] = city_names
    emissions_by_city_merged_df.set_index('city', inplace=True)
    
    emissions_by_city_merged_df.to_excel(os.path.join(save_path, f'{prefix}_city_emissions_transfer.xlsx'))
    
    # Calculate inter-city transfer matrix
    transfer_matrix = pd.DataFrame(columns=range(313), index=range(313))
    for i in range(313): # Consumer city
        for u in range(313): # Producer city
            transfer_matrix.iloc[u, i] = emissions_by_city[(20*u):(20+20*u), i].sum()  # u's emission due to i's consumption (i→u)
    
    # Calculate net transfer matrix net[u,i] = transfer[u,i] - transfer[i,u]  (net transfer from u to i)
    net_transfer = transfer_matrix - transfer_matrix.transpose()
    net_transfer.columns = city_names
    net_transfer['city'] = city_names
    net_transfer.set_index('city', inplace=True)
    net_transfer['sum'] = net_transfer.sum(axis=1)  # Rows where sum > 0 indicate that pollution is transferred to you from others
    net_transfer.to_excel(os.path.join(save_path, f'{prefix}_netTransfer_between_cities.xlsx'))
    
    city_net_transfer = pd.DataFrame(columns=['city','net_flow'])  # Net transfer dataframe
    city_net_transfer['city'] = [str(i) for i in city_names]
    city_net_transfer['net_flow'] = net_transfer['sum'].values 
    
    # ============================================== Calculate emissions by production and consumption
    # Emissions from production side
    production_by_city = np.zeros(313)
    for i in range(313):
        production_by_city[i] = waste_amount[i*20:(i+1)*20].sum()
    
    # Emissions from consumption side
    consumption_by_city = np.zeros(313)
    for i in range(313):
        consumption_by_city[i] = emissions_by_city[:, i].sum()  # Sum across columns
    
    # Summary table for production and consumption
    production_consumption = pd.DataFrame({
        'city': city_names,
        'pbe': production_by_city,
        'cbe': consumption_by_city
    })
    
    #============================================================ Calculate emission intensity (per GDP and per capita)
    intensity_by_city = df_city.copy
    
    # Merge emission intensity
    intensity_by_city = pd.merge(df_city, city_net_transfer, on='city') 
    intensity_by_city = pd.merge(intensity_by_city, production_consumption, on='city')  # Add net transfer and production/consumption variables
    
    intensity_by_city['pbe_intensity'] = intensity_by_city['pbe'] / intensity_by_city['GDP']
    intensity_by_city['cbe_intensity'] = intensity_by_city['cbe'] / intensity_by_city['GDP']
    intensity_by_city['pbe_per_capita'] = intensity_by_city['pbe'] / intensity_by_city['pop']
    intensity_by_city['cbe_per_capita'] = intensity_by_city['cbe'] / intensity_by_city['pop']
    
    intensity_by_city.to_excel(os.path.join(save_path, 'city_emissions_and_intensity.xlsx'), index=None)
    
    #============================================================================ Emission breakdown by sector
    # Breakdown production side by sector
    production_by_sector = pd.DataFrame(columns=['city', 'sector', 'pbe', 'total'])
    for i in range(313):
        city_total = production_by_city[i]
        for j in range(20):
            a = i * 20 + j
            production_by_sector.loc[a, 'sector'] = sector_names[j]
            production_by_sector.loc[a, 'pbe'] = waste_amount[a]
            production_by_sector.loc[a, 'city'] = city_names[i]
            production_by_sector.loc[a, 'total'] = city_total
    
    production_by_sector.to_excel(os.path.join(save_path, 'Production_by_sector.xlsx'), index=None)
    
    # Consumption side sector breakdown, including "exports"
    F_city_with_export = np.hstack([F_city, EX.reshape(-1, 1)])  # Add export column
    f_city_se_with_export = pd.DataFrame(columns=range(6260 + 20))  # Update columns to include "export"
    
    for j in range(314):  # Including export
        fcity = F_city_with_export[:, j]  # Get the demand for city j (including export)
        for l in range(20):  # Decompose by sector
            lst = [0] * 6260  # Create a zero array for each city's demand by sector
            index_lst = [a * 20 + l for a in range(313)]  # Indices for the l-th sector in all cities (including export)
            for k in index_lst:
                lst[k] = fcity[k]  # Keep only the demand for this sector
            f_city_se_with_export[j * 20 + l] = lst  # Add the demand for this sector to the result
    
    # Compute consumption side emissions by sector
    f_city_se_array_with_export = f_city_se_with_export.values
    cbe_se_with_export = emission_coeff @ f_city_se_array_with_export  # Emissions for each city and sector
    cbe_sum_with_export = cbe_se_with_export.sum(axis=0)  # Sum by column for total emissions for each city and sector (6260, 1)
    
    # Summarize to city level, including "export"
    cbe_ci_with_export = []
    for k in range(314):  # Including export as a new city
        ind_lst = [k * 20 + u for u in range(20)]
        cbe_ci_with_export.append(cbe_sum_with_export[ind_lst].sum())  # Sum of each city's sectors (313, 1)
    
    # Save consumption side sector breakdown, including "export"
    cbe_se_df_with_export = pd.DataFrame(columns=['city', 'sector', 'cbe'])
    city_list_with_export = []
    for k in range(314):  # Including export as a new city
        city_list_with_export.extend([city_names[k]] * 20 if k < 313 else ['Export'] * 20)  # Export as a city
    
    cbe_se_df_with_export['city'] = city_list_with_export
    cbe_se_df_with_export['sector'] = [sector_names[i % 20] for i in range(6280)]  # Total 6280 rows
    cbe_se_df_with_export['cbe'] = cbe_sum_with_export
    
    cbe_se_df_with_export.to_excel(os.path.join(save_path, 'Consumption_by_sector_with_export.xlsx'), index=None)
    
    #================================================================== Industry transfer to cities
    emissions_by_city_with_export = emission_coeff @ F_city_with_export  # Emission matrix (including exports)
    emissions_by_city_merged_with_export = np.zeros((20, 314))  # 20 industries, 313 cities (including export)

    for j in range(313):  # 313 cities
        for i in range(20):  # 20 industries
            emissions_by_city_merged_with_export[i, :] += emissions_by_city_with_export[j * 20 + i, :]  # Merge emissions for each industry

    # Column names: city names + "export"
    column_names = city_names + ["Export"]
    wide_df = pd.DataFrame(emissions_by_city_merged_with_export, index=sector_names, columns=column_names)  # Wide format (industry x cities + export)

    # Convert to long format
    wide_df_reset = wide_df.reset_index()  # Reset index so 'sector' becomes a column
    wide_df_reset = wide_df_reset.rename(columns={'index': 'sector'})  # Rename the index column to 'sector'

    long_df = pd.melt(wide_df_reset, id_vars=['sector'], value_vars=column_names, var_name='city', value_name='cbe')
    long_df = long_df[['sector', 'city', 'cbe']]

    long_df.to_excel(os.path.join(save_path, 'Industry_flow_to_cities_with_export.xlsx'), index=False)
    
    #================================================================== Net transfer broken down by industry
    # Create city-to-city labels
    city_pairs = []
    for k in range(len(city_names)):  # Starting city
        for j in range(k + 1, len(city_names)):  # Target city
            city_pairs.append(f'{city_names[k]}→{city_names[j]}')
    
    # Compute net transfer by industry for each city pair
    net_transfer_by_sector = pd.DataFrame()  
    cnt = 0  # Counter for city pairs
    for s in range(313):  # Starting city
        for l in range(s + 1, 313):  # Target city
            s_l_20 = emissions_by_city[(20*s):(20+20*s), l] - emissions_by_city[(20*l):(20+20*l), s]  # Net transfer vector (20 industries)
            net_transfer_by_sector[cnt] = s_l_20  # Store in the corresponding column
            cnt += 1
    
    net_transfer_by_sector.columns = city_pairs
    net_transfer_by_sector_transposed = net_transfer_by_sector.transpose()
    net_transfer_by_sector_transposed.columns = sector_names
    net_transfer_by_sector_transposed['sum'] = net_transfer_by_sector_transposed.sum(axis=1)
    
    # Summarize each city's net transfer by industry
    city_net_transfer_summary = pd.DataFrame()
    for s in range(313):
        net_transfer_by_industry = np.zeros(20)  # Net transfer vector by industry for each city
        for l in range(313):
            net_transfer_by_industry += emissions_by_city[(20*s):(20+20*s), l] - emissions_by_city[(20*l):(20+20*l), s]
        city_net_transfer_summary[s] = net_transfer_by_industry
    
    city_net_transfer_summary.columns = city_names
    city_net_transfer_summary_transposed = city_net_transfer_summary.transpose()
    city_net_transfer_summary_transposed.columns = sector_names
    city_net_transfer_summary_transposed['NetOutflow'] = city_net_transfer_summary_transposed.sum(axis=1)
    
    # Calculate industry contribution percentage
    industry_contribution_percentage = pd.DataFrame()
    for g in range(20):
        industry_contribution_percentage[city_net_transfer_summary_transposed.columns.tolist()[g]] = city_net_transfer_summary_transposed[city_net_transfer_summary_transposed.columns.tolist()[g]] / city_net_transfer_summary_transposed['NetOutflow'] * 100
    industry_contribution_percentage['NetOutflow'] = city_net_transfer_summary_transposed['NetOutflow']
    
    net_transfer_by_sector_transposed.to_excel(os.path.join(save_path, f'CityPair_{prefix}_netTransfer_industry_contribution.xlsx'))
  
    excel_filename = os.path.join(save_path, f'{prefix}_CityTransfer_by_industry_summary.xlsx')
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        city_net_transfer_summary_transposed.to_excel(writer, sheet_name=f'{prefix}Transfer')
        industry_contribution_percentage.to_excel(writer, sheet_name=f'{prefix}Transfer_percentage')
    
    print(f"  {save_path} results saved.")
    
    return {'emission_coeff': emission_coeff, 'emissions': emissions, 'emissions_by_city': emissions_by_city, 'transfer_matrix': transfer_matrix, 'net_transfer': net_transfer, 'city_net_transfer': city_net_transfer}

#%%
# Calculate hazardous waste
print("Calculating hazardous waste flow...")
hw_results = calculate_waste_flow(hazardous_waste, 'result/result_hw/', 'hw')

# Calculate industry solid waste
print("Calculating industry solid waste flow...")
sw_results = calculate_waste_flow(solid_waste, 'result/result_isw/', 'isw')

# Calculate carbon
print("Calculating carbon flow...")
carbon_results = calculate_waste_flow(carbon_emissions, 'result/result_c/', 'c')

print("\nAll calculations are complete!")
