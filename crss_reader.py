import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from osgeo import gdal
import rasterio
from shapely.geometry import Point, Polygon, LineString
import geopandas as gpd
import fiona
from matplotlib.colors import ListedColormap
import matplotlib.pylab as pl
from skimage import exposure
import seaborn as sns
import sys
import scipy.stats as stats
from datetime import datetime, timedelta

def create_input_data_dictionary(baseline_scenario, adaptive_scenario, folder_name = ''):
  input_data_dictionary = {}
  ###geographic layout
  input_data_dictionary['hydrography'] = 'Shapefiles_UCRB/NHDPLUS_H_1401_HU4_GDB.gdb'
  input_data_dictionary['structures'] = 'Shapefiles_UCRB/Div_5_structures.shp'
  ##basin labels
  input_data_dictionary['HUC4'] = ['1401',]
  input_data_dictionary['HUC8'] = ['14010001', '14010002', '14010003', '14010004', '14010005']
  ##locations of large agricultural aggregations
  ###snow data
  input_data_dictionary['snow'] = 'Snow_Data/'

  ###statemod input data
  ##monthly demand data
  input_data_dictionary['structure_demand'] = 'input_files/cm2015' + baseline_scenario + '.ddm'
  ##water rights data
  input_data_dictionary['structure_rights'] = 'input_files/cm2015' + baseline_scenario + '.ddr'
  ##reservoir fill rights data
  input_data_dictionary['reservoir_rights'] = 'input_files/cm2015' + baseline_scenario + '.rer'
  ##reservoir fill rights data
  input_data_dictionary['instream_rights'] = 'cm2015.ifr'
  ##full natural flow data
  input_data_dictionary['natural flows'] = 'cm2015x.xbm'
  ##flow/node network
  input_data_dictionary['downstream'] = 'input_files/cm2015.rin'
  ##historical reservoir data
  input_data_dictionary['historical_reservoirs'] = 'input_files/cm2015.eom'
  ##call data
  input_data_dictionary['calls'] = 'cm2015' + baseline_scenario + '.xca'

  ###statemod output data
  ##reservoir storage data
  input_data_dictionary['reservoir_storage'] = 'cm2015' + baseline_scenario + '.xre'
  ##diversion data
  input_data_dictionary['deliveries'] = 'cm2015' + baseline_scenario + '.xdd'
  ##return flow data
  input_data_dictionary['return_flows'] = 'cm2015' + baseline_scenario + '.xss'
  ##plan flow data
  input_data_dictionary['plan_releases'] = 'cm2015' + baseline_scenario + '.xpl'
  
  ##adaptive reservoir output data
  input_data_dictionary['reservoir_storage_new'] = folder_name + 'cm2015' + adaptive_scenario + '.xre'
  ##adaptive diversion data
  input_data_dictionary['deliveries_new'] = folder_name + 'cm2015' + adaptive_scenario + '.xdd'
  ##adaptive demand data
  input_data_dictionary['structure_demand_new'] = folder_name + 'cm2015' + adaptive_scenario + '.ddm'
  ##adaptive return flows
  input_data_dictionary['return_flows_new'] = folder_name + 'cm2015' + adaptive_scenario + '.xss'
  ##plan flow data
  input_data_dictionary['plan_releases'] = folder_name + 'cm2015' + adaptive_scenario + '.xpl'

  input_data_dictionary['snow'] = 'Snow_Data/'
  input_data_dictionary['irrigation'] = 'Shapefiles_UCRB/Div5_Irrigated_Lands_2015/Div5_Irrig_2015.shp'
  input_data_dictionary['ditches'] = 'Shapefiles_UCRB/Div5_Irrigated_Lands_2015/Div5_2015_Ditches.shp'
  input_data_dictionary['aggregated_diversions'] = 'output_files/aggregated_diversions.txt'


  return input_data_dictionary

def create_et_benefits():
  marginal_net_benefits = {}
  et_requirements = {}

  marginal_net_benefits['VEGETABLES'] = 506.0
  marginal_net_benefits['ALFALFA'] = 492.0
  marginal_net_benefits['BARLEY'] = 12.0
  marginal_net_benefits['BLUEGRASS'] = 401.0
  marginal_net_benefits['CORN_GRAIN'] = 173.0
  marginal_net_benefits['DRY_BEANS'] = 85.0
  marginal_net_benefits['GRASS_PASTURE'] = 181.0
  marginal_net_benefits['SOD_FARM'] = 181.0
  marginal_net_benefits['SMALL_GRAINS'] = 75.0
  marginal_net_benefits['SORGHUM_GRAIN'] = 311.0
  marginal_net_benefits['WHEAT_FALL'] = 112.0
  marginal_net_benefits['WHEAT_SPRING'] = 112.0

  effective_precip = 3.1
  et_requirements['VEGETABLES'] = 26.2
  et_requirements['ALFALFA'] = 44.0
  et_requirements['BARLEY'] = 22.2
  et_requirements['BLUEGRASS'] = 30.0
  et_requirements['CORN_GRAIN'] = 26.9
  et_requirements['DRY_BEANS'] = 18.1
  et_requirements['GRASS_PASTURE'] = 30.0
  et_requirements['SOD_FARM'] = 30.0
  et_requirements['SMALL_GRAINS'] = 22.2
  et_requirements['SORGHUM_GRAIN'] = 24.5
  et_requirements['WHEAT_FALL'] = 22.2
  et_requirements['WHEAT_SPRING'] = 22.2
  et_requirements['ORCHARD_WITH_COVER'] = 22.2
  et_requirements['ORCHARD_WO_COVER'] = 22.2
  et_requirements['GRAPES'] = 22.2

  total_npv_costs = 0.0
  counter = 0
  grapes_planting_costs = [-6385.0, -2599.0, -1869.0, 754.0, 2012.0, 2133.0, 2261.0] 
  grapes_baseline_revenue = [2261.0, 2261.0, 2261.0, 2261.0, 2261.0, 2261.0, 2261.0]
  for cost, baseline in zip(grapes_planting_costs, grapes_baseline_revenue):
    total_npv_costs +=  (baseline - cost)/np.power(1.025, counter)
    counter += 1
  marginal_net_benefits['GRAPES'] = total_npv_costs

  total_npv_costs = 0.0
  counter = 0
  orchard_planting_costs = [-5183.0, -2802.0, -2802.0, 395.0, 5496.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0] 
  orchard_baseline_revenue = [9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, -5183.0, -2802.0, -2802.0, 395.0, 5496.0]
  for cost, baseline in zip(orchard_planting_costs, orchard_baseline_revenue):
    total_npv_costs +=  (baseline - cost)/np.power(1.025, counter)
    counter += 1
  marginal_net_benefits['ORCHARD_WITH_COVER'] = total_npv_costs
  marginal_net_benefits['ORCHARD_WO_COVER'] = total_npv_costs
  
  return marginal_net_benefits, et_requirements

def read_text_file(filename):
  with open(filename,'r') as f:
    all_split_data = [x for x in f.readlines()]       
  f.close()
  return all_split_data

def make_control_file(old_filename, scenario_name, year_start, year_end):
  with open(old_filename + '.ctl','r') as f:
    all_split_data = [x for x in f.readlines()]       
  f.close()
  f = open(old_filename + scenario_name + '.ctl','w')
  # write firstLine # of rows as in initial file
  for i in range(0, len(all_split_data)):
    if i == 7:
      f.write('    ' + str(year_start) + '     : iystr   STARTING YEAR OF SIMULATION\n')
    elif i == 8:
      f.write('    ' + str(year_end) + '     : iyend   ENDING YEAR OF SIMULATION\n')
    else:
      f.write(all_split_data[i])
  f.close()

  return all_split_data


def initializeDDM(demand_data, filename):
  f = open('cm2015A.ddm','w')
  # write firstLine # of rows as in initial file
  i = 0
  while demand_data[i][0] == '#':
    f.write(demand_data[i])
    i += 1
  f.write(demand_data[i])
  i+=1
def writepartialDDM(demand_data, updated_demands, structures_purchase, structures_buyout, change_month, change_year, begin_year, end_year, scenario_name = 'A', structure_list = 'all'):    
  new_data = []
  use_value = 0
  start_loop = 0
  ii = 0
  while updated_demands[ii][0] == '#':
    ii += 1
  ii += 1
  for i in range(0, len(demand_data)):
    if use_value == 1:
      start_loop = 1
    if demand_data[i][0] != '#':
      use_value = 1
    if start_loop == 1:
      monthly_values = demand_data[i].split('.')
      first_data = monthly_values[0].split()
      use_line = True
      row_data = []
      try:
        year_num = int(first_data[0])
        structure_name = str(first_data[1]).strip()
      except:
        use_line = False
      if use_line:
        if structure_name in structure_list or structure_list == 'all':        
          if year_num >= begin_year and year_num <= end_year:
            this_buyout_structure = structures_buyout[structures_buyout['structure'] == structure_name]
            this_purchase_structure = structures_purchase[structures_purchase['structure'] == structure_name]
            monthly_values_new = updated_demands[ii].split('.')
            first_data_new = monthly_values_new[0].split()
            new_demands = np.zeros(13)
            row_data.extend(first_data_new)
            toggle_use = False
            if year_num == change_year:
              if len(this_buyout_structure) > 0:
                toggle_use = True
                for index, row in this_buyout_structure.iterrows():
                  new_demands[change_month] = row['demand'] * 1.0
              if len(this_purchase_structure) > 0:
                toggle_use = True
                for index, row in this_purchase_structure.iterrows():
                  new_demands[change_month] -= row['demand'] * 1.0
              if toggle_use:
                total_new_demand = 0.0
                if change_month == 0:
                  total_new_demand += new_demands[change_month]
                  row_data[2] = str(int(new_demands[change_month]))
                else:
                  total_new_demand += float(row_data[2])
                for j in range(0, len(monthly_values_new)-3):
                  if j+1 == change_month:
                    total_new_demand += new_demands[change_month]
                    row_data.append(str(int(new_demands[change_month])))
                  else:
                    total_new_demand += float(monthly_values_new[j+1])
                    row_data.append(str(int(float(monthly_values_new[j+1]))))   
                row_data.append(str(int(total_new_demand)))
              else:
                for j in range(len(monthly_values_new)-2):
                  row_data.append(str(int(float(monthly_values_new[j+1]))))
            else:
              for j in range(len(monthly_values_new)-2):
                row_data.append(str(int(float(monthly_values_new[j+1]))))            
          else:
            row_data.extend(first_data)
            for j in range(len(monthly_values)-2):
              row_data.append(str(int(float(monthly_values[j+1]))))
        else:
          row_data.extend(first_data)
          for j in range(len(monthly_values)-2):
            row_data.append(str(int(float(monthly_values[j+1]))))

      else:
        row_data.extend(first_data)
        for j in range(len(monthly_values)-2):
          row_data.append(str(monthly_values[j+1]))
      ii+=1
      new_data.append(row_data)
    # write new data to file
  f = open('cm2015' + scenario_name + '.ddm','w')
  # write firstLine # of rows as in initial file
  j = 0
  while demand_data[j][0] == '#':
    f.write(demand_data[j])
    j += 1
  f.write(demand_data[j])
  j+=1
  for i in range(len(new_data)):
    # write year, ID and first month of adjusted data
    f.write(new_data[i][0] + ' ' + new_data[i][1] + (19-len(new_data[i][1])-len(new_data[i][2]))*' ' + new_data[i][2] + '.')
    # write all but last month of adjusted data
    for j in range(len(new_data[i])-4):
      f.write((7-len(new_data[i][j+3]))*' ' + new_data[i][j+3] + '.')                
        # write last month of adjusted data
    f.write((9-len(new_data[i][-1]))*' ' + new_data[i][-1] + '.' + '\n')
  for j in range(j+len(new_data), len(demand_data)):
    f.write(demand_data[j])
    
  f.close()
    
  return None

def writenewDDM(demand_data, structures_purchase, structures_buyout, change_year, change_month, scenario_name = 'A'):    
  new_data = []
  use_value = 0
  start_loop = 0
  for i in range(0, len(demand_data)):
    if use_value == 1:
      start_loop = 1
    if demand_data[i][0] != '#':
      use_value = 1
    if start_loop == 1:
      monthly_values = demand_data[i].split('.')
      first_data = monthly_values[0].split()
      use_line = True
      row_data = []
      try:
        year_num = int(first_data[0])
        structure_name = str(first_data[1]).strip()
      except:
        use_line = False
      
      if use_line and year_num == change_year:
        this_buyout_structure = structures_buyout[structures_buyout['structure'] == structure_name]
        this_purchase_structure = structures_purchase[structures_purchase['structure'] == structure_name]
        new_demands = np.zeros(13)
        row_data.extend(first_data)
        toggle_use = False
        if len(this_buyout_structure) > 0:
          toggle_use = True
          for index, row in this_buyout_structure.iterrows():
            new_demands[change_month] = row['demand'] * 1.0
        if len(this_purchase_structure) > 0:
          toggle_use = True
          for index, row in this_purchase_structure.iterrows():
            new_demands[change_month] -= row['demand'] * 1.0
        if toggle_use:
          total_new_demand = 0.0
          if change_month == 0:
            total_new_demand += new_demands[change_month]
            row_data[2] = str(int(new_demands[change_month]))
          else:
            total_new_demand += float(row_data[2])
          for j in range(0, len(monthly_values)-3):
            if j+1 == change_month:
              total_new_demand += new_demands[change_month]
              row_data.append(str(int(new_demands[change_month])))
            else:
              total_new_demand += float(monthly_values[j+1])
              row_data.append(str(int(float(monthly_values[j+1]))))   
          row_data.append(str(int(total_new_demand)))
        else:
          for j in range(len(monthly_values)-2):
            row_data.append(str(int(float(monthly_values[j+1]))))    
      elif use_line:
        row_data.extend(first_data)
        for j in range(len(monthly_values)-2):
          row_data.append(str(int(float(monthly_values[j+1]))))
      else:
        row_data.extend(first_data)
        for j in range(len(monthly_values)-2):
          row_data.append(str(monthly_values[j+1]))
      
      new_data.append(row_data)                
    # write new data to file
  f = open('cm2015' + scenario_name + '.ddm','w')
  # write firstLine # of rows as in initial file
  i = 0
  while demand_data[i][0] == '#':
    f.write(demand_data[i])
    i += 1
  f.write(demand_data[i])
  i+=1
  for i in range(len(new_data)):
    # write year, ID and first month of adjusted data
    f.write(new_data[i][0] + ' ' + new_data[i][1] + (19-len(new_data[i][1])-len(new_data[i][2]))*' ' + new_data[i][2] + '.')
    # write all but last month of adjusted data
    for j in range(len(new_data[i])-4):
      f.write((7-len(new_data[i][j+3]))*' ' + new_data[i][j+3] + '.')                
        # write last month of adjusted data
    f.write((9-len(new_data[i][-1]))*' ' + new_data[i][-1] + '.' + '\n')            
  f.close()
    
  return None


def read_rights_data(all_data_DDR, structure_type = 'structure'):
  if structure_type == 'reservoir':
    column_lengths=[12,24,12,16,8,8,8,8,8]
    all_rights_fill_type = []
  else:
    column_lengths=[12,24,12,16,8,8]  
  split_line = ['']*len(column_lengths)
  character_breaks=np.zeros(len(column_lengths),dtype=int)
  character_breaks[0]=column_lengths[0]
  for i in range(1,len(column_lengths)):
    character_breaks[i]=character_breaks[i-1]+column_lengths[i]
  
  all_rights_name = []
  all_rights_priority = []
  all_rights_decree = []
  all_rights_structure_name = []
  for j in range(0,len(all_data_DDR)):
    if all_data_DDR[j][0] == '#':
      first_line = int(j * 1)
    else:
      split_line[0]=all_data_DDR[j][0:character_breaks[0]]
      for i in range(1,len(split_line)):
        split_line[i]=all_data_DDR[j][character_breaks[i-1]:character_breaks[i]]
      structure_name = str(split_line[2].strip())
      right_name = str(split_line[0].strip())
      right_priority = float(split_line[3].strip())
      if int(split_line[5].strip()) == 1:
        right_decree = float(split_line[4].strip())
      else:
        right_decree = 0.0
      if structure_type == 'reservoir':
        fill_type = int(split_line[8].strip())
        all_rights_fill_type.append(fill_type)
        
      all_rights_name.append(right_name)
      all_rights_priority.append(right_priority) 
      all_rights_decree.append(right_decree)
      all_rights_structure_name.append(structure_name)
      if structure_name == '3900563':
        all_rights_name.append(right_name)
        all_rights_priority.append(right_priority) 
        all_rights_decree.append(right_decree)
        all_rights_structure_name.append('3900563_I')
        all_rights_name.append(right_name)
        all_rights_priority.append(right_priority) 
        all_rights_decree.append(right_decree)
        all_rights_structure_name.append('3903505_I')
      if structure_name == '3900663':
        all_rights_name.append(right_name)
        all_rights_priority.append(right_priority) 
        all_rights_decree.append(right_decree)
        all_rights_structure_name.append('3903505_I')
      if structure_name == '7200646':
        all_rights_name.append(right_name)
        all_rights_priority.append(right_priority) 
        all_rights_decree.append(right_decree)
        all_rights_structure_name.append('7200813')
        all_rights_name.append(right_name)
        all_rights_priority.append(right_priority) 
        all_rights_decree.append(right_decree)
        all_rights_structure_name.append('7200646_I')
        all_rights_name.append(right_name)
        all_rights_priority.append(right_priority) 
        all_rights_decree.append(right_decree)
        all_rights_structure_name.append('7200813PP')
        all_rights_name.append(right_name)
        all_rights_priority.append(right_priority) 
        all_rights_decree.append(right_decree)
        all_rights_structure_name.append('7200813PW')
        
  if structure_type == 'reservoir':
    return all_rights_name, all_rights_structure_name, all_rights_priority, all_rights_decree, all_rights_fill_type
  else:
    return all_rights_name, all_rights_structure_name, all_rights_priority, all_rights_decree

def find_historical_max_deliveries(structure_deliveries, structure_id):
  month_list = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
  annual_deliveries = []
  total_annual_deliveries = 0.0
  monthly_deliveries = {}
  for x in month_list:
    monthly_deliveries[x] = []
    
  for index, row in structure_deliveries.iterrows():
    if index.month == 10 and total_annual_deliveries > 0.0:
      annual_deliveries.append(total_annual_deliveries)
      total_annual_deliveries = 0.0
    monthly_deliveries[month_list[index.month - 1]].append(row[structure_id])
    total_annual_deliveries += row[structure_id]
      
  max_annual = np.max(annual_deliveries)
  max_monthly = np.zeros(12)
  for x_cnt, x in enumerate(month_list):
    max_monthly[x_cnt] = np.max(monthly_deliveries[x])
    
  return max_monthly, max_annual

def read_historical_reservoirs(historical_reservoir_data, reservoir_list, initial_year, end_year, year_read = 'all'):
  datetime_index = []
  for year_count in range(initial_year, end_year):
    month_num = 10
    year_add = 0
    for month_count in range(0, 12):
      datetime_index.append(datetime(year_count + year_add, month_num, 1, 0, 0))
      month_num += 1
      if month_num == 13:
        month_num = 1
        year_add = 1
        
  historical_storage = pd.DataFrame(index = datetime_index, columns = [reservoir_list,])
  historical_storage[reservoir_list] = np.zeros(len(datetime_index))
  for i in range(len(historical_reservoir_data)):
    if historical_reservoir_data[i][0] != '#':
      monthly_values = historical_reservoir_data[i].split('.')
      first_data = monthly_values[0].split()
      use_line = True
      try:
        year_num = int(first_data[0])
        structure_name = str(first_data[1])
      except:
        use_line = False
      if use_line and structure_name == reservoir_list:
        datetime_val = datetime(year_num - 1, 10, 1, 0, 0)
        historical_storage.loc[datetime_val, structure_name] = float(first_data[2])
        for month_num in range(0, 11):
          if month_num < 2:
            datetime_val = datetime(year_num - 1, month_num + 11, 1, 0, 0)
          else:
            datetime_val = datetime(year_num, month_num - 1, 1, 0, 0)
          historical_storage.loc[datetime_val, structure_name] = float(monthly_values[month_num + 1])     

  return historical_storage  
  
def read_full_natural_flow(full_natural_flow_data, station_list, initial_year, end_year, year_read = 'all'):
  datetime_index = []
  for year_count in range(initial_year, end_year):
    month_num = 10
    year_add = 0
    for month_count in range(0, 12):
      datetime_index.append(datetime(year_count + year_add, month_num, 1, 0, 0))
      month_num += 1
      if month_num == 13:
        month_num = 1
        year_add = 1
        
  full_natural_flows = pd.DataFrame(index = datetime_index, columns = station_list)
  for x in station_list:
    full_natural_flows[x] = np.zeros(len(datetime_index))
  for i in range(len(full_natural_flow_data)):
    if full_natural_flow_data[i][0] != '#':
      monthly_values = full_natural_flow_data[i].split('.')
      first_data = monthly_values[0].split()
      use_line = True
      try:
        year_num = int(first_data[0])
        structure_name = str(first_data[1])
      except:
        use_line = False
      if use_line and structure_name in station_list:
        datetime_val = datetime(year_num - 1, 10, 1, 0, 0)
        full_natural_flows.loc[datetime_val, structure_name] = float(first_data[2])
        for month_num in range(0, 11):
          if month_num < 2:
            datetime_val = datetime(year_num - 1, month_num + 11, 1, 0, 0)
          else:
            datetime_val = datetime(year_num, month_num - 1, 1, 0, 0)
          full_natural_flows.loc[datetime_val, structure_name] = float(monthly_values[month_num + 1])     

  return full_natural_flows  

def read_structure_demands(demand_data, initial_year, end_year, read_from_file = False):
  if read_from_file:
    structure_demand = pd.read_csv('input_files/demand_by_structure.csv', index_col = 0)
    structure_demand.index = pd.to_datetime(structure_demand.index)
  else:
    datetime_index = []
    for year_count in range(initial_year, end_year):
      month_num = 10
      year_add = 0
      for month_count in range(0, 12):
        datetime_index.append(datetime(year_count + year_add, month_num, 1, 0, 0))
        month_num += 1
        if month_num == 13:
          month_num = 1
          year_add = 1
    year_num = 0
    structure_list = []
    i = 0
    while year_num < 1910:
      if demand_data[i][0] != '#':
        monthly_values = demand_data[i].split('.')
        first_data = monthly_values[0].split()
        use_line = True
        try:
          year_num = int(first_data[0])
          structure_name = str(first_data[1])
          if year_num < 1910:
            structure_list.append(structure_name)
        except:
          use_line = False
      i += 1
    
    structure_demand = pd.DataFrame(index = datetime_index, columns = structure_list)
  
    for x in structure_list:
      structure_demand[x] = np.zeros(len(datetime_index))
    for i in range(len(demand_data)):
      if demand_data[i][0] != '#':
        monthly_values = demand_data[i].split('.')
        first_data = monthly_values[0].split()
        use_line = True
        try:
          year_num = int(first_data[0])
          structure_name = str(first_data[1])
        except:
          use_line = False
        if use_line:
          datetime_val = datetime(year_num - 1, 10, 1, 0, 0)
          structure_demand.loc[datetime_val, structure_name] = float(first_data[2])     
          for month_num in range(0, 11):
            if month_num < 2:
              datetime_val = datetime(year_num - 1, month_num + 11, 1, 0, 0)
            else:
              datetime_val = datetime(year_num, month_num - 1, 1, 0, 0)
            structure_demand.loc[datetime_val, structure_name] = float(monthly_values[month_num + 1])     

    structure_demand.to_csv('input_files/demand_by_structure.csv')
  
  return structure_demand  
  
def read_structure_return_flows(return_flow_data, initial_year, end_year, read_from_file = False):
  if read_from_file:
    structure_return_flows = pd.read_csv('input_files/returns_by_structure.csv', index_col = 0)
    structure_return_flows.index = pd.to_datetime(structure_return_flows.index)

  else:
    counter = 10
    month_num_dict = {}
    month_name_list = ['OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP']
    structure_return_flows = pd.DataFrame(index = month_name_list)
    structure_diversions = pd.DataFrame(index = month_name_list)
    structure_returns = pd.DataFrame(index = month_name_list)
  
    counterii = 0
    counteri = 0
    for line in return_flow_data:
      counterii += 1
      if counterii == 50000:
        counteri += 1
        counterii = 0
      data = line.split()
      if data:
        if len(data) > 1 and data[0] !='#':
          struct_id = str(data[0].strip())
          try:
            month_id = data[2].strip()
            year_id = int(data[1].strip())
            if month_id in month_name_list:
              use_line = True
            else:
              use_line = False
          except:
            use_line = False            
          if use_line:
            if struct_id in structure_diversions.columns:           
              structure_diversions.loc[month_id, struct_id] += float(data[12].strip())
            else:
              structure_diversions[struct_id] = np.zeros(len(month_name_list))
              structure_diversions.loc[month_id, struct_id] += float(data[12].strip())

            if struct_id in structure_returns.columns:           
              structure_returns.loc[month_id, struct_id] += float(data[17].strip())
            else:
              structure_returns[struct_id] = np.zeros(len(month_name_list))
              structure_returns.loc[month_id, struct_id] += float(data[17].strip())
    
    for struct_id in structure_diversions.columns:
      structure_return_flows[struct_id] = np.zeros(len(month_name_list))
      for month_id in month_name_list:
        if structure_diversions.loc[month_id, struct_id] > 0.0:
          structure_return_flows.loc[month_id, struct_id] = structure_returns.loc[month_id, struct_id] / structure_diversions.loc[month_id, struct_id]
              
    structure_return_flows.to_csv('input_files/returns_by_structure.csv')
  
  return structure_return_flows  
  

def read_structure_deliveries(delivery_data, initial_year, end_year, use_list = 'all', read_from_file = False):
  if read_from_file:
    structure_deliveries = pd.read_csv('input_files/deliveries_by_structure.csv', index_col = 0)
    structure_deliveries.index = pd.to_datetime(structure_deliveries.index)

  else:
    counter = 10
    month_num_dict = {}
    for month_name in ['OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP']:
      month_num_dict[month_name] = counter
      counter += 1
      if counter == 13:
        counter = 1

    datetime_index = []
    for year_count in range(initial_year, end_year):
      month_num = 10
      year_add = 0
      for month_count in range(0, 12):
        datetime_index.append(datetime(year_count + year_add, month_num, 1, 0, 0))
        month_num += 1
        if month_num == 13:
          month_num = 1
          year_add = 1
    structure_deliveries = pd.DataFrame(index = datetime_index)
  
    counterii = 0
    counteri = 0
    for line in delivery_data:
      counterii += 1
      if counterii == 50000:
        counteri += 1
        counterii = 0
      data = line.split()
      if data:
        if len(data) > 1 and data[0] !='#':
          struct_id = str(data[0].strip())
          if use_list == 'all' or struct_id in use_list:
            if struct_id == 'Baseflow' or struct_id == 'NA':
              struct_id = str(data[1].strip())
            try:
              month_id = data[3].strip()
              year_id = int(data[2].strip())
              month_number = month_num_dict[month_id]
              use_line = True
            except:
              use_line = False            
            if use_line:             
              datetime_val = datetime(year_id, month_number, 1, 0, 0)            
              structure_deliveries.loc[datetime_val, struct_id] = float(data[16].strip()) - float(data[15].strip())
    if use_list == 'all':            
      structure_deliveries.to_csv('input_files/deliveries_by_structure.csv')
  
  return structure_deliveries  



def read_plan_flows(plan_data, initial_year, end_year, read_from_file = False):

  if read_from_file:
    plan_deliveries = pd.read_csv('input_files/plan_deliveries_by_structure.csv', index_col = 0)
    plan_deliveries.index = pd.to_datetime(plan_deliveries.index)

  else:
    counter = 10
    month_num_dict = {}
    for month_name in ['OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP']:
      month_num_dict[month_name] = counter
      counter += 1
      if counter == 13:
        counter = 1

    datetime_index = []
    for year_count in range(initial_year, end_year):
      month_num = 10
      year_add = 0
      for month_count in range(0, 12):
        datetime_index.append(datetime(year_count + year_add, month_num, 1, 0, 0))
        month_num += 1
        if month_num == 13:
          month_num = 1
          year_add = 1
    plan_deliveries = pd.DataFrame(index = datetime_index)
  
    plan_toggle = 0
    for line in plan_data:
      if plan_toggle == 0:
        data = line.split()
        if data:
          if data[0] == 'River' and data[1] == 'Location':
            oop_source = data[3]
            plan_toggle = 1
            oop_destination_list = []
      elif plan_toggle == 1:
        data = line.split()
        if data:
          if len(data) > 10:
            if data[0] == 'ID':
              for x in oop_destination_list:
                plan_deliveries[x + '_diversion'] = np.zeros(len(datetime_index))
              plan_deliveries[oop_source + '_release'] = np.zeros(len(datetime_index))
              plan_toggle = 2
            else:
              for x in range(0, len(data)):
                if data[x] == 'Destination':
                  oop_destination_list.append(data[x + 2])
      elif plan_toggle == 2:
        data = line.split()
        extract_data = False
        if data:
          if len(data) > 1:
            if data[0] == 'Plan' and data[1] == 'Summary':
              plan_toggle = 0
            else:
              extract_data = True
          else:
            extract_data = True
        else:
          extract_data = True
        if extract_data:
          monthly_values = line.split('.')
          if len(monthly_values) > 0:
            first_data = monthly_values[0].split()
            use_line = True
            if len(monthly_values) > 1 and first_data[0] !='#':
              try:
                month_id = first_data[3].strip()
                year_id = int(first_data[2].strip())
                month_number = month_num_dict[month_id]
              except:
                use_line = False
              if use_line:
                struct_id = first_data[0].strip()
                datetime_val = datetime(year_id, month_number, 1, 0, 0)
                counter = 0
                total_div = 0.0
                for oop_dest in oop_destination_list:
                  if oop_dest == 'Various' or oop_dest == '0':
                    skip_this = 1
                  else:
                    if len(monthly_values) == 25:
                      plan_deliveries.loc[datetime_val, oop_dest + '_diversion'] += float(monthly_values[counter + 2])
                      total_div += float(monthly_values[counter + 2])
                    elif len(monthly_values) == 23:
                      plan_deliveries.loc[datetime_val, oop_dest + '_diversion'] += float(monthly_values[counter+1])
                      total_div += float(monthly_values[counter+1])
                    counter += 1
                    if counter == 20:
                      break
                plan_deliveries.loc[datetime_val, oop_source + '_release'] += total_div
                            
    plan_deliveries.to_csv('input_files/plan_deliveries_by_structure.csv')
      
  return plan_deliveries  
  

def update_structure_demands(demand_data, update_year, update_month, read_from_file = False):

  structure_demand = {}
  for i in range(len(demand_data)):
    if demand_data[i][0] != '#':
      monthly_values = demand_data[i].split('.')
      first_data = monthly_values[0].split()
      use_line = True
      try:
        year_num = int(first_data[0])
        structure_name = str(first_data[1])
      except:
        use_line = False
      if use_line and year_num == update_year:
        if update_month == 10:
          structure_demand[structure_name] = float(first_data[2])
        elif update_month == 11 or update_month == 12:
          structure_demand[structure_name] = float(monthly_values[update_month - 10])
        else:
          structure_demand[structure_name] = float(monthly_values[update_month + 2])
  
  return structure_demand  
  
def update_structure_outflows(delivery_data, update_year, update_month, read_from_file = False):
  counter = 10
  month_num_dict = {}
  for month_name in ['OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP']:
    month_num_dict[month_name] = counter
    counter += 1
    if counter == 13:
      counter = 1

  structure_outflows = {}
  for i in range(len(delivery_data)):
    use_line = True
    monthly_values = delivery_data[i].split('.')
    first_data = monthly_values[0].split()
    if len(first_data) > 3:
      use_line = True
      try:
        month_id = first_data[3].strip()
        year_number = int(first_data[2].strip())
        month_number = month_num_dict[month_id]
      except:
        use_line = False
      if use_line:
        if year_number == update_year and month_number == update_month:
          struct_id = str(first_data[0].strip())
          if struct_id == 'Baseflow' or struct_id == 'NA':
            struct_id = str(first_data[1].strip())

          structure_outflows[struct_id] = float(monthly_values[30].strip())
                    
  return structure_outflows  

def update_structure_deliveries(delivery_data, update_year, update_month, read_from_file = False):

  counter = 10
  month_num_dict = {}
  for month_name in ['OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP']:
    month_num_dict[month_name] = counter
    counter += 1
    if counter == 13:
      counter = 1

  structure_deliveries = {}
  for line in delivery_data:
    data = line.split()
    if data:
      use_line = True
      if len(data) > 1 and data[0] !='#':
        try:
          month_id = data[3].strip()
          year_number = int(data[2].strip())
          month_number = month_num_dict[month_id]
        except:
          use_line = False
        if use_line:
          if year_number == update_year and month_number == update_month:
            struct_id = str(data[0].strip())
            if struct_id == 'Baseflow' or struct_id == 'NA':
              struct_id = str(data[1].strip())

            structure_deliveries[struct_id] = float(data[16].strip()) - float(data[15].strip())
                    
  return structure_deliveries  
  
def update_plan_flows(reservoir_storage_data_b, reservoir_list, update_year, update_month):
        
  month_name_dict = {}
  counter = 10
  for month_name in ['OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP']:
    month_name_dict[month_name] = counter
    counter += 1
    if counter == 13:
      counter = 1
      
  new_plan_flows = {}
  for i in range(len(reservoir_storage_data_b)):
    use_line = True
    monthly_values = reservoir_storage_data_b[i].split('.')
    first_data = monthly_values[0].split()
    if len(first_data) > 1:
      structure_name = str(first_data[0])
      if structure_name in reservoir_list:
        try:
          account_num = int(first_data[1])
          year_num = int(first_data[2])
          month_num = month_name_dict[str(first_data[3])]
        except:
          use_line = False
        if use_line and account_num == 0 and year_num == update_year and month_num == update_month:
          datetime_val = datetime(year_num, month_num, 1, 0, 0)
          new_plan_flows[structure_name] = float(monthly_values[19])
  
  return new_plan_flows

def read_simulated_diversions(delivery_data, structure_list, initial_year, end_year):
  counter = 10
  month_num_dict = {}
  for month_name in ['OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP']:
    month_num_dict[month_name] = counter
    counter += 1
    if counter == 13:
      counter = 1

  datetime_index = []
  for year_count in range(initial_year, end_year):
    month_num = 10
    year_add = 0
    for month_count in range(0, 12):
      datetime_index.append(datetime(year_count + year_add, month_num, 1, 0, 0))
      month_num += 1
      if month_num == 13:
        month_num = 1
        year_add = 1
  simulated_diversions = pd.DataFrame(index = datetime_index, columns = structure_list)
  for x in structure_list:
    simulated_diversions[x] = np.zeros(len(datetime_index))
  for line in delivery_data:
    data = line.split()
    if data:
      if len(data) > 1 and data[0] !='#':
        struct_id = str(data[1].strip())
        if struct_id in structure_list:
          use_line = True
          try:
            month_id = data[3].strip()
            year_id = int(data[2].strip())
            month_number = month_num_dict[month_id]
          except:
            use_line = False            
          if use_line:             
            datetime_val = datetime(year_id, month_number, 1, 0, 0)            
            simulated_diversions.loc[datetime_val, struct_id] = float(data[19].strip())

  return simulated_diversions
  
def read_simulated_control_release_single(delivery_data, reservoir_data, update_year, update_month, use_list = 'all'):

  counter = 10
  month_num_dict = {}
  for month_name in ['OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP']:
    month_num_dict[month_name] = counter
    counter += 1
    if counter == 13:
      counter = 1

  simulated_releases = {}
  for line in reservoir_data:
    monthly_values = line.split('.')
    if len(monthly_values) > 0:
      first_data = monthly_values[0].split()
      if len(first_data) > 1:
        struct_id = str(first_data[1].strip())
        if use_list == 'all' or struct_id in use_list:
          use_line = True
          if len(monthly_values) > 1 and first_data[0] !='#':
            try:
              month_id = first_data[3].strip()
              year_id = int(first_data[2].strip())
              month_number = month_num_dict[month_id]
            except:
              use_line = False
            if use_line:
              if month_number == update_month and year_id == update_year:
                struct_id = first_data[0].strip()
                if str(first_data[1].strip()) == '0':
                  simulated_releases[struct_id + '_flow'] = float(monthly_values[18].strip())
                  simulated_releases[struct_id + '_diverted'] = float(monthly_values[20].strip()) + float(monthly_values[5].strip()) + float(monthly_values[6].strip())
            
            
  for line in delivery_data:
    monthly_values = line.split('.')
    if len(monthly_values) > 0:
      first_data = monthly_values[0].split()
      use_line = True
      if len(monthly_values) > 1 and first_data[0] !='#':
        try:
          month_id = first_data[3].strip()
          year_id = int(first_data[2].strip())
          month_number = month_num_dict[month_id]
        except:
          use_line = False
        if use_line:
          if month_number == update_month and year_id == update_year:
            struct_id = str(first_data[1].strip())
            last_values = monthly_values[29].split()
            simulated_releases[struct_id + '_location'] = str(last_values[0].strip())
            simulated_releases[struct_id + '_available'] = float(monthly_values[28].strip())
            simulated_releases[struct_id + '_physical_supply'] = float(monthly_values[24].strip()) - float(monthly_values[28].strip()) - max(float(monthly_values[25].strip()), 0.0)
            simulated_releases[struct_id + '_outflow'] = float(monthly_values[27].strip())

  return simulated_releases

def read_simulated_control_release(delivery_data, reservoir_data, initial_year, end_year, use_list = 'all'):
  counter = 10
  month_num_dict = {}
  for month_name in ['OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP']:
    month_num_dict[month_name] = counter
    counter += 1
    if counter == 13:
      counter = 1

  datetime_index = []
  for year_count in range(initial_year, end_year):
    month_num = 10
    year_add = 0
    for month_count in range(0, 12):
      datetime_index.append(datetime(year_count + year_add, month_num, 1, 0, 0))
      month_num += 1
      if month_num == 13:
        month_num = 1
        year_add = 1
  
  simulated_releases = pd.DataFrame(index = datetime_index)
  for line in reservoir_data:
    monthly_values = line.split('.')
    if len(monthly_values) > 0:
      first_data = monthly_values[0].split()
      if len(monthly_values) > 1 and first_data[0] !='#':
        use_line = True
        struct_id = first_data[0].strip()
        if use_list == 'all' or struct_id in use_list:
          try:
            month_id = first_data[3].strip()
            year_id = int(first_data[2].strip())
            month_number = month_num_dict[month_id]
          except:
            use_line = False            
          if use_line:
            if str(first_data[1].strip()) == '0':
              datetime_val = datetime(year_id, month_number, 1, 0, 0)       
              simulated_releases.loc[datetime_val, struct_id + '_flow'] = float(monthly_values[18].strip())
              simulated_releases.loc[datetime_val, struct_id + '_diverted'] = float(monthly_values[20].strip()) + float(monthly_values[5].strip()) + float(monthly_values[6].strip())
            
            
  for line in delivery_data:
    monthly_values = line.split('.')
    if len(monthly_values) > 0:
      first_data = monthly_values[0].split()
      if len(monthly_values) > 1 and first_data[0] !='#':
        use_line = True
        struct_id = first_data[0].strip()
        if struct_id == 'Baseflow' or struct_id == 'NA':
          struct_id = first_data[1].strip()
        if use_list == 'all' or struct_id in use_list:
          try:
            month_id = first_data[3].strip()
            year_id = int(first_data[2].strip())
            month_number = month_num_dict[month_id]
          except:
            use_line = False            
          if use_line:             
            datetime_val = datetime(year_id, month_number, 1, 0, 0)
            last_values = monthly_values[29].split()
            simulated_releases.loc[datetime_val, struct_id + '_location'] = str(last_values[0].strip())
            simulated_releases.loc[datetime_val, struct_id + '_available'] = float(monthly_values[28].strip())
            simulated_releases.loc[datetime_val, struct_id + '_physical_supply'] = float(monthly_values[24].strip()) - float(monthly_values[28].strip()) - max(float(monthly_values[25].strip()), 0.0)
            simulated_releases.loc[datetime_val, struct_id + '_outflow'] = float(monthly_values[27].strip())
            
  return simulated_releases

def compare_storage_scenarios(reservoir_data_baseline, reservoir_data_adaptive, diversion_current_demand, comp_year, comp_month, storage_id, diversion_id, storage_right):
  month_name_dict = {}
  counter = 10
  for month_name in ['OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP']:
    month_name_dict[month_name] = counter
    counter += 1
    if counter == 13:
      counter = 1
  for i in range(len(reservoir_data_baseline)):
    use_line = True
    monthly_values = reservoir_data_baseline[i].split('.')
    first_data = monthly_values[0].split()
    if len(first_data) > 3:
      try:
        year_num = int(first_data[2])
      except:
        use_line = False
      if use_line:
        if year_num == comp_year:
          try:
            month_num = month_name_dict[str(first_data[3])]
          except:
            use_line = False
          if use_line:
            if month_num == comp_month:
              try:
                account_num = int(first_data[1])
              except:
                use_line = False
              if use_line:              
                if account_num == 0:
                  try:
                    structure_name = str(first_data[0])
                  except:
                    use_line = False
                  if use_line:
                    if structure_name == storage_id:
                      initial_value = float(monthly_values[15]) 
                      break
                      
  for i in range(len(reservoir_data_adaptive)):
    use_line = True
    monthly_values = reservoir_data_adaptive[i].split('.')
    first_data = monthly_values[0].split()
    if len(first_data) > 3:
      try:
        year_num = int(first_data[2])
      except:
        use_line = False
      if use_line:
        if year_num == comp_year:
          try:
            month_num = month_name_dict[str(first_data[3])]
          except:
            use_line = False
          if use_line:
            if month_num == comp_month:
              try:
                account_num = int(first_data[1])
              except:
                use_line = False
              if use_line:              
                if account_num == 0:
                  try:
                    structure_name = str(first_data[0])
                  except:
                    use_line = False
                  if use_line:
                    if structure_name == storage_id:
                      adaptive_value = float(monthly_values[15]) 
                      break
                      
  storage_change = min(initial_value - adaptive_value, 0.0)
    
  change_points_df = pd.DataFrame()    
  change_points_df['structure'] = [diversion_id,]
  change_points_df['demand'] = [storage_change,]
  change_points_df['right'] = [storage_right,]
  change_points_df['consumptive'] = [1.0,]
  change_points_df['date'] = [datetime(comp_year, comp_month, 1, 0, 0),]
  
  change_points_buyout_df = pd.DataFrame()
  change_points_buyout_df['structure'] = [diversion_id,]
  change_points_buyout_df['demand'] = [diversion_current_demand,]
  change_points_buyout_df['demand_purchase'] = [0.0,]
  change_points_buyout_df['date'] = [datetime(comp_year, comp_month, 1, 0, 0),]


  return change_points_df, change_points_buyout_df, adaptive_value


def compare_res_diversion_scenarios(reservoir_data_baseline, reservoir_data_adaptive, start_year, end_year, storage_id):

  datetime_index = []
  for year_count in range(start_year - 1, end_year + 1):
    month_num = 10
    year_add = 0
    for month_count in range(0, 12):
      datetime_index.append(datetime(year_count + year_add, month_num, 1, 0, 0))
      month_num += 1
      if month_num == 13:
        month_num = 1
        year_add = 1
  
  reservoir_diversions = pd.DataFrame(index = datetime_index, columns = [storage_id + '_res_diversion',])
  reservoir_diversions[storage_id + '_res_diversion'] = np.zeros(len(datetime_index))
  month_name_dict = {}
  counter = 10
  for month_name in ['OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP']:
    month_name_dict[month_name] = counter
    counter += 1
    if counter == 13:
      counter = 1
  for i in range(len(reservoir_data_baseline)):
    use_line = True
    monthly_values = reservoir_data_baseline[i].split('.')
    first_data = monthly_values[0].split()
    if len(first_data) > 3:
      try:
        year_num = int(first_data[2])
      except:
        use_line = False
      if use_line:
        if year_num >= start_year and year_num <= end_year:
          try:
            month_num = month_name_dict[str(first_data[3])]
          except:
            use_line = False
          if use_line:
            try:
              account_num = int(first_data[1])
            except:
              use_line = False
            if use_line:              
              if account_num == 0:
                try:
                  structure_name = str(first_data[0])
                except:
                  use_line = False
                if use_line:
                  if structure_name == storage_id:
                    datetime_val = datetime(year_num, month_num, 1, 0, 0)
                    reservoir_diversions.loc[datetime_val, storage_id + '_res_diversion'] -= (float(monthly_values[20]) - float(monthly_values[19]))
                      
  for i in range(len(reservoir_data_adaptive)):
    use_line = True
    monthly_values = reservoir_data_adaptive[i].split('.')
    first_data = monthly_values[0].split()
    if len(first_data) > 3:
      try:
        year_num = int(first_data[2])
      except:
        use_line = False
      if use_line:
        if year_num >= start_year and year_num <= end_year:
          try:
            month_num = month_name_dict[str(first_data[3])]
          except:
            use_line = False
          if use_line:
            try:
              account_num = int(first_data[1])
            except:
              use_line = False
            if use_line:              
              if account_num == 0:
                try:
                  structure_name = str(first_data[0])
                except:
                  use_line = False
                if use_line:
                  if structure_name == storage_id:
                    datetime_val = datetime(year_num, month_num, 1, 0, 0)
                    reservoir_diversions.loc[datetime_val, storage_id + '_res_diversion'] += (float(monthly_values[20]) - float(monthly_values[19]))
                    reservoir_diversions.loc[datetime_val, storage_id + '_res_diversion'] = max(reservoir_diversions.loc[datetime_val, storage_id + '_res_diversion'], 0.0)

  return reservoir_diversions

def read_plan_flows_2(reservoir_data, reservoir_list, initial_year, end_year):
  datetime_index = []
  for year_count in range(initial_year, end_year):
    month_num = 10
    year_add = 0
    for month_count in range(0, 12):
      datetime_index.append(datetime(year_count + year_add, month_num, 1, 0, 0))
      month_num += 1
      if month_num == 13:
        month_num = 1
        year_add = 1
        
  month_name_dict = {}
  counter = 10
  for month_name in ['OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP']:
    month_name_dict[month_name] = counter
    counter += 1
    if counter == 13:
      counter = 1
      
  simulated_storage = pd.DataFrame(index = datetime_index, columns = reservoir_list)
  for res in reservoir_list:
    simulated_storage[res] = np.zeros(len(datetime_index))
  for i in range(len(reservoir_data)):
    use_line = True
    monthly_values = reservoir_data[i].split('.')
    first_data = monthly_values[0].split()
    if len(first_data) > 1:
      structure_name = str(first_data[0])
      if structure_name in reservoir_list:
        try:
          account_num = int(first_data[1])
          year_num = int(first_data[2])
          month_num = month_name_dict[str(first_data[3])]
        except:
          use_line = False
        if use_line and account_num == 0:
          datetime_val = datetime(year_num, month_num, 1, 0, 0)
          simulated_storage.loc[datetime_val, structure_name] = float(monthly_values[19])

  return simulated_storage  


def read_simulated_reservoirs(reservoir_data, reservoir_list, initial_year, end_year, year_read = 'all'):
  datetime_index = []
  for year_count in range(initial_year, end_year):
    month_num = 10
    year_add = 0
    for month_count in range(0, 12):
      datetime_index.append(datetime(year_count + year_add, month_num, 1, 0, 0))
      month_num += 1
      if month_num == 13:
        month_num = 1
        year_add = 1
        
  month_name_dict = {}
  counter = 10
  for month_name in ['OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP']:
    month_name_dict[month_name] = counter
    counter += 1
    if counter == 13:
      counter = 1
      
  simulated_storage = pd.DataFrame(index = datetime_index, columns = [reservoir_list,])
  simulated_storage[reservoir_list] = np.zeros(len(datetime_index))
  for i in range(len(reservoir_data)):
    use_line = True
    monthly_values = reservoir_data[i].split('.')
    first_data = monthly_values[0].split()
    if len(first_data) > 1:
      structure_name = str(first_data[0])
      if structure_name == reservoir_list:
        try:
          account_num = int(first_data[1])
          year_num = int(first_data[2])
          month_num = month_name_dict[str(first_data[3])]
        except:
          use_line = False
        if use_line and account_num == 0:
          datetime_val = datetime(year_num, month_num, 1, 0, 0)
          simulated_storage.loc[datetime_val, structure_name] = float(first_data[4])
          simulated_storage.loc[datetime_val, structure_name + '_diversions'] = float(monthly_values[8])
        else:
          datetime_val = datetime(year_num, month_num, 1, 0, 0)
          simulated_storage.loc[datetime_val, structure_name + '_account_' + str(account_num)] = float(first_data[4])
          simulated_storage.loc[datetime_val, structure_name + '_diversions_' + str(account_num)] = float(monthly_values[8])

  return simulated_storage  

def update_simulated_reservoirs(reservoir_data, update_year, update_month, use_list = 'all', year_read = 'all'):
      
  simulated_storage = {}
  
  month_name_dict = {}
  counter = 10
  for month_name in ['OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP']:
    month_name_dict[month_name] = counter
    counter += 1
    if counter == 13:
      counter = 1

  for i in range(len(reservoir_data)):
    use_line = True
    monthly_values = reservoir_data[i].split('.')
    first_data = monthly_values[0].split()
    if len(first_data) > 1:
      structure_name = str(first_data[0])
      if use_list == 'all' or structure_name in use_list:
        try:
          account_num = int(first_data[1])
          year_num = int(first_data[2])
          month_num = month_name_dict[str(first_data[3])]
        except:
          use_line = False
        if use_line and account_num == 0:
          if update_year == year_num and update_month == month_num:
            simulated_storage[structure_name] = float(first_data[4])
            simulated_storage[structure_name + '_diversions'] = float(monthly_values[8])
          else:
            simulated_storage[structure_name + '_account_' + str(account_num)] = float(first_data[4])
            simulated_storage[structure_name + '_diversions_' + str(account_num)] = float(monthly_values[8])

  return simulated_storage  


def read_rights(downstream_data, column_lengths, station_name):
  split_line = ['']*len(column_lengths)
  character_breaks=np.zeros(len(column_lengths),dtype=int)
  character_breaks[0]=column_lengths[0]
  for i in range(1,len(column_lengths)):
    character_breaks[i]=character_breaks[i-1]+column_lengths[i]

  id_list = []
  upstream_list = []
  downstream_pairs = {}
  for j in range(0,len(downstream_data)):
    if downstream_data[j][0] == '#':
      first_line = int(j * 1)
    else:
      split_line[0]=downstream_data[j][0:character_breaks[0]]
      for i in range(1,len(split_line)):
        split_line[i]=downstream_data[j][character_breaks[i-1]:character_breaks[i]]
      if split_line[2].strip() in downstream_pairs:        
        downstream_pairs[split_line[2].strip()].append(split_line[0].strip())
      else:
        downstream_pairs[split_line[2].strip()] = [split_line[0].strip(),]
  upstream_list = []
  upstream_list.extend(self.find_upstream_list(station_name, downstream_pairs))


      
def read_full_natural_flows(self, station_number, all_split_data_XBM = 'none'):
  if all_split_data_XBM == 'none':
    full_natural_flows_all = pd.read_csv('UCRB_analysis-master/Sobol_sample/Experiment_files/full_natural_flows.csv', index_col = 0)
    full_natural_flows = full_natural_flows_all[station_number]
  else:
    for j in range(0, len(all_split_data_XBM)):
      if all_split_data_DDM[j][0] == '#':
        toggle_on = 1
      elif toggle_on == 1:
        first_line = int(j * 1)
        toggle_on = 0    
      else:
        this_row = all_split_data_XBM[j].split('.')
        row_data = []
        row_data.extend(this_row[0].split())
        start_year_index = (int(row_data[0].strip()) - 1909) * 12
        structure_name = str(row_data[1].strip())
        self.structures_objects[structure_name].monthly_fnf[start_year_index] = float(row_data[2].strip())
        for x in range(1, 12):
            self.structures_objects[structure_name].monthly_fnf[start_year_index + x] = float(this_row[x].strip())

def find_upstream_list(lookup_station, downstream_pairs):
  upstream_list = []
  for x in downstream_pairs[lookup_station]:
    upstream_list.extend(self.find_upstream_list(x))
    
  return upstream_list



def writeDDM(demand_filename, structures, reduced_demand, new_demand, structure_receive, year_change, month_change):  
  with open(demand_filename,'r') as f:
    all_split_data_DDM = [x.split('.') for x in f.readlines()]       
  f.close()        
  # get unsplit data to rewrite firstLine # of rows
  first_line_toggle = 0
  fl_counter = 0
  with open(demand_filename,'r') as f:
    all_data_DDM = [x for x in f.readlines()]
    if x[0] != '#' and first_line_toggle == 0:
      firstLine = fl_counter
      first_line_toggle = 1
    fl_counter += 1      
  f.close()
  
  allstructures = []
  for m in range(len(structures)):
    allstructures.extend(structures[m])
  new_data = []

  f = open(demand_filename,'w')
  for i in range(firstLine):
    f.write(all_data_DDM[i])            

  for i in range(len(all_split_data_DDM)-firstLine):
    row_data = []
    # To store the change between historical and sample irrigation demand (12 months + Total)
    change = np.zeros(13) 
    # Split first 3 columns of row on space
    # This is because the first month is lumped together with the year and the ID when spliting on periods
    row_data.extend(all_split_data_DDM[i+firstLine][0].split())
    # If the structure is not in the ones we care about then do nothing
    if int(row_data[0]) == year_change and row_data[1] in structures: #If the structure is irrigation
      total_demand = 0.0
      for j in range(len(all_split_data_DDM[i+firstLine])-2):
        if j == month_change - 1:
          row_data.append(str(int(reduced_demand[row_data[1]])))
          total_demand += int(reduced_demand[row_data[1]])
        else:
          row_data.append(str(int(float(all_split_data_DDM[i+firstLine][j+1]))))
          total_demand += int(float(all_split_data_DDM[i+firstLine][j+1]))
      row_data.append(str(total_demand))      
      f.write(row_data[0] + ' ' + row_data[1] + (19-len(row_data[1])-len(row_data[2]))*' ' + row_data[2] + '.')
      for j in range(len(row_data)-4):
        f.write((7-len(row_data[j+3]))*' ' + row_data[j+3] + '.')                
      f.write((9-len(new_data[i][-1]))*' ' + new_data[i][-1] + '.' + '\n')            
      
    else:
      f.write(all_data_DDM[i+firstLine])            
  f.close()
      
  return None

  
def writeCTL(control_filename, row_change, year_end):  
  with open(control_filename,'r') as f:
    all_data_CTL = [x for x in f.readlines()]       
  f.close()
  
  allstructures = []
  for m in range(len(structures)):
    allstructures.extend(structures[m])
  new_data = []

  f = open(control_filename,'w')
  for i in range(0,row_change):
    f.write(all_data_CTL[i])            
  
  f.write(all_data_CTL[row_change][0:4])
  f.write(str(year_end))
  f.write(all_data_CTL[row_change][8:])
  for i in range(row_change + 1, len(all_data_CTL)):
    f.write(all_data_CTL[i])            
  
  f.close()
  return None