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


def read_text_file(filename):
  with open(filename,'r') as f:
    all_split_data = [x for x in f.readlines()]       
  f.close()
  return all_split_data

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
        
  historical_storage = pd.DataFrame(index = datetime_index, columns = reservoir_list)
  for x in reservoir_list:
    historical_storage[x] = np.zeros(len(datetime_index))
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
      if use_line and structure_name in reservoir_list:
        datetime_val = datetime(year_num - 1, 10, 1, 0, 0)
        historical_storage.loc[datetime_val, structure_name] = float(first_data[2])
        for month_num in range(0, 11):
          if month_num < 2:
            datetime_val = datetime(year_num - 1, month_num + 11, 1, 0, 0)
          else:
            datetime_val = datetime(year_num, month_num - 1, 1, 0, 0)
          historical_storage.loc[datetime_val, structure_name] = float(monthly_values[month_num + 1])     

  return historical_storage  

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

def read_simulated_control_release(delivery_data, structure_list, initial_year, end_year):
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
  
  release_list = []
  for x in structure_list:
    release_list.append(x + '_free')
    release_list.append(x + '_controlled')
    release_list.append(x + '_location')
    release_list.append(x + '_flow')
  simulated_releases = pd.DataFrame(index = datetime_index, columns = release_list)
  for x in structure_list:
    simulated_releases[x + '_free'] = np.zeros(len(datetime_index))
    simulated_releases[x + '_controlled'] = np.zeros(len(datetime_index))
    simulated_releases[x + '_flow'] = np.zeros(len(datetime_index))
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
            simulated_releases.loc[datetime_val, struct_id + '_free'] = float(data[32].strip())
            simulated_releases.loc[datetime_val, struct_id + '_controlled'] = float(data[31].strip())
            simulated_releases.loc[datetime_val, struct_id + '_location'] = data[33].strip()
            simulated_releases.loc[datetime_val, struct_id + '_flow'] = float(data[28].strip())

  return simulated_releases


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
      
  simulated_storage = pd.DataFrame(index = datetime_index, columns = reservoir_list)
  for x in reservoir_list:
    simulated_storage[x] = np.zeros(len(datetime_index))
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
          simulated_storage.loc[datetime_val, structure_name] = float(first_data[4])
        else:
          datetime_val = datetime(year_num, month_num, 1, 0, 0)
          simulated_storage.loc[datetime_val, structure_name + '_account_' + str(account_num)] = float(first_data[4])
        

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