import requests
import csv
import json
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import io
import numpy as np
import os

project_folder = 'UCRB_analysis-master/'
station_types = ['SNOW', 'SNOWTEL']
station_filenames = [project_folder + 'colorado_snow_stations.csv', project_folder + 'colorado_snow_courses.csv']
station_titles = ['Station', 'station_name']
basin_titles = ['Hydrologic_Unit', 'basin']
snow_stations = pd.read_csv(project_folder + 'colorado_snow_stations.csv')
snotel_stations = pd.read_csv(project_folder + 'colorado_snow_courses.csv')
filename_base = 'https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customSingleStationReport/monthly/start_of_period/'
filename_mid = ':CO:SNTL'
filename_mid2 = ':CO:SNOW'
filename_end = '%7Cid=""%7Cname/POR_BEGIN,POR_END/WTEQ::value,SNWD::value,PREC::value,TOBS::value,TMAX::value,TMIN::value,TAVG::value?fitToScreen=false'

#Turn month abbreviations used in snotel dates into integers
headerlines = 3
snowtel_month_order = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
snowtel_month_count = {}
counter = 0
for x in snowtel_month_order:
  snowtel_month_count[x] = counter
  counter += 1
 
##Create Datetime Index for scraped data 
year_start = 1908
df_dict = {}
total_annual_accumulation = {}
monthly_date_index = []
end_of_year_boolean = {}
for t in range(0, 12):
  end_of_year_boolean[str(t)] = []
for x in range(1908, 2021):
  month_count = 10
  year_add = 0
  for t in range(0, 12):
    monthly_date_index.append(datetime(x + year_add, month_count, 1, 0, 0))
    for tt in range(0, 12):
      if tt == t:
        end_of_year_boolean[str(tt)].append(True)
      else:
        end_of_year_boolean[str(tt)].append(False)
    month_count += 1
    if month_count == 13:
      month_count = 1
      year_add = 1

#each loop variable here has two potential values, one for the SNOW sites and one for the SNOTEL sites
for station_type, station_filename, station_title, basin_title  in zip(station_types, station_filenames, station_titles, basin_titles):
  #read all stations (either the snow list or the snotel list)
  snow_stations = pd.read_csv(station_filename)
  #loop through all the stations in each list

  for index, row in snow_stations.iterrows():
  
    #get the station name and basin name so the results can be organized
    #into a single .csv file per basin, with each station in the basin
    #filling one column of monthly data in the .csv    
    if station_type == 'SNOW':
      station_name = row[station_title]
    else:
      station_name = ''
      for x in row[station_title]:
        if x.isdigit():
          station_name += x
    basin_name = ''
    for x in row[basin_title]:
      if x.isdigit():
        basin_name += x
    basin_name = basin_name[0:8]
    
    #make API link
    toggle_use = 1
    if station_type == 'SNOW':
      if row['Ntwk'] == 'SNOW':
        response = requests.get(filename_base + station_name + filename_mid2 + filename_end)
      else:
        toggle_use = 0
    else:
      response = requests.get(filename_base + station_name + filename_mid + filename_end)
      
    if toggle_use == 1:  
      print(station_name)      
      #initialize data values for the station with (-999), so we know when the data begins
      #(different for every station)
      if basin_name in df_dict:
        df_dict[basin_name][station_name] = np.ones((2021 - year_start)* 12) * -999
        total_annual_accumulation[basin_name][station_name] = []
      else:
        df_dict[basin_name] = pd.DataFrame(index = monthly_date_index)
        df_dict[basin_name][station_name] = np.ones((2021 - year_start)* 12) * -999
        total_annual_accumulation[basin_name] = {}
        total_annual_accumulation[basin_name][station_name] = []

      #read api data  
      counter = 0
      prev_mnth = 0
      for line in response.text.splitlines():
        if line[0] == '#':
          skip_line = 1
        else:
          counter += 1
          if counter >= headerlines:
            entries = line.split(',')
            date_str = entries[0]
            
            month_cnt = snowtel_month_count[date_str[0:3]]
            if month_cnt < 3:
              year_cnt = int(date_str[4:8]) - year_start
            else:
              year_cnt = int(date_str[4:8]) - year_start - 1
            if month_cnt == 0:
              prev_mnth = 0
            if len(entries[1]) > 0:
              df_dict[basin_name][station_name][year_cnt*12 + month_cnt] = max(float(entries[1]), prev_mnth)
              prev_mnth = max(float(entries[1]), prev_mnth) 
            else:
              df_dict[basin_name][station_name][year_cnt*12 + month_cnt] = max(0.0, prev_mnth)
              prev_mnth = max(0.0, prev_mnth)
            #in addition to the data dictionary, we keep a list of total annual accumulation
            #for each station
            if month_cnt == 11:
              total_annual_accumulation[basin_name][station_name].append(prev_mnth)
            
for basin_name in df_dict:
  #get average total accumulation for each station
  station_avs = {}
  for station_name in df_dict[basin_name]:
    station_avs[station_name] = np.mean(np.asarray(total_annual_accumulation[basin_name][station_name]))
    
  #at each time-step, we calculate the snowpack as the % of average total accumulation
  #then average together all of the station percentages
  basinwide_average = []
  for index, row in df_dict[basin_name].iterrows():
    mean_pct_ave = []
    for station_name in df_dict[basin_name]:
      if row[station_name] > -1.0 and station_avs[station_name] > 0.0:
        mean_pct_ave.append(row[station_name] / station_avs[station_name])
    if len(mean_pct_ave) > 0:
      basinwide_average.append(np.mean(np.asarray(mean_pct_ave)))
    else:
      basinwide_average.append(-999.9)
  
  df_dict[basin_name]['basinwide_average'] = basinwide_average
  df_dict[basin_name].to_csv(project_folder + 'Adaptive_experiment/Snow_Data/' + basin_name[0:8] + '.csv')





