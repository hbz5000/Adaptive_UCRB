from osgeo import gdal, osr
from osgeo.gdalconst import *
import os
import matplotlib.pyplot as plt
import rasterio as rio
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, LineString, mapping
import shapely.ops as ops
import geopandas as gpd
import math
import fiona
from matplotlib.colors import ListedColormap
import matplotlib.pylab as pl
import rasterio
import rasterio.plot
from rasterio.mask import mask
from rasterio.merge import merge
import pyproj
from mapper import Mapper
import seaborn as sns
import imageio
from matplotlib.patches import Patch
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from datetime import datetime
from geopy.geocoders import Nominatim
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score
import geopy.distance
from functools import partial
import json
from PyPDF2 import PdfFileReader
import tabula
from copy import copy
from basin import Basin
import crss_reader as crss
from plotter import Plotter

####Create Filename Dictionary####
input_data_dictionary = crss.create_input_data_dictionary('B', 'A')

print('create basin')
#Create Basin class with the same extent as StateMod Basin
#as defined by the input_data_dictionary files
project_folder = 'UCRB_analysis-master/'
ucrb = Basin(input_data_dictionary)
year_start = 1908
year_start_adaptive = 1950
year_end = 2013

reservoir_transfer_to = '5104055'
tunnel_transfer_to = '5104634'

##Initialize basin reservoirs
##there are other reservoirs in StateMod - these are just the 
##places where we want to look at water supply metrics
ucrb.reservoir_list = []
ucrb.create_reservoir('Green Mountain', '3603543', 154645.0)
ucrb.create_reservoir('Dillon', '3604512', 257000.0)
ucrb.create_reservoir('Homestake', '3704516', 43600.0)
ucrb.create_reservoir('Wolford', '5003668', 65985.0)
ucrb.create_reservoir('Williams Fork', '5103709', 96822.0)
ucrb.create_reservoir('Granby', '5104055', 539758.0)

os.system("StateMod_Model_15.exe cm2015B -simulate")        
##load input file data
#load river network
print('load input data')
downstream_data = crss.read_text_file(input_data_dictionary['downstream'])
#load historical reservoir data
historical_reservoir_data = crss.read_text_file(input_data_dictionary['historical_reservoirs'])
#load 'baseline' reservoir data
reservoir_storage_data_b = crss.read_text_file(input_data_dictionary['reservoir_storage'])
#load 'baseline' rights data
reservoir_rights_data = crss.read_text_file(input_data_dictionary['reservoir_rights'])
structure_rights_data = crss.read_text_file(input_data_dictionary['structure_rights'])
#load 'baseline' demand and deliveries data
demand_data = crss.read_text_file(input_data_dictionary['structure_demand'])
delivery_data = crss.read_text_file(input_data_dictionary['deliveries'])

#create baseline timeseries for reservoir storage - compare to historical observations
print('create historical reservoir timeseries')
for res in [reservoir_transfer_to,]:
  #historical
  ucrb.structures_objects[res].historical_reservoir_timeseries = crss.read_historical_reservoirs(historical_reservoir_data, res, year_start, year_end)
  #baseline statemod
  ucrb.structures_objects[res].simulated_reservoir_timeseries = crss.read_simulated_reservoirs(reservoir_storage_data_b, res, year_start, year_end)
  ucrb.structures_objects[res].adaptive_reservoir_timeseries = ucrb.structures_objects[res].simulated_reservoir_timeseries.copy(deep = True)
  
print('apply rights to structures')
#load rights data for structures + reservoirs
reservoir_rights_name, reservoir_rights_structure_name, reservoir_rights_priority, reservoir_rights_decree, reservoir_fill_rights = crss.read_rights_data(reservoir_rights_data, structure_type = 'reservoir')
structure_rights_name, structure_rights_structure_name, structure_rights_priority, structure_rights_decree = crss.read_rights_data(structure_rights_data)
#using rights data from inputs, the 'basin' class creates structure & reservoir objects
#in each structure/reservoir object, there are one or more 'rights' objects
ucrb.set_rights_to_reservoirs(reservoir_rights_name, reservoir_rights_structure_name, reservoir_rights_priority, reservoir_rights_decree, reservoir_fill_rights)
ucrb.set_rights_to_structures(structure_rights_name, structure_rights_structure_name, structure_rights_priority, structure_rights_decree)
#create 'rights stack' - all rights listed in the order of their priority, w/ structure names, decree amounts, etc.
ucrb.combine_rights_data(structure_rights_name, structure_rights_structure_name, structure_rights_priority, structure_rights_decree, reservoir_rights_name, reservoir_rights_structure_name, reservoir_rights_priority, reservoir_rights_decree)

print('apply demands to structures')
#read demand data and apply to structures
structure_demands = crss.read_structure_demands(demand_data,year_start, year_end, read_from_file = False)
ucrb.set_structure_demands(structure_demands)

print('apply deliveries to structures')
#read delivery data and apply to structures
structure_deliveries = crss.read_structure_deliveries(delivery_data, year_start, year_end, read_from_file = False)
ucrb.set_structure_deliveries(structure_deliveries)

monthly_maximums = ucrb.adjust_structure_deliveries()
baseline_release_timeseries = crss.read_simulated_control_release(delivery_data, reservoir_storage_data_b, ucrb.reservoir_list, year_start, year_end)
adaptive_release_timeseries = baseline_release_timeseries.copy(deep = True)
adaptive_release_timeseries.index = pd.to_datetime(adaptive_release_timeseries.index)
#calculate water supply metrics (inc. snowpack & storage) at each of the indentified reservoirs


print('set structure types')
structures_ucrb = gpd.read_file(input_data_dictionary['structures'])
irrigation_ucrb = gpd.read_file(input_data_dictionary['irrigation'])
ditches_ucrb = gpd.read_file(input_data_dictionary['ditches'])
irrigation_ucrb = irrigation_ucrb.to_crs(epsg = 3857)
agg_diversions = pd.read_csv(input_data_dictionary['aggregated_diversions'])
aggregated_diversions = {}
for index, row in agg_diversions.iterrows():
  if row['statemod_diversion'] in aggregated_diversions:
    aggregated_diversions[row['statemod_diversion']].append(str(row['individual_diversion']))
  else:
    aggregated_diversions[row['statemod_diversion']] = [str(row['individual_diversion']), ]
ucrb.set_structure_types(aggregated_diversions, irrigation_ucrb, ditches_ucrb, structures_ucrb)

print('calculate initial water supply metrics')
snow_coefs_tot = {}
for res in ucrb.reservoir_list:
  snow_coefs_tot[res] = ucrb.make_snow_regressions(baseline_release_timeseries, ucrb.basin_snowpack['14010001'], res, 1950, 2013)

res_thres = {}
res_thres['3603543'] = 200.0
res_thres['3604512'] = 250.0
res_thres['3704516'] = 50.0
res_thres['5003668'] = 50.0
res_thres['5103709'] = 120.0
res_thres['5104055'] = 550.0
all_change1 = pd.DataFrame()
all_change2 = pd.DataFrame()
all_change3 = pd.DataFrame()
all_change4 = pd.DataFrame()
print('availability')
#crss.initializeDDM(demand_data, 'cm2015A.ddm')
for res in [reservoir_transfer_to,]:
  adaptive_toggle = 0
  for year_num in range(year_start_adaptive, year_end):
    year_add = 0
    month_start = 10
    ytd_diversions = 0.0
    for month_num in range(0, 12):
      if month_start + month_num == 13:
        month_start -= 12
        year_add = 1
      datetime_val = datetime(year_num + year_add, month_start + month_num, 1, 0, 0)
      if adaptive_toggle == 1:

        structure_deliveries = crss.update_structure_deliveries(delivery_data, datetime_val.year, datetime_val.month, read_from_file = False)
        structure_demands = crss.update_structure_demands(demand_data, datetime_val.year, datetime_val.month, read_from_file = False)
        ucrb.update_structure_demand_delivery(structure_deliveries, structure_demands, monthly_maximums, datetime_val)

        new_releases = crss.read_simulated_control_release_single(delivery_data, reservoir_storage_data_a, res, datetime_val.year, datetime_val.month)
        for release_type in new_releases:
          adaptive_release_timeseries.loc[datetime_val, release_type] = new_releases[release_type]
      total_water, this_month_diversions = ucrb.find_available_water(adaptive_release_timeseries, snow_coefs_tot[res], ytd_diversions, res, '14010001', datetime_val)
      
      ytd_diversions += this_month_diversions
      print(datetime_val, end = " ")
      print(total_water)
      if total_water < res_thres[res] and datetime_val.month >= 4 and datetime_val.month < 9:
        print(ucrb.structures_objects[tunnel_transfer_to].historical_monthly_deliveries.loc[datetime_val, 'deliveries'], end = " ")
        print(ucrb.structures_objects[tunnel_transfer_to].adaptive_monthly_deliveries.loc[datetime_val, 'deliveries'], end = " ")
        print(ucrb.structures_objects[tunnel_transfer_to].historical_monthly_demand.loc[datetime_val, 'demand'], end = " ")
        print(ucrb.structures_objects[tunnel_transfer_to].adaptive_monthly_demand.loc[datetime_val, 'demand'])
      
        current_control_location = adaptive_release_timeseries.loc[datetime_val, res + '_location']
        current_physical_supply = adaptive_release_timeseries.loc[datetime_val, res + '_physical_supply']
        change_points_purchase_1, change_points_buyout_1, last_right, last_structure = ucrb.find_adaptive_purchases(downstream_data, res, datetime_val, current_control_location, current_physical_supply)
        change_points_buyout_2 = ucrb.find_buyout_partners(last_right, last_structure, res, datetime_val)
        all_change1 = pd.concat([all_change1, change_points_purchase_1])
        all_change2 = pd.concat([all_change2, change_points_buyout_1])
        all_change3 = pd.concat([all_change3, change_points_buyout_2])
        change_points_buyout = pd.concat([change_points_buyout_1, change_points_buyout_2])

        crss.writenewDDM(demand_data, change_points_purchase_1, change_points_buyout, year_num + 1, month_num)
        os.system("StateMod_Model_15.exe cm2015A -simulate")        

        reservoir_storage_data_a = crss.read_text_file(input_data_dictionary['reservoir_storage_new'])
        
        initial_diversion_demand = ucrb.structures_objects[tunnel_transfer_to].adaptive_monthly_deliveries.loc[datetime_val, 'deliveries'] * 1.0
        change_points_purchase_2, change_points_buyout_1a = crss.compare_storage_scenarios(reservoir_storage_data_b, reservoir_storage_data_a, initial_diversion_demand, datetime_val.year, datetime_val.month, res, tunnel_transfer_to)
        change_points_buyout = pd.concat([change_points_buyout, change_points_buyout_1a])
        change_points_buyout_1 = pd.concat([change_points_buyout_1, change_points_buyout_1a])
        
        all_change4 = pd.concat([all_change4, change_points_purchase_2])
        change_points_purchase = pd.concat([change_points_purchase_1, change_points_purchase_2])
        
        crss.writenewDDM(demand_data, change_points_purchase, change_points_buyout, year_num + 1, month_num)
        os.system("StateMod_Model_15.exe cm2015A -simulate")        
        
        demand_data = crss.read_text_file(input_data_dictionary['structure_demand_new'])
        delivery_data = crss.read_text_file(input_data_dictionary['deliveries_new'])
        reservoir_storage_data_a = crss.read_text_file(input_data_dictionary['reservoir_storage_new'])
        
        structure_deliveries = crss.update_structure_deliveries(delivery_data, datetime_val.year, datetime_val.month, read_from_file = False)
        structure_demands = crss.update_structure_demands(demand_data, datetime_val.year, datetime_val.month, read_from_file = False)
        ucrb.update_structure_demand_delivery(structure_deliveries, structure_demands, monthly_maximums, datetime_val)
        print(np.sum(change_points_purchase_1['demand']), end = " ")
        print(np.sum(change_points_purchase_2['demand']), end = " ")
        print(np.sum(change_points_buyout_1['demand']), end = " ")
        print(np.sum(change_points_buyout_2['demand']), end = " ")
        print(current_control_location, end = " ")
        print(current_physical_supply, end = " ")
        print(ucrb.structures_objects[tunnel_transfer_to].historical_monthly_deliveries.loc[datetime_val, 'deliveries'], end = " ")
        print(ucrb.structures_objects[tunnel_transfer_to].adaptive_monthly_deliveries.loc[datetime_val, 'deliveries'], end = " ")
        print(ucrb.structures_objects[tunnel_transfer_to].historical_monthly_demand.loc[datetime_val, 'demand'], end = " ")
        print(ucrb.structures_objects[tunnel_transfer_to].adaptive_monthly_demand.loc[datetime_val, 'demand'])
        
        
        adaptive_toggle = 1

        all_change1.to_csv('output_files/purchases_' + res + '.csv')
        all_change2.to_csv('output_files/buyouts_' + res + '.csv')
        all_change3.to_csv('output_files/buyouts_2_' + res + '.csv')
        all_change4.to_csv('output_files/diversions_' + res + '.csv')