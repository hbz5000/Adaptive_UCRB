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
input_data_dictionary = crss.create_input_data_dictionary('B', 'D')

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

ucrb.tunnel_list = {}
ucrb.tunnel_list['Adams'] = '5104634'
ucrb.tunnel_list['Roberts'] ='3604684'
ucrb.tunnel_list['Boustead'] = '3804625SU'
ucrb.tunnel_list['Moffat'] = '5104655'
ucrb.tunnel_list['Twin Lakes'] = '3804617'
ucrb.tunnel_list['Homestake'] = '3704614'
ucrb.tunnel_list['Grand River'] = '5104601'
ucrb.tunnel_list['Hoosier Pass'] = '3604683SU'
tunnel_id_list = []
for x in ucrb.tunnel_list:
  tunnel_id_list.append(ucrb.tunnel_list[x])

ucrb.tunnel_reservoirs = {}
ucrb.tunnel_reservoirs['Adams'] = '5104055'
ucrb.tunnel_reservoirs['Roberts'] ='3604512'
ucrb.tunnel_reservoirs['Boustead'] = 'none'
ucrb.tunnel_reservoirs['Moffat'] = '5103709'
ucrb.tunnel_reservoirs['Twin Lakes'] = 'none'
ucrb.tunnel_reservoirs['Homestake'] = '3704516'
ucrb.tunnel_reservoirs['Grand River'] = 'none'
ucrb.tunnel_reservoirs['Hoosier Pass'] = 'none'
tunnel_reservoir_list = []
tunnel_use_list = []
name_use_list = []
for x in ucrb.tunnel_reservoirs:
  if ucrb.tunnel_reservoirs[x] != 'none':
    tunnel_reservoir_list.append(ucrb.tunnel_reservoirs[x])
    tunnel_use_list.append(ucrb.tunnel_list[x])
    name_use_list.append(x)

delivery_data = crss.read_text_file(input_data_dictionary['deliveries'])
delivery_data_new = crss.read_text_file(input_data_dictionary['deliveries_new'])

downstream_data = crss.read_text_file(input_data_dictionary['downstream'])
demand_data = crss.read_text_file(input_data_dictionary['structure_demand'])
reservoir_storage_data = crss.read_text_file(input_data_dictionary['reservoir_storage'])
baseline_release_timeseries = crss.read_simulated_control_release(delivery_data, reservoir_storage_data, year_start, year_end, use_list = tunnel_reservoir_list)
structure_demands = crss.read_structure_demands(demand_data,year_start, year_end, read_from_file = False)
ucrb.set_structure_demands(structure_demands, use_rights = False)
structure_deliveries = crss.read_structure_deliveries(delivery_data, year_start, year_end, read_from_file = False)
structure_deliveries_new = crss.read_structure_deliveries(delivery_data_new, year_start, year_end, read_from_file = False)
ucrb.set_structure_deliveries(structure_deliveries, structure_deliveries_adaptive = structure_deliveries_new, use_rights = False)

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
marginal_net_benefits, et_requirements = crss.create_et_benefits()
ucrb.set_structure_types(aggregated_diversions, irrigation_ucrb, ditches_ucrb, structures_ucrb, marginal_net_benefits, et_requirements)
ucrb.find_annual_change_by_wyt(1950, 2013, tunnel_use_list)
axis_breaks = {}
axis_breaks['wet_1'] = 75.
axis_breaks['wet_2'] = 16.25
axis_breaks['normal_1'] = 30.
axis_breaks['normal_2'] = 6.25
axis_breaks['dry_1'] = 4.
axis_breaks['dry_2'] = 0.75

for plot_type in ['wet', 'normal', 'dry']:
  data_figure = Plotter('change_from_reoperation_' + plot_type + '_year.png', nr = 2, figsize = (16, 8))
  data_figure.plot_structure_changes_by_wyt(ucrb.structures_objects,downstream_data, plot_type, axis_breaks, tunnel_use_list, name_use_list)


for tunnel_name in ucrb.tunnel_list:
  if ucrb.tunnel_reservoirs[tunnel_name] != 'none':
    struct_id = ucrb.tunnel_list[tunnel_name]
    data_figure = Plotter('historical_exports_' + struct_id + '.png', nr = 2, figsize = (16, 12))
    data_figure.plot_historical_exports(structure_deliveries,structure_deliveries_new, ucrb.basin_snowpack['14010001'], struct_id, tunnel_name, 1950, 2013)


print('calculate initial water supply metrics')

abcd = efgh
snow_coefs_tot = {}
monthly_max = {}
annual_max = {}
all_change1 = pd.DataFrame()
for tunnel_name in ucrb.tunnel_reservoirs:
  if ucrb.tunnel_reservoirs[tunnel_name] != 'none':
    res_id = ucrb.tunnel_reservoirs[tunnel_name]
    tunnel_id = ucrb.tunnel_list[tunnel_name]
    snow_coefs_tot[res_id] = ucrb.make_snow_regressions(baseline_release_timeseries, ucrb.basin_snowpack['14010001'], res_id, 1950, 2013)
    ucrb.structures_objects[res_id].adaptive_reservoir_timeseries = crss.read_simulated_reservoirs(reservoir_storage_data, res_id, year_start, year_end)
  tunnel_id = ucrb.tunnel_list[tunnel_name]
  monthly_max[tunnel_name], annual_max[tunnel_name] = crss.find_historical_max_deliveries(structure_deliveries, tunnel_id)

adaptive_toggle = 0
ytd_diversions = {}
for year_num in range(year_start_adaptive, year_end):
  year_add = 0
  month_start = 10
  for tunnel_name in ucrb.tunnel_reservoirs:
    if ucrb.tunnel_reservoirs[tunnel_name] != 'none':
      ytd_diversions[tunnel_name] = 0.0
    
  crss.make_control_file('cm2015', 'D', year_start_adaptive - 10, min(year_num + 2, year_end))
  for month_num in range(0, 12):

    if month_start + month_num == 13:
      month_start -= 12
      year_add = 1

    datetime_val = datetime(year_num + year_add, month_start + month_num, 1, 0, 0)
    structure_deliveries = crss.update_structure_deliveries(delivery_data, datetime_val.year, datetime_val.month, read_from_file = False)

    change_points_buyout = pd.DataFrame(columns = ['demand', 'structure'])
    change_points_purchase = pd.DataFrame(columns = ['demand', 'structure'])
    
    if adaptive_toggle == 1:
      structure_storage = crss.update_simulated_reservoirs(reservoir_storage_data_d, datetime_val.year, datetime_val.month, use_list = tunnel_reservoir_list, year_read = 'all')
      print(structure_storage)
      ucrb.update_structure_storage(structure_storage, datetime_val)
      new_releases = crss.read_simulated_control_release_single(delivery_data, reservoir_storage_data_d, datetime_val.year, datetime_val.month, use_list = tunnel_reservoir_list)
      ucrb.update_structure_outflows(new_releases, datetime_val)
    
    new_structures = []
    new_demands = []
    old_demands = []
    for tunnel_name in ucrb.tunnel_reservoirs:
      if ucrb.tunnel_reservoirs[tunnel_name] != 'none':
        tunnel_id = ucrb.tunnel_list[tunnel_name]
        res_id = ucrb.tunnel_reservoirs[tunnel_name]
        original_diversion = structure_deliveries[tunnel_id]
        total_water, this_month_diversions = ucrb.find_available_water(snow_coefs_tot[res_id], ytd_diversions[tunnel_name] + original_diversion/1000.0, res_id, '14010001', datetime_val)

        this_month_diversion = min(min(total_water * 1000.0 / 2.0, annual_max[tunnel_name]) / np.sum(monthly_max[tunnel_name]), 1.0) * monthly_max[tunnel_name][datetime_val.month - 1]
        print(tunnel_id, end = " ")
        print(total_water, end = " ")
        print(original_diversion, end = " ")
        print(this_month_diversion, end = " ")
        print(annual_max[tunnel_name], end = " ")
        print(np.sum(monthly_max[tunnel_name]), end = " ")
        print(monthly_max[tunnel_name][datetime_val.month - 1])
        new_structures.append(tunnel_id)
        new_demands.append(original_diversion - this_month_diversion)
        old_demands.append(original_diversion)
    change_points_purchase['demand'] = new_demands
    change_points_purchase['structure'] = new_structures
    change_points_buyout['demand'] = old_demands
    change_points_buyout['structure'] = new_structures
    
    #print(change_points_purchase)
    #if adaptive_toggle == 1:
    #  crss.writepartialDDM(demand_data, demand_data_new, change_points_purchase, change_points_buyout, month_num, year_num + 1, year_start_adaptive, year_num + 1, scenario_name = 'D', structure_list = new_structures)
    #else:
    #  crss.writenewDDM(demand_data, change_points_purchase, change_points_buyout, year_num + 1, month_num, scenario_name = 'D')
    #os.system("StateMod_Model_15.exe cm2015D -simulate")        

    demand_data_new = crss.read_text_file(input_data_dictionary['structure_demand_new'])
    delivery_data = crss.read_text_file(input_data_dictionary['deliveries_new'])
    reservoir_storage_data_d = crss.read_text_file(input_data_dictionary['reservoir_storage_new'])
        
    structure_deliveries = crss.update_structure_deliveries(delivery_data, datetime_val.year, datetime_val.month, read_from_file = False)
    for tunnel_name in ucrb.tunnel_list:
      if ucrb.tunnel_reservoirs[tunnel_name] != 'none':
        tunnel_id = ucrb.tunnel_list[tunnel_name]
        new_diversion = structure_deliveries[tunnel_id]
        ytd_diversions[tunnel_name] += new_diversion/1000.0
        print(tunnel_id, end = " ")
        print(new_diversion)
    new_releases = crss.read_simulated_control_release_single(delivery_data, reservoir_storage_data_d, datetime_val.year, datetime_val.month, use_list = tunnel_reservoir_list)
    ucrb.update_structure_outflows(new_releases, datetime_val)

    all_change1 = pd.concat([all_change1, change_points_purchase])
    all_change1.to_csv('output_files/reoperation.csv')
             
    adaptive_toggle = 1


for struct_id in tunnel_id_list:
  data_figure = Plotter('historical_exports_' + struct_id + '.png', nr = 2)
  data_figure.plot_historical_exports(structure_deliveries,struct_id)


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
delivery_data_old = crss.read_text_file(input_data_dictionary['deliveries'])
for res in [reservoir_transfer_to,]:
  adaptive_toggle = 0
  remaining_storage = 0.0
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
        ucrb.update_structure_demand_delivery(structure_demands, structure_deliveries, monthly_maximums, datetime_val)

        new_releases = crss.read_simulated_control_release_single(delivery_data, reservoir_storage_data_a, datetime_val.year, datetime_val.month)
        ucrb.update_structure_outflows(new_releases, datetime_val)
      total_water, this_month_diversions = ucrb.find_available_water(snow_coefs_tot[res], ytd_diversions, res, '14010001', datetime_val)



print(structure_deliveries)

#os.system("StateMod_Model_15.exe cm2015B -simulate")        
plan_data = crss.read_text_file(input_data_dictionary['plan_releases'])
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
instream_rights_data = crss.read_text_file(input_data_dictionary['instream_rights'])
#load 'baseline' demand and deliveries data
demand_data = crss.read_text_file(input_data_dictionary['structure_demand'])
delivery_data = crss.read_text_file(input_data_dictionary['deliveries'])
return_flow_data = crss.read_text_file(input_data_dictionary['return_flows'])

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
instream_rights_name, instream_rights_structure_name, instream_rights_priority, instream_rights_decree = crss.read_rights_data(instream_rights_data)
#using rights data from inputs, the 'basin' class creates structure & reservoir objects
#in each structure/reservoir object, there are one or more 'rights' objects
ucrb.set_rights_to_reservoirs(reservoir_rights_name, reservoir_rights_structure_name, reservoir_rights_priority, reservoir_rights_decree, reservoir_fill_rights)
ucrb.set_rights_to_structures(structure_rights_name, structure_rights_structure_name, structure_rights_priority, structure_rights_decree)
ucrb.set_rights_to_instream(instream_rights_name, instream_rights_structure_name, instream_rights_priority, instream_rights_decree)
#create 'rights stack' - all rights listed in the order of their priority, w/ structure names, decree amounts, etc.
ucrb.combine_rights_data(structure_rights_name, structure_rights_structure_name, structure_rights_priority, structure_rights_decree, reservoir_rights_name, reservoir_rights_structure_name, reservoir_rights_priority, reservoir_rights_decree, instream_rights_name, instream_rights_structure_name, instream_rights_priority, instream_rights_decree)

print('apply demands to structures')
#read demand data and apply to structures
structure_demands = crss.read_structure_demands(demand_data,year_start, year_end, read_from_file = False)
ucrb.set_structure_demands(structure_demands)

print('apply deliveries to structures')
#read delivery data and apply to structures
structure_deliveries = crss.read_structure_deliveries(delivery_data, year_start, year_end, read_from_file = False)
ucrb.set_structure_deliveries(structure_deliveries)

structure_return_flows = crss.read_structure_return_flows(return_flow_data, year_start, year_end, read_from_file = False)
ucrb.set_return_fractions(structure_return_flows)

for structure_name in ucrb.structures_objects:
  if np.sum(ucrb.structures_objects[structure_name].historical_monthly_deliveries['deliveries']) > 0.0 and len(ucrb.structures_objects[structure_name].rights_list) == 0:
    print('no rights', end = " ")
    print(np.sum(ucrb.structures_objects[structure_name].historical_monthly_demand['demand']), end = " ")
    print(structure_name)
for structure_name in ucrb.structures_objects:
  if np.sum(ucrb.structures_objects[structure_name].historical_monthly_deliveries['deliveries']) > 0.0 and np.sum(ucrb.structures_objects[structure_name].historical_monthly_demand['demand']) == 0.0:
    print('no demand', end = " ")
    print(np.sum(ucrb.structures_objects[structure_name].historical_monthly_deliveries['deliveries']), end = " ")
    print(structure_name)

monthly_maximums = ucrb.adjust_structure_deliveries()
baseline_release_timeseries = crss.read_simulated_control_release(delivery_data, reservoir_storage_data_b, year_start, year_end)
ucrb.set_structure_outflows(baseline_release_timeseries)
