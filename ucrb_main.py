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
input_data_dictionary = {}
###geographic layout
input_data_dictionary['hydrography'] = 'Shapefiles_UCRB/NHDPLUS_H_1401_HU4_GDB.gdb'
input_data_dictionary['structures'] = 'Shapefiles_UCRB/Div_5_structures.shp'
##basin labels
input_data_dictionary['HUC4'] = ['1401',]
input_data_dictionary['HUC8'] = ['14010001', '14010002', '14010003', '14010004', '14010005']
##locations of large agricultural aggregations
input_data_dictionary['combined_structures'] = {}
input_data_dictionary['combined_structures']['14010001'] = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', 
                                                            '011', '012', '013', '014', '015', '016', '020', '021', '022', '023', 
                                                            '024', '025', '026', '027', '028', '032']
input_data_dictionary['combined_structures']['14010002'] = ['017', '018', '019']
input_data_dictionary['combined_structures']['14010003'] = ['029', '030', '031']
input_data_dictionary['combined_structures']['14010004'] = ['033', '034', '035', '036', '037', '038', '039', '040']
input_data_dictionary['combined_structures']['14010005'] = ['041', '042', '043', '044', '045', '046', '047', '048', '049', '050', 
                                                            '051', '052', '053', '054', '055', '056', '057', '058', '059', '060', 
                                                            '061', '062', '063', '064', '065']
###snow data
input_data_dictionary['snow'] = 'Snow_Data/'

###statemod input data
##monthly demand data
input_data_dictionary['structure_demand'] = 'input_files/cm2015B.ddm'
##water rights data
input_data_dictionary['structure_rights'] = 'input_files/cm2015B.ddr'
##reservoir fill rights data
input_data_dictionary['reservoir_rights'] = 'input_files/cm2015B.rer'
##full natural flow data
input_data_dictionary['natural flows'] = 'cm2015x.xbm'
##flow/node network
input_data_dictionary['downstream'] = 'input_files/cm2015.rin'
##historical reservoir data
input_data_dictionary['historical_reservoirs'] = 'input_files/cm2015.eom'
##call data
input_data_dictionary['calls'] = 'output_files/cm2015B.xca'

###statemod output data
##reservoir storage data
input_data_dictionary['reservoir_storage'] = 'output_files/cm2015B.xre'
##diversion data
input_data_dictionary['deliveries'] = 'output_files/cm2015B.xdd'

##adaptive reservoir output data
input_data_dictionary['reservoir_storage_new'] = 'cm2015A.xre'
##adaptive diversion data
input_data_dictionary['deliveries_new'] = 'cm2015A.xdd'
##adaptive demand data
input_data_dictionary['structure_demand_new'] = 'cm2015A.ddm'

print('create basin')
#Create Basin class with the same extent as StateMod Basin
#as defined by the input_data_dictionary files
project_folder = 'UCRB_analysis-master/'
ucrb = Basin(input_data_dictionary)
year_start = 1908
year_start_adaptive = 1950
year_end = 2013

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
for res in ['5104055',]:
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
baseline_release_timeseries = crss.read_simulated_control_release(delivery_data, reservoir_storage_data_b, ucrb.reservoir_list, year_start, year_end)
adaptive_release_timeseries = baseline_release_timeseries.copy(deep = True)
adaptive_release_timeseries.index = pd.to_datetime(adaptive_release_timeseries.index)
#calculate water supply metrics (inc. snowpack & storage) at each of the indentified reservoirs

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
res_thres['5104055'] = 600.0
print('availability')
crss.initializeDDM(demand_data, 'cm2015A.ddm')
for res in ['5104055',]:
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
        ucrb.update_structure_demand_delivery(structure_deliveries, structure_demands, datetime_val)

        new_releases = crss.read_simulated_control_release_single(delivery_data, reservoir_storage_data_a, res, datetime_val.year, datetime_val.month)
        for release_type in new_releases:
          adaptive_release_timeseries.loc[datetime_val, release_type] = new_releases[release_type]
      total_water, this_month_diversions = ucrb.find_available_water(adaptive_release_timeseries, snow_coefs_tot[res], ytd_diversions, res, '14010001', datetime_val)
      
      ytd_diversions += this_month_diversions
      print(datetime_val, end = " ")
      print(total_water)
      if total_water < res_thres[res]:
      
        current_control_location = adaptive_release_timeseries.loc[datetime_val, res + '_location']
        current_physical_supply = adaptive_release_timeseries.loc[datetime_val, res + '_physical_supply']
        change_points1, last_right, last_structure = ucrb.find_adaptive_purchases(downstream_data, res, datetime_val, current_control_location, current_physical_supply)
        change_points2 = ucrb.find_buyout_partners(last_right, last_structure, res, datetime_val)
        
        change_points = pd.concat([change_points1, change_points2])
        crss.writenewDDM(demand_data, change_points,  year_num + 1, month_num)
        os.system("StateMod_Model_15.exe cm2015A -simulate")        

        reservoir_storage_data_a = crss.read_text_file(input_data_dictionary['reservoir_storage_new'])
        
        change_points3 = crss.compare_storage_scenarios(reservoir_storage_data_b, reservoir_storage_data_a, datetime_val.year, datetime_val.month, res, '5104634')
        change_points = pd.concat([change_points, change_points3])
        print(change_points)        
        crss.writenewDDM(demand_data, change_points, year_num + 1, month_num)
        os.system("StateMod_Model_15.exe cm2015A -simulate")        
        
        demand_data = crss.read_text_file(input_data_dictionary['structure_demand_new'])
        delivery_data = crss.read_text_file(input_data_dictionary['deliveries_new'])
        reservoir_storage_data_a = crss.read_text_file(input_data_dictionary['reservoir_storage_new'])
        
        structure_deliveries = crss.update_structure_deliveries(delivery_data, datetime_val.year, datetime_val.month, read_from_file = False)
        structure_demands = crss.update_structure_demands(demand_data, datetime_val.year, datetime_val.month, read_from_file = False)
        ucrb.update_structure_demand_delivery(structure_deliveries, structure_demands, datetime_val)
        print(ucrb.structures_objects['5104634'].historical_monthly_deliveries.loc[datetime_val, 'deliveries'], end = " ")
        print(ucrb.structures_objects['5104634'].adaptive_monthly_deliveries.loc[datetime_val, 'deliveries'], end = " ")
        print(ucrb.structures_objects['5104634'].historical_monthly_demand.loc[datetime_val, 'demand'], end = " ")
        print(ucrb.structures_objects['5104634'].adaptive_monthly_demand.loc[datetime_val, 'demand'])
        
        
        adaptive_toggle = 1

