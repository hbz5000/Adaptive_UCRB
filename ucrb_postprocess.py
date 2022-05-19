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
import scipy.stats as stats

total_acres = 0.0
low_value_acres = 0.0
input_data_dictionary = crss.create_input_data_dictionary('B', 'A', folder_name = 'results_550/')
irrigation_ucrb = gpd.read_file(input_data_dictionary['irrigation'])
irrigation = irrigation_ucrb[irrigation_ucrb['SW_WDID1'] == '7200646']
for index, row in irrigation_ucrb.iterrows():
  total_acres += row['ACRES']
  if row['CROP_TYPE'] == 'GRASS_PASTURE' or row['CROP_TYPE'] == 'ALFALFA':
    low_value_acres += row['ACRES']
print(total_acres, end = " ")
print(low_value_acres)

input_data_dictionary = crss.create_input_data_dictionary('B', 'A', folder_name = 'results_550/')
irrigation_ucrb = gpd.read_file(input_data_dictionary['irrigation'])
irrigation = irrigation_ucrb[irrigation_ucrb['SW_WDID1'] == '7200646']
for index, row in irrigation.iterrows():
  print(row['CROP_TYPE'], end = " ")
  print(row['ACRES'])
input_data_dictionary = crss.create_input_data_dictionary('B', 'A', folder_name = 'results_550/')
input_data_dictionary2 = crss.create_input_data_dictionary('B', 'A', folder_name = 'results_600/')
input_data_dictionary3 = crss.create_input_data_dictionary('B', 'A', folder_name = 'results_650/')
input_data_dictionary4 = crss.create_input_data_dictionary('B', 'A', folder_name = 'results_700/')
structures_ucrb = gpd.read_file(input_data_dictionary['structures'])
print('create basin')
#Create Basin class with the same extent as StateMod Basin
#as defined by the input_data_dictionary files
project_folder = 'UCRB_analysis-master/'
ucrb = Basin(input_data_dictionary)
ucrb2 = Basin(input_data_dictionary2)
ucrb3 = Basin(input_data_dictionary3)
ucrb4 = Basin(input_data_dictionary4)
year_start = 1950
year_start_adaptive = 1950
year_end = 2013

structure_buyouts = pd.read_csv('output_files/buyouts_2_5104055.csv')
structure_buyouts['WDID'] = structure_buyouts['structure']
structure_buyouts['datetime'] = pd.to_datetime(structure_buyouts['date'])
buyout_years = np.zeros(len(structure_buyouts.index))
counter = 0
for index, row in structure_buyouts.iterrows():
  buyout_years[counter] = row['datetime'].year
  counter += 1
structure_buyouts['yearnum'] = buyout_years  

structure_purchases = pd.read_csv('output_files/purchases_5104055.csv')
structure_purchases['WDID'] = structure_purchases['structure']
structure_purchases['datetime'] = pd.to_datetime(structure_purchases['date'])
purchase_years = np.zeros(len(structure_purchases.index))
counter = 0
for index, row in structure_purchases.iterrows():
  purchase_years[counter] = row['datetime'].year
  counter += 1
structure_purchases['yearnum'] = purchase_years  

delivery_data = crss.read_text_file(input_data_dictionary['deliveries'])
delivery_data_new = crss.read_text_file(input_data_dictionary['deliveries_new'])
delivery_data_new2 = crss.read_text_file(input_data_dictionary2['deliveries_new'])
delivery_data_new3 = crss.read_text_file(input_data_dictionary3['deliveries_new'])
delivery_data_new4 = crss.read_text_file(input_data_dictionary4['deliveries_new'])

downstream_data = crss.read_text_file(input_data_dictionary['downstream'])
demand_data = crss.read_text_file(input_data_dictionary['structure_demand'])
reservoir_storage_data = crss.read_text_file(input_data_dictionary['reservoir_storage'])
reservoir_storage_data_new = crss.read_text_file(input_data_dictionary['reservoir_storage_new'])
reservoir_storage_data_new2 = crss.read_text_file(input_data_dictionary2['reservoir_storage_new'])
reservoir_storage_data_new3 = crss.read_text_file(input_data_dictionary3['reservoir_storage_new'])
reservoir_storage_data_new4 = crss.read_text_file(input_data_dictionary4['reservoir_storage_new'])
print('set demand/delivery')


structure_demands = crss.read_structure_demands(demand_data,year_start, year_end, read_from_file = False)
ucrb.set_structure_demands(structure_demands, use_rights = False)
ucrb2.set_structure_demands(structure_demands, use_rights = False)
ucrb3.set_structure_demands(structure_demands, use_rights = False)
ucrb4.set_structure_demands(structure_demands, use_rights = False)
structure_deliveries = crss.read_structure_deliveries(delivery_data, year_start, year_end, read_from_file = False)
structure_deliveries_new = crss.read_structure_deliveries(delivery_data_new, year_start, year_end, read_from_file = False)
structure_deliveries_new2 = crss.read_structure_deliveries(delivery_data_new2, year_start, year_end, read_from_file = False)
structure_deliveries_new3 = crss.read_structure_deliveries(delivery_data_new3, year_start, year_end, read_from_file = False)
structure_deliveries_new4 = crss.read_structure_deliveries(delivery_data_new4, year_start, year_end, read_from_file = False)
ucrb.set_structure_deliveries(structure_deliveries, structure_deliveries_adaptive = structure_deliveries_new, use_rights = False)
ucrb2.set_structure_deliveries(structure_deliveries, structure_deliveries_adaptive = structure_deliveries_new2, use_rights = False)
ucrb3.set_structure_deliveries(structure_deliveries, structure_deliveries_adaptive = structure_deliveries_new3, use_rights = False)
ucrb4.set_structure_deliveries(structure_deliveries, structure_deliveries_adaptive = structure_deliveries_new4, use_rights = False)

return_flow_data = crss.read_text_file(input_data_dictionary['return_flows'])
structure_return_flows = crss.read_structure_return_flows(return_flow_data, year_start, year_end, read_from_file = False)
ucrb.set_return_fractions(structure_return_flows)

baseline_release_timeseries = crss.read_simulated_control_release(delivery_data, reservoir_storage_data, year_start, year_end)
baseline_release_timeseries.index = pd.to_datetime(baseline_release_timeseries.index)
#calculate water supply metrics (inc. snowpack & storage) at each of the indentified reservoirs
snow_coefs_tot = ucrb.make_snow_regressions(baseline_release_timeseries, ucrb.basin_snowpack['14010001'], '5104055', year_start, year_end)
simulated_reservoir_timeseries = crss.read_simulated_reservoirs(reservoir_storage_data, '5104055', 1950, 2013)
data_figure = Plotter('available_water.png', figsize = (16,6))
total_water = data_figure.plot_available_water(simulated_reservoir_timeseries, ucrb.basin_snowpack['14010001'], baseline_release_timeseries, structure_deliveries, snow_coefs_tot, '5104055', '5104634', 1949, 2013, 5, show_plot = False)

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

print('set finance')
marginal_net_benefits, et_requirements = crss.create_et_benefits()
ucrb.set_structure_types(aggregated_diversions, irrigation_ucrb, ditches_ucrb, structures_ucrb, marginal_net_benefits, et_requirements)
ucrb2.set_structure_types(aggregated_diversions, irrigation_ucrb, ditches_ucrb, structures_ucrb, marginal_net_benefits, et_requirements)
ucrb3.set_structure_types(aggregated_diversions, irrigation_ucrb, ditches_ucrb, structures_ucrb, marginal_net_benefits, et_requirements)
ucrb4.set_structure_types(aggregated_diversions, irrigation_ucrb, ditches_ucrb, structures_ucrb, marginal_net_benefits, et_requirements)
reservoir_diversions = crss.compare_res_diversion_scenarios(reservoir_storage_data, reservoir_storage_data_new, 1950, 2013, '5104055')
reservoir_diversions2 = crss.compare_res_diversion_scenarios(reservoir_storage_data, reservoir_storage_data_new2, 1950, 2013, '5104055')
reservoir_diversions3 = crss.compare_res_diversion_scenarios(reservoir_storage_data, reservoir_storage_data_new3, 1950, 2013, '5104055')
reservoir_diversions4 = crss.compare_res_diversion_scenarios(reservoir_storage_data, reservoir_storage_data_new4, 1950, 2013, '5104055')
ucrb.find_annual_change_by_wyt(1950, 2013, ['5104634',], 0)
ucrb2.find_annual_change_by_wyt(1950, 2013, ['5104634',], 1)
ucrb3.find_annual_change_by_wyt(1950, 2013, ['5104634',], 2)
ucrb4.find_annual_change_by_wyt(1950, 2013, ['5104634',], 3)
data_figure = Plotter('frequency_impact.png', figsize = (16, 8))
data_figure.plot_transfer_tradeoffs()
print('start plotting')


total_annual_revenues = ucrb.find_station_revenues('70_ADC049', et_requirements, marginal_net_benefits, irrigation_ucrb, aggregated_diversions, year_start, year_end)
data_figure = Plotter('scatterplot_buyout_mitigation_70_ADC049.png', figsize = (16,8))
data_figure.plot_buyout_mitigation('70_ADC049', total_annual_revenues, total_water, year_start, 20.0)
data_figure = Plotter('cost_per_af.png', figsize = (16, 8))
color_map = 'rocket'
for it_no, folder_name, diversions_use in zip([0, 3, 2, 1], ['550', '700', '650', '600'], [reservoir_diversions, reservoir_diversions4, reservoir_diversions3, reservoir_diversions2]):
  structure_buyouts = pd.read_csv('results_' + folder_name + '/buyouts_2_5104055.csv')
  structure_purchases = pd.read_csv('results_' + folder_name + '/purchases_5104055.csv')
  cost_per_af, total_informal_exports = ucrb.find_informal_water_price('5104055', diversions_use, structure_purchases, structure_buyouts, 1.5, 20.0)
  data_figure.plot_cost_per_af(cost_per_af, total_informal_exports, it_no, color_map, 1000.0)

cost_multiple = [1.5, 2.5]
buyout_price = [20.0, 40.0] 
structure_buyouts = pd.read_csv('results_550/buyouts_2_5104055.csv')
structure_purchases = pd.read_csv('results_550/purchases_5104055.csv')
data_figure = Plotter('cost_per_af_sens.png', figsize = (16, 8))
color_map = 'RdBu'
counter = 0
for by_cnt, by in enumerate(buyout_price):
  for cm_cnt, cm in enumerate(cost_multiple):
    cost_per_af, total_informal_exports = ucrb.find_informal_water_price('5104055', reservoir_diversions, structure_purchases, structure_buyouts, cm, by)
    data_figure.plot_cost_per_af(cost_per_af, total_informal_exports, counter, color_map, 1500.0)
    counter += 1
    
axis_breaks = {}
axis_breaks['wet_1'] = 75.
axis_breaks['wet_2'] = 16.25
axis_breaks['normal_1'] = 30.
axis_breaks['normal_2'] = 6.25
axis_breaks['dry_1'] = 4.
axis_breaks['dry_2'] = 0.75


for plot_type in ['all',]:

  data_figure = Plotter('change_from_reoperation_all_' + plot_type + '_year.png', nr = 3, figsize = (16, 8))
  for it_no, folder_name, structure_use in zip([3, 2, 1, 0], ['700', '650', '600', '550'], [ucrb4.structures_objects, ucrb3.structures_objects, ucrb2.structures_objects, ucrb.structures_objects]):
    structure_buyouts = pd.read_csv('results_' + folder_name + '/buyouts_2_5104055.csv')
    structure_buyouts['WDID'] = structure_buyouts['structure']
    structure_buyouts['datetime'] = pd.to_datetime(structure_buyouts['date'])
    buyout_years = np.zeros(len(structure_buyouts.index))
    counter = 0
    for index, row in structure_buyouts.iterrows():
      buyout_years[counter] = row['datetime'].year
      counter += 1
    structure_buyouts['yearnum'] = buyout_years  

    structure_purchases = pd.read_csv('results_' + folder_name + '/purchases_5104055.csv')
    structure_purchases['WDID'] = structure_purchases['structure']
    structure_purchases['datetime'] = pd.to_datetime(structure_purchases['date'])
    purchase_years = np.zeros(len(structure_purchases.index))
    counter = 0
    for index, row in structure_purchases.iterrows():
      purchase_years[counter] = row['datetime'].year
      counter += 1
    structure_purchases['yearnum'] = purchase_years  

    buyouts_list = np.unique(list(structure_buyouts['structure']))
    purchase_list = np.unique(list(structure_purchases['structure']))
    data_figure.plot_structure_changes_by_wyt(structure_use, downstream_data, plot_type, axis_breaks, ['5104634',], ['Adams Tunnel:'], purchase_transfers = structure_purchases, buyout_transfers = structure_buyouts, purchase_list = purchase_list, buyouts_list = buyouts_list, show_partners = 'revenue', iteration_no = it_no)


print('past all')
abcs = efgh
for plot_type in ['1955', '1977', '2002']:
  this_year_buyouts = structure_buyouts[structure_buyouts['yearnum']==int(plot_type)]
  this_year_purchase = structure_purchases[structure_purchases['yearnum']==int(plot_type)]
  buyouts_list = np.unique(list(this_year_buyouts['structure']))
  purchase_list = np.unique(list(this_year_purchase['structure']))
  data_figure = Plotter('change_from_reoperation_partners_' + plot_type + '_year.png', nr = 2, figsize = (16, 8))
  data_figure.plot_structure_changes_by_wyt(ucrb.structures_objects,downstream_data, plot_type, axis_breaks, ['5104634',], ['Adams Tunnel:'], purchase_list = purchase_list, buyouts_list = buyouts_list, show_partners = 'partners')

for plot_type in ['1955', '1977', '2002']:
  this_year_buyouts = structure_buyouts[structure_buyouts['yearnum']==int(plot_type)]
  this_year_purchase = structure_purchases[structure_purchases['yearnum']==int(plot_type)]
  buyouts_list = np.unique(list(this_year_buyouts['structure']))
  purchase_list = np.unique(list(this_year_purchase['structure']))
  data_figure = Plotter('change_from_reoperation_partners_revenue_' + plot_type + '_year.png', nr = 2, figsize = (16, 8))
  data_figure.plot_structure_changes_by_wyt(ucrb.structures_objects,downstream_data, plot_type, axis_breaks, ['5104634',], ['Adams Tunnel:'], purchase_transfers = this_year_purchase, buyout_transfers = this_year_buyouts, purchase_list = purchase_list, buyouts_list = buyouts_list, show_partners = 'revenue')

for plot_type in ['1955', '1977', '2002']:
  this_year_buyouts = structure_buyouts[structure_buyouts['yearnum']==int(plot_type)]
  this_year_purchase = structure_purchases[structure_purchases['yearnum']==int(plot_type)]
  buyouts_list = np.unique(list(this_year_buyouts['structure']))
  purchase_list = np.unique(list(this_year_purchase['structure']))
  data_figure = Plotter('change_from_reoperation_thirdparty_' + plot_type + '_year.png', nr = 2, figsize = (16, 8))
  data_figure.plot_structure_changes_by_wyt(ucrb.structures_objects,downstream_data, plot_type, axis_breaks, ['5104634',], ['Adams Tunnel:'], purchase_list = purchase_list, buyouts_list = buyouts_list, show_partners = 'thirdparty')

for plot_type in ['1955', '1977', '2002']:
  this_year_buyouts = structure_buyouts[structure_buyouts['yearnum']==int(plot_type)]
  this_year_purchase = structure_purchases[structure_purchases['yearnum']==int(plot_type)]
  buyouts_list = np.unique(list(this_year_buyouts['structure']))
  purchase_list = np.unique(list(this_year_purchase['structure']))
  data_figure = Plotter('change_from_reoperation_thirdparty_' + plot_type + '_year.png', nr = 2, figsize = (16, 8))
  data_figure.plot_structure_changes_by_wyt(ucrb.structures_objects,downstream_data, plot_type, axis_breaks, ['5104634',], ['Adams Tunnel:'], purchase_list = purchase_list, buyouts_list = buyouts_list, show_partners = 'thirdparty')


for animation_pane in range(0,4):
  data_figure = Plotter('export_risks_' + str(animation_pane) + '.png', figsize = (16, 12))
  data_figure.plot_simple_risk(animation_pane)

abcd = efgh


##Initialize basin reservoirs
##there are other reservoirs in StateMod - these are just the 
##places where we want to look at water supply metrics

##load input file data
#load river network
print('load input data')
#load 'baseline' rights data
reservoir_rights_data = crss.read_text_file(input_data_dictionary['reservoir_rights'])
structure_rights_data = crss.read_text_file(input_data_dictionary['structure_rights'])
instream_rights_data = crss.read_text_file(input_data_dictionary['instream_rights'])
#load 'baseline' demand and deliveries data
demand_data = crss.read_text_file(input_data_dictionary['structure_demand'])
delivery_data = crss.read_text_file(input_data_dictionary['deliveries'])
demand_data_a = crss.read_text_file(input_data_dictionary['structure_demand_new'])
delivery_data_a = crss.read_text_file(input_data_dictionary['deliveries_new'])
downstream_data = crss.read_text_file(input_data_dictionary['downstream'])

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
structure_demands_a = crss.read_structure_demands(demand_data_a,year_start, year_end, read_from_file = False)
ucrb.set_structure_demands(structure_demands, structure_demands_adaptive = structure_demands_a)

print('apply deliveries to structures')
#read delivery data and apply to structures
structure_deliveries = crss.read_structure_deliveries(delivery_data, year_start, year_end, read_from_file = False)
structure_deliveries_a = crss.read_structure_deliveries(delivery_data_a, year_start, year_end, read_from_file = False)
ucrb.set_structure_deliveries(structure_deliveries, structure_deliveries_adaptive = structure_deliveries_a)

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
ucrb.set_structure_types(aggregated_diversions, irrigation_ucrb, ditches_ucrb, structures_ucrb)
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

#ax.set_ylim([0.0, 10000.0])
structure_buyouts = pd.read_csv('output_files/buyouts_2_5104055.csv')
structure_buyouts['WDID'] = structure_buyouts['structure']
structure_purchases = pd.read_csv('output_files/purchases_5104055.csv')
structure_purchases['WDID'] = structure_purchases['structure']
buyouts_list = list(structure_buyouts['structure'])
purchase_list = list(structure_purchases['structure'])
marginal_net_benefits, et_requirements = crss.create_et_benefits()
#percent_filled = ucrb.find_percent_delivered(marginal_net_benefits, et_requirements, aggregated_diversions, irrigation_ucrb, ditches_ucrb, structure_buyouts, structure_purchases, downstream_data)
#data_figure = Plotter('percent_filled.png', nr = 1, nc = 4)
#data_figure.plot_structure_changes(percent_filled,downstream_data, buyouts_list, purchase_list)


structures_ucrb = gpd.read_file(input_data_dictionary['structures'])
irrigation_ucrb = gpd.read_file(input_data_dictionary['irrigation'])
ditches_ucrb = gpd.read_file(input_data_dictionary['ditches'])
irrigation_ucrb = irrigation_ucrb.to_crs(epsg = 3857)
crop_list = np.unique(irrigation_ucrb['CROP_TYPE'])
overall_crop_areas = {}

for index, row in irrigation_ucrb.iterrows():
  if row['CROP_TYPE'] in overall_crop_areas:
    overall_crop_areas[row['CROP_TYPE']] += row['ACRES']
  else:
    overall_crop_areas[row['CROP_TYPE']] = row['ACRES']
  

print(crop_list)
marginal_net_benefits = {}
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
et_requirements = {}
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


grapes_planting_costs = [-6385.0, -2599.0, -1869.0, 754.0, 2012.0, 2133.0, 2261.0] 
grapes_baseline_revenue = [2261.0, 2261.0, 2261.0, 2261.0, 2261.0, 2261.0, 2261.0]
total_npv_costs = 0.0
counter = 0
for cost, baseline in zip(grapes_planting_costs, grapes_baseline_revenue):
  total_npv_costs +=  (baseline - cost)/np.power(1.025, counter)
  counter += 1
marginal_net_benefits['GRAPES'] = total_npv_costs
orchard_planting_costs = [-5183.0, -2802.0, -2802.0, 395.0, 5496.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0] 
orchard_baseline_revenue = [9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, -5183.0, -2802.0, -2802.0, 395.0, 5496.0]
total_npv_costs = 0.0
counter = 0
for cost, baseline in zip(orchard_planting_costs, orchard_baseline_revenue):
  total_npv_costs +=  (baseline - cost)/np.power(1.025, counter)
  counter += 1
marginal_net_benefits['ORCHARD_WITH_COVER'] = total_npv_costs
marginal_net_benefits['ORCHARD_WO_COVER'] = total_npv_costs

water_costs = np.zeros(len(marginal_net_benefits))
crop_list = []
counter = 0
for x in marginal_net_benefits:
  water_costs[counter] = marginal_net_benefits[x] / (et_requirements[x] / 12.0)
  crop_list.append(x)
  counter += 1
sorted_index = np.argsort(water_costs*(-1.0))
crop_list_new = np.asarray(crop_list)
sorted_crops = crop_list_new[sorted_index]

fig, ax = plt.subplots(figsize = (16,6))
current_water_use = 0.0
counter = 0
running_area = 0.0
for x in sorted_crops:
  total_cost = marginal_net_benefits[x] / (et_requirements[x] / 12.0)
  if total_cost > 300.0:
    total_cost -= 8600.0
  total_area = overall_crop_areas[x]
  print(x, end = " ")
  print(total_area, end = " ")
  print(total_cost)
  ax.fill_between([running_area, running_area + total_area], np.zeros(2), [total_cost, total_cost], facecolor = 'indianred', edgecolor = 'black', linewidth = 2.0)
  counter += 1.5
  running_area += total_area
ax.set_yticks([0.0, 100.0, 200.0, 300.0, 9000.0, 9100.0, 9200.0])
ax.set_yticklabels(['$0', '$100', '$200', '$300', '$9000', '$9100', '$9200'])
counter = 0.5
label_list = ['Grapes', 'Orchard', 'Vegetables', 'Bluegrass', 'Sorghum', 'Alfalfa', 'Corn', 'Pasture', 'Wheat', 'Beans', 'Grain', 'Barley']
ax.set_ylabel('Cost of Fallowing per AF', fontsize = 18, weight = 'bold', fontname = 'Gill Sans MT')
ax.set_xlabel('Basinwide Acreage', fontsize = 18, weight = 'bold', fontname = 'Gill Sans MT')
ax.set_ylim([0.0, 9200.0])
for item in (ax.get_xticklabels()):
  item.set_fontsize(14)
  item.set_fontname('Gill Sans MT')
for axesnum in range(0, 3):
  for item in (ax.get_yticklabels()):
    item.set_fontsize(14)
    item.set_fontname('Gill Sans MT')

plt.savefig('Shapefiles_UCRB/crop_types.png', dpi = 300, bbox_inches = 'tight', pad_inches = 0.0)
plt.show()

structures_ucrb = structures_ucrb.to_crs(epsg = 3857)
ditches_ucrb = ditches_ucrb.to_crs(epsg = 3857)

irrigation_structures = list(irrigation_ucrb['SW_WDID1'].astype(str))
ditch_structures = list(ditches_ucrb['wdid'].astype(str))
all_structures = list(structures_ucrb['WDID'].astype(str))

structure_data = {}
structure_buyouts['date'] = pd.to_datetime(structure_buyouts['date'])
for index, row in structure_buyouts.iterrows():
  
  if row['WDID'] in irrigation_structures:
    if row['WDID'] in structure_data:
      if row['date'].month > 9:
        structure_data[row['WDID']]['buyouts'][row['date'].year + 1 - 1950] += float(row['demand'])
      else:
        structure_data[row['WDID']]['buyouts'][row['date'].year - 1950] += float(row['demand'])
    else:
      structure_data[row['WDID']] = {}
      this_irrigated_area = irrigation_ucrb[irrigation_ucrb['SW_WDID1'] == row['WDID']]
      structure_data[row['WDID']]['shape'] = [this_irrigated_area,]
      structure_data[row['WDID']]['type'] = ['irrigation',]
      structure_data[row['WDID']]['acres'] = np.sum(this_irrigated_area['ACRES'])
      structure_data[row['WDID']]['acres_tot'] = {}
      for crop_ind in crop_list:
        structure_data[row['WDID']]['acres_tot'][crop_ind] = 0.0
      for index_ia, row_ia in this_irrigated_area.iterrows():
        structure_data[row['WDID']]['acres_tot'][row_ia['CROP_TYPE']] += row_ia['ACRES']
                
      structure_data[row['WDID']]['buyouts'] = np.zeros(64)
      if row['date'].month > 9:
        structure_data[row['WDID']]['buyouts'][row['date'].year + 1 - 1950] += float(row['demand'])
      else:
        structure_data[row['WDID']]['buyouts'][row['date'].year - 1950] += float(row['demand'])
    
  elif row['WDID'] in aggregated_diversions:
    if row['WDID'] in structure_data:
      if row['date'].month > 9:
        structure_data[row['WDID']]['buyouts'][row['date'].year + 1 - 1950] += float(row['demand'])
      else:
        structure_data[row['WDID']]['buyouts'][row['date'].year - 1950] += float(row['demand'])
    else:
      structure_data[row['WDID']] = {}
      structure_data[row['WDID']]['shape'] = []
      structure_data[row['WDID']]['type'] = []
      structure_data[row['WDID']]['acres'] = 0.0
      structure_data[row['WDID']]['buyouts'] = np.zeros(64)
      structure_data[row['WDID']]['acres_tot'] = {}
      for crop_ind in crop_list:
        structure_data[row['WDID']]['acres_tot'][crop_ind] = 0.0
      if row['date'].month > 9:
        structure_data[row['WDID']]['buyouts'][row['date'].year + 1 - 1950] += float(row['demand'])
      else:
        structure_data[row['WDID']]['buyouts'][row['date'].year - 1950] += float(row['demand'])
      
      for ind_structure in aggregated_diversions[row['WDID']]:
        if ind_structure in irrigation_structures:
          this_irrigated_area = irrigation_ucrb[irrigation_ucrb['SW_WDID1'] == ind_structure]
          structure_data[row['WDID']]['shape'].append(this_irrigated_area)
          structure_data[row['WDID']]['type'].append('irrigation')
          structure_data[row['WDID']]['acres'] += np.sum(this_irrigated_area['ACRES'])
          for index_ia, row_ia in this_irrigated_area.iterrows():
            structure_data[row['WDID']]['acres_tot'][row_ia['CROP_TYPE']] += row_ia['ACRES']

        elif ind_structure in all_structures:
          pass
        else:
          print('no location for individual aggregated', end = " ")
          print(ind_structure)
  elif row['WDID'] in ditch_structures:
    pass
  elif row['WDID'] in all_structures:
    if row['WDID'] not in structure_data:
      structure_data[row['WDID']] = {}
      structure_data[row['WDID']]['shape'] = [structures_ucrb[structures_ucrb['WDID'] == row['WDID']],]
      structure_data[row['WDID']]['type'] = ['structure',]
      structure_data[row['WDID']]['acres'] = 0.0
      structure_data[row['WDID']]['acres_tot'] = {}
      for crop_ind in crop_list:
        structure_data[row['WDID']]['acres_tot'][crop_ind] = 0.0
      structure_data[row['WDID']]['buyouts'] = np.zeros(64)
      print('non irrigation purchase', end = " ")
      print(row['WDID'])

  else:
    structure_data[row['WDID']] = {}
    print('no structure', end = " ")
    print(row['WDID'])
    structure_data[row['WDID']]['type'] = ['structure',]
    structure_data[row['WDID']]['acres'] = 0.0
    structure_data[row['WDID']]['shape'] = []
    structure_data[row['WDID']]['acres_tot'] = {}
    for crop_ind in crop_list:
      structure_data[row['WDID']]['acres_tot'][crop_ind] = 0.0
    structure_data[row['WDID']]['buyouts'] = np.zeros(64)
    
structure_data2 = {}
structure_purchases['date'] = pd.to_datetime(structure_purchases['date'])
for index, row in structure_purchases.iterrows():
  
  if row['WDID'] in irrigation_structures:
    if row['WDID'] in structure_data2:
      if row['date'].month > 9:
        structure_data2[row['WDID']]['buyouts'][row['date'].year + 1 - 1950] += float(row['demand'])
      else:
        structure_data2[row['WDID']]['buyouts'][row['date'].year - 1950] += float(row['demand'])
    else:
      structure_data2[row['WDID']] = {}
      this_irrigated_area = irrigation_ucrb[irrigation_ucrb['SW_WDID1'] == row['WDID']]
      structure_data2[row['WDID']]['shape'] = [this_irrigated_area,]
      structure_data2[row['WDID']]['type'] = ['irrigation',]
      structure_data2[row['WDID']]['acres'] = np.sum(this_irrigated_area['ACRES'])
      structure_data2[row['WDID']]['acres_tot'] = {}
      for crop_ind in crop_list:
        structure_data2[row['WDID']]['acres_tot'][crop_ind] = 0.0
      for index_ia, row_ia in this_irrigated_area.iterrows():
        structure_data2[row['WDID']]['acres_tot'][row_ia['CROP_TYPE']] += row_ia['ACRES']
                
      structure_data2[row['WDID']]['buyouts'] = np.zeros(64)
      if row['date'].month > 9:
        structure_data2[row['WDID']]['buyouts'][row['date'].year + 1 - 1950] += float(row['demand'])
      else:
        structure_data2[row['WDID']]['buyouts'][row['date'].year - 1950] += float(row['demand'])
    
  elif row['WDID'] in aggregated_diversions:
    if row['WDID'] in structure_data2:
      if row['date'].month > 9:
        structure_data2[row['WDID']]['buyouts'][row['date'].year + 1 - 1950] += float(row['demand'])
      else:
        structure_data2[row['WDID']]['buyouts'][row['date'].year - 1950] += float(row['demand'])
    else:
      structure_data2[row['WDID']] = {}
      structure_data2[row['WDID']]['shape'] = []
      structure_data2[row['WDID']]['type'] = []
      structure_data2[row['WDID']]['acres'] = 0.0
      structure_data2[row['WDID']]['buyouts'] = np.zeros(64)
      structure_data2[row['WDID']]['acres_tot'] = {}
      for crop_ind in crop_list:
        structure_data2[row['WDID']]['acres_tot'][crop_ind] = 0.0
      if row['date'].month > 9:
        structure_data2[row['WDID']]['buyouts'][row['date'].year + 1 - 1950] += float(row['demand'])
      else:
        structure_data2[row['WDID']]['buyouts'][row['date'].year - 1950] += float(row['demand'])
      
      for ind_structure in aggregated_diversions[row['WDID']]:
        if ind_structure in irrigation_structures:
          this_irrigated_area = irrigation_ucrb[irrigation_ucrb['SW_WDID1'] == ind_structure]
          structure_data2[row['WDID']]['shape'].append(this_irrigated_area)
          structure_data2[row['WDID']]['type'].append('irrigation')
          structure_data2[row['WDID']]['acres'] += np.sum(this_irrigated_area['ACRES'])
          for index_ia, row_ia in this_irrigated_area.iterrows():
            structure_data2[row['WDID']]['acres_tot'][row_ia['CROP_TYPE']] += row_ia['ACRES']

        elif ind_structure in all_structures:
          pass
        else:
          print('no location for individual aggregated', end = " ")
          print(ind_structure)
  elif row['WDID'] in ditch_structures:
    pass
  elif row['WDID'] in all_structures:
    if row['WDID'] not in structure_data2:
      structure_data2[row['WDID']] = {}
      structure_data2[row['WDID']]['shape'] = [structures_ucrb[structures_ucrb['WDID'] == row['WDID']],]
      structure_data2[row['WDID']]['type'] = ['structure',]
      structure_data2[row['WDID']]['acres'] = 0.0
      structure_data2[row['WDID']]['acres_tot'] = {}
      for crop_ind in crop_list:
        structure_data2[row['WDID']]['acres_tot'][crop_ind] = 0.0
      structure_data2[row['WDID']]['buyouts'] = np.zeros(64)
      print('non irrigation purchase', end = " ")
      print(row['WDID'])

  else:
    structure_data2[row['WDID']] = {}
    print('no structure', end = " ")
    print(row['WDID'])
    structure_data2[row['WDID']]['type'] = ['structure',]
    structure_data2[row['WDID']]['acres'] = 0.0
    structure_data2[row['WDID']]['shape'] = []
    structure_data2[row['WDID']]['acres_tot'] = {}
    for crop_ind in crop_list:
      structure_data2[row['WDID']]['acres_tot'][crop_ind] = 0.0
    structure_data2[row['WDID']]['buyouts'] = np.zeros(64)


buyouts_timeseries = pd.read_csv('output_files/buyouts_5104055.csv')
purchase_timeseries = pd.read_csv('output_files/purchases_5104055.csv')
diversions_timeseries = pd.read_csv('output_files/diversions_5104055.csv')
diversions_timeseries['date'] = pd.to_datetime(diversions_timeseries['date'])
buyouts_timeseries['date'] = pd.to_datetime(buyouts_timeseries['date'])
purchase_timeseries['date'] = pd.to_datetime(purchase_timeseries['date'])
year_num = 1950
month_num = 1
current_datetime = datetime(year_num, month_num, 1, 0, 0)
fig, ax = plt.subplots()
diversion_costs = []
diversion_volumes = []
purchase_price_increase = 1.25
buyout_price = 20.0
while current_datetime < datetime(2014, 1, 1, 0, 0):
  if month_num == 1:
    total_buyouts = 0.0
    purchase_cost = 0.0
    total_diversions = 0.0
  this_diversions = diversions_timeseries[diversions_timeseries['date'] == current_datetime]
  if len(this_diversions) > 0:
    total_diversions += np.sum(this_diversions['demand']) * (-1.0)
  this_buyouts = buyouts_timeseries[buyouts_timeseries['date'] == current_datetime]
  if len(this_buyouts) > 0:
    total_buyouts += np.sum(this_buyouts['demand'])
  buyout_cost = total_buyouts * 20.0
  this_purchases = purchase_timeseries[purchase_timeseries['date'] == current_datetime]
  total_current_purchases = np.sum(this_purchases['demand'])
  if len(this_purchases) > 0:
    for index, row in this_purchases.iterrows():
      min_cost = 99999.9
      if structure_data2[str(row['structure'])]['type'] == 'structure':
        purchase_cost += row['demand'] * 1000.0
      else:
        for x in structure_data2[str(row['structure'])]['acres_tot']:
          if structure_data2[str(row['structure'])]['acres_tot'][x] > 0.0:
            min_cost = min(min_cost, marginal_net_benefits[x] / (et_requirements[x] / 12.0))
        if min_cost == 99999.9:
          min_cost = 200.0
        purchase_cost += row['demand'] * max(min_cost * purchase_price_increase, 100)
        print(row['demand'], end = " ")
        print(min_cost)
  month_num += 1
  if month_num == 13:
    #ax.fill_between([datetime(year_num, 1, 1, 0, 0), datetime(year_num + 1, 1, 1, 0, 0)], np.zeros(2), [total_buyouts, total_buyouts], facecolor = 'goldenrod', edgecolor = 'black', linewidth = 0.0)
    #ax.fill_between([datetime(year_num, 1, 1, 0, 0), datetime(year_num + 1, 1, 1, 0, 0)], np.zeros(2), [total_purchases, total_purchases], facecolor = 'indianred', edgecolor = 'black', linewidth = 0.0)
    #ax.fill_between([datetime(year_num, 1, 1, 0, 0), datetime(year_num + 1, 1, 1, 0, 0)], np.zeros(2), [total_diversions, total_diversions], facecolor = 'steelblue', edgecolor = 'black', linewidth = 0.0)
    if total_diversions > 0.0:
      diversion_costs.append((purchase_cost + buyout_cost)/total_diversions)
      diversion_volumes.append(total_diversions)
    year_num += 1
    month_num = 1    
  current_datetime = datetime(year_num, month_num, 1, 0, 0)
pos1 = np.linspace(0.0, 1500.0, 15)
diversion_costs = np.asarray(diversion_costs)
diversion_volumes = np.asarray(diversion_volumes)
print(diversion_costs)
print(diversion_volumes)
last_bound = 0.0
labels_list = ['0', ]
for x in range(1, len(pos1)):
  if x == len(pos1)-1:
    total_volume = np.sum(diversion_volumes[diversion_costs > last_bound])
    labels_list.append('> ' + str(int(x)))

  else:
    total_volume = np.sum(diversion_volumes[np.logical_and(diversion_costs > last_bound, diversion_costs < pos1[x])])
    labels_list.append(str(int(x)))
  ax.fill_between([last_bound, pos1[x]], np.zeros(2), [total_volume, total_volume], facecolor = 'indianred', edgecolor = 'black')
  last_bound = pos1[x] * 1.0

  
ax.set_xticks(pos1)
ax.set_xticklabels(labels_list)
ax.set_ylabel('Annual ', fontsize = 18, weight = 'bold', fontname = 'Gill Sans MT')
ax.set_ylim([0.0, np.log(10000) + 0.5])
for item in (ax.get_xticklabels()):
  item.set_fontsize(14)
  item.set_fontname('Gill Sans MT')
for axesnum in range(0, 3):
  for item in (ax.get_yticklabels()):
    item.set_fontsize(14)
    item.set_fontname('Gill Sans MT')

plt.savefig('Shapefiles_UCRB/crop_types.png', dpi = 300, bbox_inches = 'tight', pad_inches = 0.0)
plt.show()

  
nhd_database_filename = 'Shapefiles_UCRB/NHDPLUS_H_1401_HU4_GDB.gdb'
extended_table = gpd.read_file(nhd_database_filename, layer = 'WBDHU4')
ucrb_basin = extended_table[extended_table['HUC4'] == '1401']
extended_table8 = gpd.read_file(nhd_database_filename, layer = 'WBDHU8')
ucrb_huc8 = gpd.sjoin(extended_table8, ucrb_basin, how = 'inner', op = 'within')
streams_ucrb = gpd.read_file('Shapefiles_UCRB/UCRBstreams.shp')
ucrb_huc8 = ucrb_huc8.to_crs(epsg = 3857)
streams_ucrb = streams_ucrb.to_crs(epsg = 3857)

column1 = []
for x in range(0, len(ucrb_huc8.index)):
  column1.append('UCRB')
ucrb_huc8 = gpd.GeoDataFrame(pd.DataFrame(column1, index = ucrb_huc8.index, columns = ['column1',]), crs = ucrb_huc8.crs, geometry = ucrb_huc8.geometry)
ucrb_huc8 = ucrb_huc8.dissolve(by = 'column1')

county_filename = 'Shapefiles_UCRB/cb_2018_us_state_500k/cb_2018_us_state_500k.shp'
state_all = gpd.read_file(county_filename)
state_all_T = state_all.to_crs(epsg = 3857)
state_use = state_all_T[state_all_T['STUSPS'] == 'CO']

streams_ucrb = streams_ucrb.to_crs(epsg = 3857)
structures_ucrb = structures_ucrb.to_crs(epsg = 3857)
crop_list = list(set(irrigation_ucrb['CROP_TYPE']))
plot_type = 'fig1'
data_figure = Mapper()
projection_string = 'EPSG:3857'#project raster to data projection
background_map_filename = 'Shapefiles_UCRB/06-B5-mos/colorado_mosaic'
data_figure.plot_scalar_raster(projection_string, background_map_filename, 'bone')

qalycolors = sns.color_palette('Reds', 100)
streams_ucrb = gpd.sjoin(streams_ucrb, ucrb_huc8, how = 'inner', op = 'intersects')

data_figure.plot_scale(ucrb_huc8, 'depth', type = 'polygons', solid_color = 'none', solid_alpha = 0.5, linewidth_size = 2.0, outline_color = 'black')
plot_use = 'type2'
data_figure.plot_scale(irrigation_ucrb, 'depth', type = 'polygons', solid_color = 'forestgreen', solid_alpha = 1.0, linewidth_size = 0.0, outline_color = 'forestgreen')
data_figure.plot_scale(streams_ucrb, 'depth', type = 'polygons', solid_color = 'xkcd:cobalt', solid_alpha = 1.0, linewidth_size = 1.0, outline_color = 'xkcd:cobalt')

for x in structure_data:
  for ag_parcel, parcel_type in zip(structure_data[x]['shape'], structure_data[x]['type']):
    if parcel_type == 'structure':
      if ag_parcel.loc[ag_parcel.index[0], 'WDID'] == '3600881' or ag_parcel.loc[ag_parcel.index[0], 'WDID'] == '3604684':
        data_figure.plot_scale(ag_parcel, 'depth', type = 'polygons', solid_color = 'black', solid_alpha = 1.0, linewidth_size = 0.0, outline_color = 'forestgreen')
    if parcel_type == 'irrigation':
      data_figure.plot_scale(ag_parcel, 'depth', type = 'polygons', solid_color = 'crimson', solid_alpha = 1.0, linewidth_size = 0.0, outline_color = 'crimson')
      if plot_use == 'type4':
        buyout_per_acre = np.max(structure_data[x]['buyouts']) / structure_data[x]['acres']
        per_acre_df = pd.DataFrame([buyout_per_acre,], columns = ['bpa',])
        for ag_shape in range(0, len(ag_parcel.index)):
          per_acre_gdf = gpd.GeoDataFrame(per_acre_df, crs = ag_parcel.crs, geometry = [ag_parcel.loc[ag_parcel.index[ag_shape], 'geometry'],])
          print(buyout_per_acre)
          data_figure.plot_scale(per_acre_gdf, 'bpa', type = 'polygons', solid_color = 'scaled', colorscale = 'autumn_r', value_lim = (0.0, 2.0), linewidth_size = 0.0)

if plot_use == 'type2':
  for x in structure_data2:
    for ag_parcel, parcel_type in zip(structure_data2[x]['shape'], structure_data2[x]['type']):
      if parcel_type == 'structure':
        if ag_parcel.loc[ag_parcel.index[0], 'WDID'] == '3600881' or ag_parcel.loc[ag_parcel.index[0], 'WDID'] == '3604684':
          data_figure.plot_scale(ag_parcel, 'depth', type = 'polygons', solid_color = 'black', solid_alpha = 1.0, linewidth_size = 0.0, outline_color = 'forestgreen')
      if parcel_type == 'irrigation':
        data_figure.plot_scale(ag_parcel, 'depth', type = 'polygons', solid_color = 'xkcd:dandelion', solid_alpha = 1.0, linewidth_size = 0.0, outline_color = 'xkcd:dandelion')


#data_figure.add_colorbar_offmap('Largest Annual Purchase (AF/ac)', ['0', '1', '2'])
#plot colorbar w/perimeter
xl = ucrb_huc8.total_bounds[0]
xr = ucrb_huc8.total_bounds[2]
by = ucrb_huc8.total_bounds[1]
uy = ucrb_huc8.total_bounds[3]
xrange = ucrb_huc8.total_bounds[2] - ucrb_huc8.total_bounds[0]
yrange = ucrb_huc8.total_bounds[3] - ucrb_huc8.total_bounds[1]
data_figure.add_inset_figure(state_all_T, (xl, xr), (by, uy), (xl - 2.0*xrange, xr + 2.0*xrange), (by - 3.0*yrange, uy + 3.0*yrange), 3857, inset_location_number = 4) 
data_figure.format_plot(xlim = (xl - xrange*0.0025, xr + xrange*0.0025), ylim = (by - yrange*0.0025, uy + yrange*0.0025))
legend_location = 'upper left'
if plot_use == 'type1':
  legend_element = [Patch(facecolor='none', edgecolor='black', label='Upper Colorado River Basin'),
                  Patch(facecolor='crimson', edgecolor='black', label='Buyout Partners'),
                  Patch(facecolor='forestgreen', edgecolor='black', label='Other Agriculture'),
                  Line2D([0], [0], markerfacecolor='black', markeredgecolor='black', lw = 0, marker = 'o', markersize = 10, label='Other buyouts')]
elif plot_use == 'type2':
  legend_element = [Patch(facecolor='none', edgecolor='black', label='Upper Colorado River Basin'),
                  Patch(facecolor='crimson', edgecolor='black', label='Buyout Partners'),
                  Patch(facecolor='xkcd:dandelion', edgecolor='black', label='Purchase Partners'),
                  Patch(facecolor='forestgreen', edgecolor='black', label='Other Agriculture'),
                  Line2D([0], [0], markerfacecolor='black', markeredgecolor='black', lw = 0, marker = 'o', markersize = 10, label='Other buyouts')]

legend_properties = {'family':'Gill Sans MT','weight':'bold','size':12}
data_figure.add_legend(legend_location, legend_element, legend_properties)
plt.savefig('Shapefiles_UCRB/purchase_partners_' + plot_use + '.png', dpi = 300, bbox_inches = 'tight', pad_inches = 0.0)


buyout_structures = structures_ucrb.merge(structure_buyouts, on = 'WDID', how = 'inner')
structures_ucrb['WDID'] = structures_ucrb['WDID'].astype(str)
ditches_ucrb['WDID'] = ditches_ucrb['wdid'].astype(str)
irrigation_ucrb['WDID'] = irrigation_ucrb['SW_WDID1'].astype(str)
structure_buyouts['WDID'] = structure_buyouts['WDID'].astype(str)
structure_data = {}
for index, row in irrigation_ucrb.iterrows():
  if row['WDID'] in list(ucrb.structures_objects):
    if row['WDID'] in structure_data:
      structure_data[row['WDID']].append(row)
    else:
      structure_data[row['WDID']] = [row,]
  else:
    if row['WDID'] + '_D' in list(ucrb.structures_objects):
      if row['WDID'] in structure_data:
        structure_data[row['WDID'] + '_D'].append(row)
      else:
        structure_data[row['WDID'] + '_D'] = [row,]
        

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

##load input file data
#load river network
print('load input data')
downstream_data = crss.read_text_file(input_data_dictionary['downstream'])
#load historical reservoir data
historical_reservoir_data = crss.read_text_file(input_data_dictionary['historical_reservoirs'])
#load 'baseline' reservoir data
reservoir_storage_data_b = crss.read_text_file(input_data_dictionary['reservoir_storage'])
reservoir_storage_data_a = crss.read_text_file(input_data_dictionary['reservoir_storage_new'])
#load 'baseline' rights data
reservoir_rights_data = crss.read_text_file(input_data_dictionary['reservoir_rights'])
structure_rights_data = crss.read_text_file(input_data_dictionary['structure_rights'])
#load 'baseline' demand and deliveries data
demand_data_b = crss.read_text_file(input_data_dictionary['structure_demand'])
delivery_data_b = crss.read_text_file(input_data_dictionary['deliveries'])
demand_data_a = crss.read_text_file(input_data_dictionary['structure_demand_new'])
delivery_data_a = crss.read_text_file(input_data_dictionary['deliveries_new'])

for res in ['5104055',]:
  #historical
  ucrb.structures_objects[res].simulated_reservoir_timeseries = crss.read_simulated_reservoirs(reservoir_storage_data_b, res, year_start, year_end)
  ucrb.structures_objects[res].adaptive_reservoir_timeseries = crss.read_simulated_reservoirs(reservoir_storage_data_a, res, year_start, year_end)

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
structure_demands = crss.read_structure_demands(demand_data_b,year_start, year_end, read_from_file = False)
structure_demands_a = crss.read_structure_demands(demand_data_a,year_start, year_end, read_from_file = False)
ucrb.set_structure_demands(structure_demands, structure_demands_adaptive = structure_demands_a)

print('apply deliveries to structures')
#read delivery data and apply to structures
structure_deliveries = crss.read_structure_deliveries(delivery_data_b, year_start, year_end, read_from_file = False)
structure_deliveries_a = crss.read_structure_deliveries(delivery_data_a, year_start, year_end, read_from_file = False)
ucrb.set_structure_deliveries(structure_deliveries, structure_deliveries_adaptive = structure_deliveries_a)
fig, ax = plt.subplots()
for res in ['5104055',]:
  ax.plot(ucrb.structures_objects[res].simulated_reservoir_timeseries[res], color = 'blue')
  ax.plot(ucrb.structures_objects[res].adaptive_reservoir_timeseries[res], color = 'red')
plt.show()
plt.close()
cumulative_losses = np.zeros(len(ucrb.structures_objects['5104634'].adaptive_monthly_deliveries['deliveries']))
counter = 0
for baseline, adaptive in zip(ucrb.structures_objects['5104634'].historical_monthly_deliveries['deliveries'], ucrb.structures_objects['5104634'].adaptive_monthly_deliveries['deliveries']):
  if counter > 0:
    cumulative_losses[counter] = cumulative_losses[counter-1] + adaptive - baseline
  else:
    cumulative_losses[counter] = adaptive - baseline
  counter += 1
  
fig, ax = plt.subplots()
ax.plot(cumulative_losses, color = 'blue')
plt.show()

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
      print(year_num + year_add, end = " ")
      print(month_start + month_num, end = " ")
      total_water, this_month_diversions = ucrb.find_available_water(adaptive_release_timeseries, snow_coefs_tot[res], ytd_diversions, res, '14010001', datetime_val)
      ytd_diversions += this_month_diversions
      #print(datetime_val, end = " ")
      #print(total_water)
      if total_water < res_thres[res]:
      
        current_control_location = adaptive_release_timeseries.loc[datetime_val, res + '_location']
        current_physical_supply = adaptive_release_timeseries.loc[datetime_val, res + '_physical_supply']
        change_points1, last_right, last_structure = ucrb.find_adaptive_purchases(downstream_data, res, datetime_val, current_control_location, current_physical_supply)
        change_points2 = ucrb.find_buyout_partners(last_right, last_structure, res, datetime_val)
        
        change_points = pd.concat([change_points1, change_points2])
        crss.writenewDDM(demand_data, change_points, datetime_val.year, datetime_val.month)
        #os.system("StateMod_Model_15.exe cm2015A -simulate")        

        reservoir_storage_data_a = crss.read_text_file(input_data_dictionary['reservoir_storage_new'])
        
        change_points3 = crss.compare_storage_scenarios(reservoir_storage_data_b, reservoir_storage_data_a, datetime_val.year, datetime_val.month, res, '5104634')
        print(change_points3)        
        change_points = pd.concat([change_points, change_points3])
        crss.writenewDDM(demand_data, change_points, datetime_val.year, datetime_val.month)
        #os.system("StateMod_Model_15.exe cm2015A -simulate")        
        
        demand_data = crss.read_text_file(input_data_dictionary['structure_demand_new'])
        delivery_data = crss.read_text_file(input_data_dictionary['deliveries_new'])
        reservoir_storage_data_a = crss.read_text_file(input_data_dictionary['reservoir_storage_new'])
        
        structure_deliveries = crss.update_structure_deliveries(delivery_data, datetime_val.year, datetime_val.month, read_from_file = False)
        structure_demands = crss.update_structure_demands(demand_data, datetime_val.year, datetime_val.month, read_from_file = False)
        ucrb.update_structure_demand_delivery(structure_deliveries, structure_demands, datetime_val)
        
        adaptive_toggle = 1



northern_districts = gpd.read_file('Shapefiles_UCRB/Subdistrict_Boundary/Subdistrict_Boundary.shp')
northern_area= gpd.read_file('Shapefiles_UCRB/Northern_Water_Boundary/Northern_Water_Boundary.shp')
base_file = gpd.GeoDataFrame()
counter = 0
for index, row in northern_districts.iterrows():
  if counter == 0:
    clipped_tract = northern_districts[northern_districts.index == index]
  else:
    this_tract = northern_districts[northern_districts.index == index]
    clipped_tract = gpd.overlay(this_tract, base_file, how = 'difference')
  base_file = pd.concat([base_file, clipped_tract])
  base_file = base_file.drop_duplicates()

for index, row in northern_area.iterrows():
  this_tract = northern_area[northern_area.index == index]
  clipped_tract = gpd.overlay(this_tract, base_file, how = 'difference')
  base_file = pd.concat([base_file, clipped_tract])
  base_file = base_file.drop_duplicates()
  
base_file = base_file.geometry.unary_union
new_file = gpd.GeoDataFrame(pd.DataFrame(['Northern Water',], columns = ['Utility',]), crs = northern_districts.crs, geometry = [base_file,])
print(new_file)
new_file.to_file(driver = 'ESRI Shapefile', filename = 'Shapefiles_UCRB/Service_Areas/Northern_Water_SA.shp')
fig, ax = plt.subplots()
new_file.plot(ax = ax)
plt.show()

unincor = gpd.read_file('Shapefiles_UCRB\census_tracts_2010\census_tracts_2010.shp')
canals = gpd.read_file('Shapefiles_UCRB\muni_2021\muni_2021.shp')
city_list = ['Bow Mar', 'Cherry Hills Village', 'Columbine Valley', 'Denver', 'Edgewater', 'Greenwood Village', 'Littleton', 'Lakeside','Mountain View', 'Sheridan', 'Wheat Ridge']
base_file = gpd.GeoDataFrame()
for index, row in canals.iterrows():
  if row['city'] in city_list:
    this_city = canals[canals.index == index]
    base_file = pd.concat([base_file, this_city])
lakewood_city = canals[canals['city'] == 'Lakewood']
centennial_city = canals[canals['city'] == 'Centennial']
lone_tree_city = canals[canals['city'] == 'Lone Tree']

ken_caryl_tracts = ['120.22', '120.23', '120.24', '120.46', '120.47' ,'120.57', '120.59', '120.60']
columbine_tracts = ['56.21', '120.48', '120.49', '120.51', '120.52', '120.53', '120.55']
dakota_ridge_tracts = ['159', '120.38', '120.39', '120.41', '120.42', '120.43', '120.44', '120.45']
#commerce_city_tacts = ['89.01','95.53', '95.01','95.02','95.53']
commerce_city_tacts = ['90.01', '90.02', '95.01', '95.02', '95.53', '96.06', '96.07', '97.51', '97.52', '150']
lakewood_tracts = ['158', '110','111', '158', '9800', '159', '108.01', '105.04', '109.02', '112.02', '114.01', '114.02','115.50', '116.01', '116.02', '117.01', '117.02', '117.08', '117.09', '117.10', '117.11', '117.12', '117.23', '117.26', '117.27', '117.28', '117.29', '117.30', '117.33', '117.32', '118.03', '118.04', '118.05', '118.06', '119.04', '109.01', '109.02', '120.50', '120.54']
centennial_tracts = ['56.12', '56.11', '56.14', '56.36', '56.35', '56.25', '56.26', '56.27', '56.28', '56.32', '56.31', '56.30', '56.29', '67.06', '67.07', '67.08', '67.09', '67.11', '67.13']
lone_tree_tracts = ['141.13', '141.14', '141.15', '141.16', '141.40']
acres_green_tracts = ['141.14',]
for index, row in unincor.iterrows():
  if str(row['name10']) in ken_caryl_tracts:
    this_tract = unincor[unincor.index == index]
    clipped_tract = gpd.overlay(this_tract, base_file, how = 'difference')
  elif str(row['name10']) in columbine_tracts:
    this_tract = unincor[unincor.index == index]
    clipped_tract = gpd.overlay(this_tract, base_file, how = 'difference')
  elif str(row['name10']) in dakota_ridge_tracts:
    this_tract = unincor[unincor.index == index]
    clipped_tract = gpd.overlay(this_tract, base_file, how = 'difference')
  elif str(row['name10']) in commerce_city_tacts:
    this_tract = unincor[unincor.index == index]
    clipped_tract = gpd.overlay(this_tract, base_file, how = 'difference')          
  elif str(row['name10']) in lakewood_tracts:
    this_tract = unincor[unincor.index == index]
    clipped_tract = gpd.overlay(this_tract, lakewood_city, how = 'intersection')
    clipped_tract = gpd.overlay(clipped_tract, base_file, how = 'difference')
  elif str(row['name10']) in centennial_tracts:
    this_tract = unincor[unincor.index == index]
    clipped_tract = gpd.overlay(this_tract, centennial_city, how = 'intersection')
    clipped_tract = gpd.overlay(clipped_tract, base_file, how = 'difference')
  elif str(row['name10']) in lone_tree_tracts:
    this_tract = unincor[unincor.index == index]
    clipped_tract = gpd.overlay(this_tract, lone_tree_city, how = 'intersection')
    clipped_tract = gpd.overlay(clipped_tract, base_file, how = 'difference')
  elif str(row['name10']) in acres_green_tracts:
    this_tract = unincor[unincor.index == index]
    clipped_tract = gpd.overlay(this_tract, lone_tree_city, how = 'difference')
    clipped_tract = gpd.overlay(clipped_tract, base_file, how = 'difference')
  base_file = pd.concat([base_file, clipped_tract])
  base_file = base_file.drop_duplicates()
  
base_file = base_file.geometry.unary_union
new_file = gpd.GeoDataFrame(pd.DataFrame(['Denver Water'], columns = ['Utility',]), crs = canals.crs, geometry = [base_file,])
new_file.to_file(driver = 'ESRI Shapefile', filename = 'Shapefiles_UCRB\Service_Areas\Denver_Water_SA.shp')

#fort collins, 18,855 units
fig,ax = plt.subplots()
new_file.plot(ax = ax)
plt.show()
#1 = Arvada
#2 = Golden
#3 = Edgewater (use)
#4 = Morrison
#5 = Bow Mar (use)
#6 = Mountain View (use)
#7 = County area
#8 = Wheat Ridge (use)
#9 = Lakewood (use)
#10 = Mountain View (use)
#11 = Westminster
#12 = nothing
#13 = Superior
#14 = unnamed (use)
all_codes = np.unique(canals['CITYCODE'])
print(all_codes)
fig, ax = plt.subplots()
c = [Point(-105.110831, 39.577661), Point(-105.069975, 39.590229), Point(-105.131087,39.617871)]
unused_cities = gpd.GeoDataFrame(pd.DataFrame(['Ken Caryl', 'Columbine', 'Dakota Ridge']), crs = canals.crs, geometry = c)

for city_code in range(0, 20):
  if city_code == 7:
    this_parcel = canals[canals['CITYCODE'] == str(city_code).zfill(2)]
    print(this_parcel)
    for index, row in this_parcel.iterrows():
      smaller_parcel = this_parcel[this_parcel.index == index]
            
      find_areas = gpd.sjoin(smaller_parcel, unused_cities, how = 'inner', op = 'contains')
      if len(find_areas) > 0:
        print(index)
        find_areas.plot(ax = ax)
plt.show()
plt.close()
use_list = [3, 5, 6, 8, 9, 10, 14]
fig, ax = plt.subplots()
for city_code in range(0, len(use_list)):
  this_parcel = canals[canals['CITYCODE'] == str(use_list[city_code]).zfill(2)]
  this_parcel.plot(ax = ax)
plt.show()
plt.close()
canals.plot(ax = ax)
plt.show()
for index, row in canals.iterrows():
  if row['COUNTY'] == 'Denver':
    this_parcel = canals[canals.index == index]
    
    this_parcel.plot(ax = ax)
plt.show()

