from osgeo import gdal, osr
from osgeo.gdalconst import *
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
from skimage import exposure
import crss_reader as crss

                   
####Create Filename Dictionary####
input_data_dictionary = crss.create_input_data_dictionary('B', 'A')
epsg_use = 3857

area_ucrb = gpd.read_file('Shapefiles_UCRB/DIV3CO.shp')
area_ucrb = area_ucrb[area_ucrb['DIV'] == 5]
area_ucrb = area_ucrb.to_crs(epsg = epsg_use)

districts_ucrb = gpd.read_file('Shapefiles_UCRB/Water_Districts.shp')
districts_ucrb = districts_ucrb.to_crs(epsg = epsg_use)
ditches_ucrb = gpd.read_file('Shapefiles_UCRB/Div5_Irrigated_Lands_2015/Div5_2015_Ditches.shp')
ditches_ucrb = ditches_ucrb.to_crs(epsg = epsg_use)
irrigation_ucrb = gpd.read_file('Shapefiles_UCRB/Div5_Irrigated_Lands_2015/Div5_Irrig_2015.shp')
irrigation_ucrb = irrigation_ucrb.to_crs(epsg = epsg_use)

irrigation_northern = gpd.read_file('Shapefiles_UCRB/Div1_Irrigated_Lands_2020/Div1_Irrigated_Lands_2020/Div1_Irrig_2020.shp')
irrigation_northern = irrigation_northern.to_crs(epsg = epsg_use)
streams_ucrb = gpd.read_file('Shapefiles_UCRB/UCRBstreams.shp')
streams_ucrb = streams_ucrb.to_crs(epsg = epsg_use)
structures_ucrb = gpd.read_file('Shapefiles_UCRB/Div_5_structures.shp')
structures_ucrb = structures_ucrb.to_crs(epsg = epsg_use)
flowlines_ucrb = gpd.read_file('Shapefiles_UCRB/flowline.shp')
flowlines_ucrb = flowlines_ucrb.to_crs(epsg = epsg_use)

denver_water = gpd.read_file('Shapefiles_UCRB/Service_Areas/Denver_Water_SA.shp')
denver_water = denver_water.to_crs(epsg = epsg_use)
northern_districts = gpd.read_file('Shapefiles_UCRB/Service_Areas/Northern_Water_SA.shp')
northern_districts = northern_districts.to_crs(epsg = epsg_use)
#northern_area= gpd.read_file('Adaptive_experiment/StateMod/Shapefiles_UCRB/Northern_Water_Boundary/Northern_Water_Boundary.shp')
#northern_area = northern_area.to_crs(epsg = epsg_use)

riverplatte = gpd.read_file('Shapefiles_UCRB/river/river.shp')
riverplatte = riverplatte.to_crs(epsg = epsg_use)
riverplatte = gpd.overlay(riverplatte, northern_districts, how = 'intersection')
streamplatte = gpd.read_file('Shapefiles_UCRB/streams/streams.shp')
streamplatte = streamplatte.to_crs(epsg = epsg_use)
streamplatte = gpd.overlay(streamplatte, northern_districts, how = 'intersection')

county_filename = 'Shapefiles_UCRB/cb_2018_us_state_500k/cb_2018_us_state_500k.shp'
state_all = gpd.read_file(county_filename)
state_all_T = state_all.to_crs(epsg = epsg_use)

nhd_database_filename = 'Shapefiles_UCRB/NHDPLUS_H_1401_HU4_GDB.gdb'
extended_table = gpd.read_file(nhd_database_filename, layer = 'WBDHU4')
ucrb = extended_table[extended_table['HUC4'] == '1401']
extended_table8 = gpd.read_file(nhd_database_filename, layer = 'WBDHU8')
ucrb_huc8 = gpd.sjoin(extended_table8, ucrb, how = 'inner', op = 'within')
ucrb_huc8 = ucrb_huc8.to_crs(epsg = epsg_use)


def plot_informal_purchases(res):
  purchased_water = pd.read_csv('output_files/purchases_' + res + '.csv')
  buyout_water = pd.read_csv('output_files/buyouts_2_' + res + '.csv')
  purchased_water['date'] = pd.to_datetime(purchased_water['date'])
  buyout_water['date'] = pd.to_datetime(buyout_water['date'])
  structure_purchase_list = {}
  structure_buyout_list = {}
  prev_demand = 0.0
  prev_date = datetime(1950, 1, 1, 0, 0)
  start_year = 1950
  date_index = []

  for year in range(1950, 2014):
    for month in range(1, 13):
      date_index.append(datetime(year, month, 1, 0, 0))
  informal_transfer = pd.DataFrame(index = date_index, columns = ['purchased', 'buyouts', 'cumulative_purchase', 'cumulative_buyout'])
  informal_transfer['purchased'] = np.zeros(len(date_index))
  informal_transfer['buyouts'] = np.zeros(len(date_index))
  informal_transfer['cumulative_purchase'] = np.zeros(len(date_index))
  informal_transfer['cumulative_buyout'] = np.zeros(len(date_index))

  total_date_purchase = 0.0
  for index, row in purchased_water.iterrows():
    date_val = row['date']
    if date_val > prev_date:
      informal_transfer.loc[prev_date, 'purchased'] = total_date_purchase
      prev_demand = 0.0
      total_date_purchase = 0.0
      prev_date =  row['date']

    if row['demand'] > prev_demand:
      if row['structure'] in structure_purchase_list:
        structure_purchase_list[row['structure']].append(row['demand'] - prev_demand)
      else:
        structure_purchase_list[row['structure']] = []
        structure_purchase_list[row['structure']].append(row['demand'] - prev_demand)
      total_date_purchase += row['demand'] - prev_demand
      prev_demand = row['demand'] * 1.0

  total_date_buyout = 0.0
  prev_date = datetime(1950, 1, 1, 0, 0)
  for index, row in buyout_water.iterrows():
    date_val = row['date']
    if date_val > prev_date:
      informal_transfer.loc[prev_date, 'buyouts'] = total_date_buyout
      prev_demand = 0.0
      total_date_buyout = 0.0
      prev_date =  row['date']

    if row['demand'] > 0.0:
      if row['structure'] in structure_buyout_list:
        structure_buyout_list[row['structure']].append(row['demand'])
      else:
        structure_buyout_list[row['structure']] = []
        structure_buyout_list[row['structure']].append(row['demand'])
      total_date_buyout += row['demand']
  prev_purchase = 0.0
  prev_buyouts = 0.0
  for index, row in informal_transfer.iterrows():
    if row['purchased'] > 0.0:
      informal_transfer.loc[index, 'cumulative_purchase'] = prev_purchase + row['purchased']
      prev_purchase += row['purchased']
    else:
      prev_purhcase = 0.0
    if row['buyouts'] > 0.0:
      informal_transfer.loc[index, 'cumulative_buyouts'] = prev_buyouts + row['buyouts']
      prev_buyouts += row['buyouts']
    else:
      prev_buyouts = 0.0
      
  return informal_transfer, structure_purchase_list, structure_buyout_list

informal_transfer, structure_purchase_list, structure_buyout_list = plot_informal_purchases('5104055')
strut_list = []
strut_length = []
for x in structure_buyout_list:
  strut_list.append(x)
  strut_length.append(len(structure_buyout_list[x]))
structure_buyouts = pd.DataFrame()
structure_buyouts['WDID'] = strut_list
structure_buyouts['length'] = strut_length

purchase_list = []
buyout_list = []
scenarios_use = ['550',]
for folder in scenarios_use:
  purchased_water = pd.read_csv('results_'+ folder + '/purchases_5104055.csv')
  buyout_water = pd.read_csv('results_'+ folder + '/buyouts_2_5104055.csv')
  for index, row in purchased_water.iterrows():
    if row['structure'] in purchase_list:
      pass
    else:
      purchase_list.append(row['structure'])
  for index, row in buyout_water.iterrows():
    if row['structure'] in buyout_list:
      pass
    else:
      buyout_list.append(row['structure'])
      
buyout_structures = {}
buyout_geometries = []
buyout_structures['members'] = []
buyout_structures['id'] = []

purchase_structures = {}
purchase_geometries = []
purchase_structures['members'] = []
purchase_structures['id'] = []

storage_structures = {}
storage_geometries = []
storage_structures['name'] = []
storage_structures['id'] = []
print(len(buyout_list))
print(len(purchase_list))

agg_diversions = pd.read_csv(input_data_dictionary['aggregated_diversions'])
for struct_id in buyout_list:
  num_buyouts = 0.0
  for folder in scenarios_use:
    buyout_water = pd.read_csv('results_'+ folder + '/buyouts_2_5104055.csv')
    num_buyouts += float(len(buyout_water[buyout_water['structure'] == struct_id]))

  this_structure = structures_ucrb[structures_ucrb['WDID'] == struct_id]
  this_irr = irrigation_ucrb[irrigation_ucrb['SW_WDID1'] == struct_id]
  this_ditch = ditches_ucrb[ditches_ucrb['wdid'] == struct_id]
  this_agg = agg_diversions[agg_diversions['statemod_diversion'] == struct_id]
  if len(this_structure) > 0:
    cent_cords = this_structure.dissolve().centroid.iloc[0].coords
    cent_tuple = cent_cords[0]
    this_point = Point(cent_tuple[0], cent_tuple[1])
    buyout_geometries.append(this_point)
    buyout_structures['id'].append(struct_id)
    buyout_structures['members'].append(num_buyouts)
  elif len(this_irr) > 0:
    cent_cords = this_irr.dissolve().centroid.iloc[0].coords
    cent_tuple = cent_cords[0]
    this_point = Point(cent_tuple[0], cent_tuple[1])
    buyout_geometries.append(this_point)
    buyout_structures['id'].append(struct_id)
    buyout_structures['members'].append(num_buyouts)
  elif len(this_ditch) > 0:
    cent_cords = this_ditch.dissolve().centroid.iloc[0].coords
    cent_tuple = cent_cords[0]
    this_point = Point(cent_tuple[0], cent_tuple[1])
    buyout_geometries.append(this_point)
    buyout_structures['id'].append(struct_id)
    buyout_structures['members'].append(num_buyouts)
  elif len(this_agg) > 0:
    this_agg_struct = {}
    this_agg_struct['name'] = []
    this_agg_struct['id'] = []
    this_agg_geom = []
    for index, row in this_agg.iterrows():
      new_irr = irrigation_ucrb[irrigation_ucrb['SW_WDID1'] == str(row['individual_diversion'])]
      for index_irr, row_irr in new_irr.iterrows():
        this_agg_struct['name'].append(row_irr['CROP_TYPE'])
        this_agg_struct['id'].append(row_irr['SW_WDID1'])
        this_agg_geom.append(row_irr['geometry'])
      new_struct = structures_ucrb[structures_ucrb['WDID'] == str(row['individual_diversion'])]
      for index_irr, row_irr in new_struct.iterrows():
        this_agg_struct['name'].append(row_irr['StructName'])
        this_agg_struct['id'].append(row_irr['WDID'])
        this_agg_geom.append(row_irr['geometry'])
      new_ditch = ditches_ucrb[ditches_ucrb['wdid'] == str(row['individual_diversion'])]
      for index_irr, row_irr in new_ditch.iterrows():
        this_agg_struct['name'].append(row_irr['wdid'])
        this_agg_struct['id'].append(row_irr['wdid'])
        this_agg_geom.append(row_irr['geometry'])
    if len(this_agg_geom) > 0:
      this_agg_df = pd.DataFrame(this_agg_struct)
      this_agg_gdf = gpd.GeoDataFrame(this_agg_df, crs = irrigation_ucrb.crs, geometry = this_agg_geom)
      cent_cords = this_agg_gdf.dissolve().centroid.iloc[0].coords
      cent_tuple = cent_cords[0]
      this_point = Point(cent_tuple[0], cent_tuple[1])
      buyout_geometries.append(this_point)
      buyout_structures['id'].append(struct_id)
      buyout_structures['members'].append(num_buyouts)
      
for struct_id in purchase_list:
  this_structure = structures_ucrb[structures_ucrb['WDID'] == struct_id]
  num_purchases = 0.0
  for folder in scenarios_use:
    purchase_water = pd.read_csv('results_'+ folder + '/purchases_5104055.csv')
    num_purchases += float(len(purchase_water[purchase_water['structure'] == struct_id]))
  if len(this_structure) > 0:
    cent_cords = this_structure.dissolve().centroid.iloc[0].coords
    cent_tuple = cent_cords[0]
    this_point = Point(cent_tuple[0], cent_tuple[1])
    purchase_geometries.append(this_point)
    purchase_structures['id'].append(struct_id)
    purchase_structures['members'].append(num_purchases)
  this_irr = irrigation_ucrb[irrigation_ucrb['SW_WDID1'] == struct_id]
  if len(this_irr) > 0:
    cent_cords = this_irr.dissolve().centroid.iloc[0].coords
    cent_tuple = cent_cords[0]
    this_point = Point(cent_tuple[0], cent_tuple[1])
    purchase_geometries.append(this_point)
    purchase_structures['id'].append(struct_id)
    purchase_structures['members'].append(num_purchases)
  this_ditch = ditches_ucrb[ditches_ucrb['wdid'] == struct_id]
  if len(this_ditch) > 0:
    cent_cords = this_ditch.dissolve().centroid.iloc[0].coords
    cent_tuple = cent_cords[0]
    this_point = Point(cent_tuple[0], cent_tuple[1])
    purchase_geometries.append(this_point)
    purchase_structures['id'].append(struct_id)
    purchase_structures['members'].append(num_purchases)
  this_agg = agg_diversions[agg_diversions['statemod_diversion'] == struct_id]
  total_acres = {}
  if len(this_irr) > 0:
    for index, row in this_irr.iterrows():
      if row['CROP_TYPE'] in total_acres:
        total_acres[row['CROP_TYPE']] += row['ACRES']
      else:
        total_acres[row['CROP_TYPE']] = row['ACRES']
    print(struct_id)
    print(total_acres)
  print(this_structure)
  print(this_ditch)
  if len(this_agg) > 0:
    for index, row in this_agg.iterrows():
      new_irr = irrigation_ucrb[irrigation_ucrb['SW_WDID1'] == str(row['individual_diversion'])]
      for index, row in new_irr.iterrows():
        if row['CROP_TYPE'] in total_acres:
          total_acres[row['CROP_TYPE']] += row['ACRES']
        else:
          total_acres[row['CROP_TYPE']] = row['ACRES']
    print(struct_id)
    print(total_acres)
  
  if len(this_agg) > 0:
    this_agg_struct = {}
    this_agg_struct['name'] = []
    this_agg_struct['id'] = []
    this_agg_geom = []
    for index, row in this_agg.iterrows():
      new_irr = irrigation_ucrb[irrigation_ucrb['SW_WDID1'] == str(row['individual_diversion'])]
      for index_irr, row_irr in new_irr.iterrows():
        this_agg_struct['name'].append(row_irr['CROP_TYPE'])
        this_agg_struct['id'].append(row_irr['SW_WDID1'])
        this_agg_geom.append(row_irr['geometry'])
      new_struct = structures_ucrb[structures_ucrb['WDID'] == str(row['individual_diversion'])]
      for index_irr, row_irr in new_struct.iterrows():
        this_agg_struct['name'].append(row_irr['StructName'])
        this_agg_struct['id'].append(row_irr['WDID'])
        this_agg_geom.append(row_irr['geometry'])
      new_ditch = ditches_ucrb[ditches_ucrb['wdid'] == str(row['individual_diversion'])]
      for index_irr, row_irr in new_ditch.iterrows():
        this_agg_struct['name'].append(row_irr['wdid'])
        this_agg_struct['id'].append(row_irr['wdid'])
        this_agg_geom.append(row_irr['geometry'])
    if len(this_agg_geom) > 0:
      this_agg_df = pd.DataFrame(this_agg_struct)
      this_agg_gdf = gpd.GeoDataFrame(this_agg_df, crs = irrigation_ucrb.crs, geometry = this_agg_geom)
      cent_cords = this_agg_gdf.dissolve().centroid.iloc[0].coords
      cent_tuple = cent_cords[0]
      this_point = Point(cent_tuple[0], cent_tuple[1])
      purchase_geometries.append(this_point)
      purchase_structures['id'].append(struct_id)
      purchase_structures['members'].append(num_purchases)


for index, row in structures_ucrb.iterrows():
  if row['StructName'] =='CBT GRANBY RESERVOIR':
    storage_structures['name'].append(row['StructName'])
    storage_structures['id'].append(row['WDID'])
    storage_geometries.append(row['geometry'])

purchase_df = pd.DataFrame(purchase_structures)
purchase_gdf = gpd.GeoDataFrame(purchase_df, crs = irrigation_ucrb.crs, geometry = purchase_geometries)

buyout_df = pd.DataFrame(buyout_structures)
buyout_gdf = gpd.GeoDataFrame(buyout_df, crs = irrigation_ucrb.crs, geometry = buyout_geometries)

storage_df = pd.DataFrame(storage_structures)
storage_gdf = gpd.GeoDataFrame(storage_df, crs = irrigation_ucrb.crs, geometry = storage_geometries)

purchase_gdf.to_file('Shapefiles_UCRB/purchase_partners.shp')
buyout_gdf.to_file('Shapefiles_UCRB/buyout_partners.shp')
storage_gdf.to_file('Shapefiles_UCRB/storage_partners.shp')

streamplatte = streamplatte[streamplatte['ORDER'] > 5]
new_structures = structures_ucrb.merge(structure_buyouts, on = 'WDID')
new_structures = gpd.GeoDataFrame(new_structures, crs = structures_ucrb.crs, geometry = structures_ucrb.geometry)

data_figure = Mapper()
state_use = state_all_T[state_all_T['STUSPS'] == 'CO']
projection_string = 'EPSG:3857'#project raster to data projection
background_map_filename = 'Shapefiles_UCRB/06-B5-mos/colorado_mosaic'
data_figure.plot_scalar_raster(projection_string, background_map_filename, 'Greys_r')
##raster file names
raster_name_pt1 = 'LC08_L1TP_'
raster_name_pt2 = '_02_T1'
raster_band_list = ['_B4', '_B3', '_B2']
raster_id_list = {}
raster_id_list['034033'] = ['20200702_20200913',]
raster_id_list['034032'] = ['20200702_20200913',]
raster_id_list['036032'] = ['20200817_20200920',]
raster_id_list['036033'] = ['20200817_20200920',]
raster_id_list['035032'] = ['20200709_20200912',]
raster_id_list['035033'] = ['20200709_20200912',]
with open('Shapefiles_UCRB/transboundary_diversions.json') as json_data:
  transboundary_stations = json.load(json_data)
lats = []
longs = []
names = []
for x in transboundary_stations['features']:
  lats.append(x['properties']['Latitude'])
  longs.append(x['properties']['Longitude'])
  names.append(x['properties']['TransbasinDiversionName'])
geometry = [Point(xy) for xy in zip(longs, lats)]
transboundary_stations_gdf = gpd.GeoDataFrame(names, crs = 'EPSG:4326', geometry = geometry)
transboundary_stations_gdf = transboundary_stations_gdf.to_crs(epsg = 3857)
transboundary_stations_gdf_all = gpd.GeoDataFrame()
plot_type = 'fig2'
for index, row in transboundary_stations_gdf.iterrows():
  if 'Colorado-Big Thompson' in row[0]: 
    transboundary_stations_gdf_all = pd.concat([transboundary_stations_gdf_all, transboundary_stations_gdf[transboundary_stations_gdf.index == index]])
  elif 'Roberts' in row[0] and plot_type == 'fig_1':
    transboundary_stations_gdf_all = pd.concat([transboundary_stations_gdf_all, transboundary_stations_gdf[transboundary_stations_gdf.index == index]])
  elif 'Moffat' in row[0] and plot_type == 'fig_1':
    transboundary_stations_gdf_all = pd.concat([transboundary_stations_gdf_all, transboundary_stations_gdf[transboundary_stations_gdf.index == index]])
#data_figure.load_batch_raster('UCRB_analysis-master/stitched_satellite/', raster_id_list, raster_name_pt1, raster_name_pt2, raster_band_list, projection_string, max_bright = (100.0, 20000.0), use_gamma = 0.8)

structures_ucrb_all = gpd.GeoDataFrame()
structures_ucrb_all = pd.concat([structures_ucrb_all, structures_ucrb[structures_ucrb['StructName'] == 'CBT GRANBY RESERVOIR']])
if plot_type == 'fig1':
  structures_ucrb_all = pd.concat([structures_ucrb_all, structures_ucrb[structures_ucrb['StructName'] == 'DILLON RESERVOIR']])
    
column1 = []
for x in range(0, len(ucrb_huc8.index)):
  column1.append('UCRB')
ucrb_huc8 = gpd.GeoDataFrame(pd.DataFrame(column1, index = ucrb_huc8.index, columns = ['column1',]), crs = ucrb_huc8.crs, geometry = ucrb_huc8.geometry)
ucrb_huc8 = ucrb_huc8.dissolve(by = 'column1')

agg_diversions = pd.read_csv(input_data_dictionary['aggregated_diversions'])
purchases = pd.read_csv('results_550\purchases_5104055.csv')
aggregated_diversions = []
leased_irr = gpd.GeoDataFrame()
for station_id in ['5100848', '5100585', '51_ADC001']:
  new_irr = irrigation_ucrb[irrigation_ucrb['SW_WDID1'] == station_id]
  total_purchases = purchases[purchases['structure'] == station_id]
  total_lease_list = []
  total_geometry_list = []
  if len(new_irr) > 0:
    for index, row in new_irr.iterrows():
      total_lease_list.append(np.sum(total_purchases['demand']))
      total_geometry_list.append(row['geometry'])
    new_purchases = gpd.GeoDataFrame(pd.DataFrame(total_lease_list, columns = ['Total Leases (AF)']), crs = irrigation_ucrb.crs, geometry = total_geometry_list)
    leased_irr = pd.concat([leased_irr, new_purchases])
  else:
    for index, row in agg_diversions.iterrows():
      if row['statemod_diversion'] == station_id:
        new_irr = irrigation_ucrb[irrigation_ucrb['SW_WDID1'] == str(row['individual_diversion'])]
        for index, row in new_irr.iterrows():
          total_lease_list.append(np.sum(total_purchases['demand']))
          total_geometry_list.append(row['geometry'])
    new_purchases = gpd.GeoDataFrame(pd.DataFrame(total_lease_list, columns = ['Total Leases (AF)']), crs = irrigation_ucrb.crs, geometry = total_geometry_list)
    leased_irr = pd.concat([leased_irr, new_purchases])
leased_irr = gpd.GeoDataFrame(leased_irr, crs = irrigation_ucrb.crs, geometry = leased_irr['geometry'])

crop_list = list(set(irrigation_ucrb['CROP_TYPE']))
perennial_crops = irrigation_ucrb[irrigation_ucrb['PERENNIAL'] == 'YES']
perennial_northern = irrigation_northern[irrigation_northern['PERENNIAL'] == 'YES']
streams_ucrb = gpd.sjoin(streams_ucrb, area_ucrb, how = 'inner', op = 'intersects')
#data_figure.plot_scale(area_ucrb, 'depth', type = 'polygons', solid_color = 'none', solid_alpha = 0.5, linewidth_size = 2.0, outline_color = 'black')
#data_figure.plot_scale(irrigation_ucrb, 'depth', type = 'polygons', solid_color = 'forestgreen', solid_alpha = 1.0, linewidth_size = 1.5, outline_color = 'forestgreen')
#data_figure.plot_scale(state_use, 'depth', type = 'polygons', solid_color = 'none', solid_alpha = 1.0, linewidth_size = 3.0, outline_color = 'black')
#data_figure.plot_scale(northern_area, 'depth', type = 'polygons', solid_color = 'cornsilk', solid_alpha = 0.8, linewidth_size = 1.5, outline_color = 'black')
#data_figure.plot_scale(irrigation_northern, 'depth', type = 'polygons', solid_color = 'forestgreen', solid_alpha = 1.0, linewidth_size = 0.0, outline_color = 'forestgreen')
#data_figure.plot_scale(ditches_ucrb, 'depth', type = 'points', solid_color = 'black', solid_alpha = 0.5, linewidth_size = 0.0, outline_color = 'black', markersize = 2)
if plot_type == 'fig1':
  data_figure.plot_scale(ucrb_huc8, 'depth', type = 'polygons', solid_color = 'forestgreen', solid_alpha = 0.4, linewidth_size = 1.0, outline_color = 'black')
  data_figure.plot_scale(ucrb_huc8, 'depth', type = 'polygons', solid_color = 'none', solid_alpha = 1.0, linewidth_size = 1.0, outline_color = 'black')
  data_figure.plot_scale(northern_districts, 'depth', type = 'polygons', solid_color = 'beige', solid_alpha = 0.4, linewidth_size = 1.0, outline_color = 'black')
  data_figure.plot_scale(northern_districts, 'depth', type = 'polygons', solid_color = 'none', solid_alpha = 1.0, linewidth_size = 1.0, outline_color = 'black')
  data_figure.plot_scale(denver_water, 'depth', type = 'polygons', solid_color = 'beige', solid_alpha = 0.4, linewidth_size = 1.0, outline_color = 'black')
  data_figure.plot_scale(denver_water, 'depth', type = 'polygons', solid_color = 'none', solid_alpha = 1.0, linewidth_size = 1.0, outline_color = 'black')
  data_figure.plot_scale(riverplatte, 'depth', type = 'polygons', solid_color = 'steelblue', solid_alpha = 1.0, linewidth_size = 1.5, outline_color = 'navy')
  data_figure.plot_scale(streamplatte, 'depth', type = 'polygons', solid_color = 'steelblue', solid_alpha = 1.0, linewidth_size = 1.5, outline_color = 'navy')
  data_figure.plot_scale(streams_ucrb, 'depth', type = 'polygons', solid_color = 'steelblue', solid_alpha = 1.0, linewidth_size = 1.5, outline_color = 'navy')
  #data_figure.plot_scale(snow_stations_gdf, 'depth', type = 'points', solid_color = 'navy', solid_alpha = 1.0, linewidth_size = 0.0, outline_color = 'black', markersize = 15)
  #data_figure.plot_scale(perennial_crops, 'depth', type = 'polygons', solid_color = 'goldenrod', solid_alpha = 1.0, linewidth_size = 0.2, outline_color = 'goldenrod')
  #data_figure.plot_scale(perennial_northern, 'depth', type = 'polygons', solid_color = 'goldenrod', solid_alpha = 1.0, linewidth_size = 0.2, outline_color = 'goldenrod')
  #data_figure.plot_scale(new_structures, 'length', type = 'points', solid_color = 'scaled', solid_alpha = 1.0, linewidth_size = 0.0, outline_color = 'black', markersize = 50, value_lim = (0,70), colorscale = 'RdYlBu', zorder = 20)
  data_figure.plot_scale(transboundary_stations_gdf_all, 'depth', type = 'points', solid_color = 'indianred', solid_alpha = 1.0, linewidth_size = 0.0, outline_color = 'black', markersize = 150)
if plot_type == 'fig2':
  data_figure.plot_scale(ucrb_huc8, 'depth', type = 'polygons', solid_color = 'forestgreen', solid_alpha = 0.4, linewidth_size = 1.0, outline_color = 'black')
  data_figure.plot_scale(ucrb_huc8, 'depth', type = 'polygons', solid_color = 'none', solid_alpha = 1.0, linewidth_size = 1.0, outline_color = 'black')
  data_figure.plot_scale(northern_districts, 'depth', type = 'polygons', solid_color = 'beige', solid_alpha = 0.4, linewidth_size = 1.0, outline_color = 'black')
  data_figure.plot_scale(northern_districts, 'depth', type = 'polygons', solid_color = 'none', solid_alpha = 1.0, linewidth_size = 1.0, outline_color = 'black')
  data_figure.plot_scale(transboundary_stations_gdf_all, 'depth', type = 'points', solid_color = 'indianred', solid_alpha = 1.0, linewidth_size = 0.0, outline_color = 'black', markersize = 200)
  data_figure.plot_scale(riverplatte, 'depth', type = 'polygons', solid_color = 'steelblue', solid_alpha = 1.0, linewidth_size = 1.5, outline_color = 'navy')
  data_figure.plot_scale(streamplatte, 'depth', type = 'polygons', solid_color = 'steelblue', solid_alpha = 1.0, linewidth_size = 1.5, outline_color = 'navy')
  data_figure.plot_scale(streams_ucrb, 'depth', type = 'polygons', solid_color = 'steelblue', solid_alpha = 1.0, linewidth_size = 1.5, outline_color = 'navy')
  data_figure.plot_scale(structures_ucrb_all, 'depth', type = 'points', solid_color = 'steelblue', solid_alpha = 1.0, linewidth_size = 0.0, outline_color = 'black', markersize = 200)
if plot_type == 'fig3':
  data_figure.plot_scale(ucrb_huc8, 'depth', type = 'polygons', solid_color = 'forestgreen', solid_alpha = 0.4, linewidth_size = 1.0, outline_color = 'black')
  data_figure.plot_scale(ucrb_huc8, 'depth', type = 'polygons', solid_color = 'none', solid_alpha = 1.0, linewidth_size = 1.0, outline_color = 'black')
  data_figure.plot_scale(northern_districts, 'depth', type = 'polygons', solid_color = 'beige', solid_alpha = 0.4, linewidth_size = 1.0, outline_color = 'black')
  data_figure.plot_scale(northern_districts, 'depth', type = 'polygons', solid_color = 'none', solid_alpha = 1.0, linewidth_size = 1.0, outline_color = 'black')
  data_figure.plot_scale(riverplatte, 'depth', type = 'polygons', solid_color = 'steelblue', solid_alpha = 1.0, linewidth_size = 1.0, outline_color = 'steelblue')
  data_figure.plot_scale(streamplatte, 'depth', type = 'polygons', solid_color = 'steelblue', solid_alpha = 1.0, linewidth_size = 1.0, outline_color = 'steelblue')
  data_figure.plot_scale(streams_ucrb, 'depth', type = 'polygons', solid_color = 'steelblue', solid_alpha = 1.0, linewidth_size = 1.0, outline_color = 'steelblue')
  data_figure.plot_scale(leased_irr, 'Total Leases (AF)', type = 'polygons', solid_color = 'scaled', solid_alpha = 1.0, linewidth_size = 0.25, outline_color = 'black', value_lim = (0,50000), colorscale = 'OrRd', zorder = 20)
  data_figure.plot_scale(transboundary_stations_gdf_all, 'depth', type = 'points', solid_color = 'indianred', solid_alpha = 1.0, linewidth_size = 0.0, outline_color = 'black', markersize = 150)
  data_figure.plot_scale(structures_ucrb_all, 'depth', type = 'points', solid_color = 'steelblue', solid_alpha = 1.0, linewidth_size = 0.0, outline_color = 'black', markersize = 150)
if plot_type == 'fig4':
  data_figure.plot_scale(ucrb_huc8, 'depth', type = 'polygons', solid_color = 'forestgreen', solid_alpha = 0.4, linewidth_size = 1.0, outline_color = 'black')
  data_figure.plot_scale(ucrb_huc8, 'depth', type = 'polygons', solid_color = 'none', solid_alpha = 1.0, linewidth_size = 1.0, outline_color = 'black')
  data_figure.plot_scale(streams_ucrb, 'depth', type = 'polygons', solid_color = 'steelblue', solid_alpha = 0.6, linewidth_size = 1.0, outline_color = 'steelblue')
  data_figure.plot_scale(buyout_gdf, 'members', type = 'points', solid_color = 'scaled', solid_alpha = 1.0, linewidth_size = 0.0, outline_color = 'black', markersize = 50, value_lim = (0,np.max(purchase_gdf['members'])), colorscale = 'Reds')
  data_figure.plot_scale(purchase_gdf, 'depth', type = 'points', solid_color = 'navy', solid_alpha = 1.0, linewidth_size = 0.0, outline_color = 'black', markersize = 75, zorder = 25)
  data_figure.plot_scale(storage_gdf, 'depth', type = 'points', solid_color = 'gold', solid_alpha = 1.0, linewidth_size = 0.0, outline_color = 'black', markersize = 150, zorder = 30)
  
#plot colorbar w/perimeter
xl = state_use.total_bounds[0]
xr = state_use.total_bounds[2]
by = state_use.total_bounds[1]
uy = state_use.total_bounds[3]
xrange = northern_districts.total_bounds[2] - ucrb_huc8.total_bounds[0]
yrange = northern_districts.total_bounds[3] - ucrb_huc8.total_bounds[1]
county_filename = 'BraysBayou/tl_2021_us_state/tl_2021_us_state.shp'
#data_figure.add_inset_figure(state_all, (xl, xr), (by, uy), (xl - 1.5*xrange, xr + 1.5*xrange), (by - 1.5*yrange, uy + 1.5*yrange), 3857) 
if plot_type == 'fig1':
  legend_location = 'lower right'
  data_figure.format_plot(xlim = (ucrb_huc8.total_bounds[0] + xrange*0.2, northern_districts.total_bounds[2] - xrange*0.2), ylim = (ucrb_huc8.total_bounds[1], northern_districts.total_bounds[3]))
  legend_element = [Patch(facecolor='forestgreen', edgecolor='black', label='Upper Colorado Basin', alpha = 0.4),
                  Patch(facecolor='beige', edgecolor='black', label='Front Range Districts', alpha = 0.4),
                  Line2D([0], [0], markerfacecolor='indianred', markeredgecolor='black',  lw = 0, marker = 'o', markersize = 10, label='Transboundary Diversion')]
if plot_type == 'fig2':
  legend_location = 'lower right'
  data_figure.format_plot(xlim = (ucrb_huc8.total_bounds[0] + xrange*0.225, northern_districts.total_bounds[2] - xrange*0.175), ylim = (ucrb_huc8.total_bounds[1], northern_districts.total_bounds[3]))
  legend_element = [Patch(facecolor='forestgreen', edgecolor='black', label='Upper Colorado\nRiver Basin', alpha = 0.8),
                  Patch(facecolor='beige', edgecolor='black', label='Colorado - Big\nThompson Service Area', alpha = 0.8),
                  Line2D([0], [0], markerfacecolor='indianred', markeredgecolor='black',  lw = 0, marker = 'o', markersize = 15, label='Adams Tunnel'),
                  Line2D([0], [0], markerfacecolor='steelblue', markeredgecolor='black',  lw = 0, marker = 'o', markersize = 15, label='Lake Granby')]
if plot_type == 'fig3':
  legend_location = 'lower right'
  lx = ucrb_huc8.total_bounds[0]
  rx = northern_districts.total_bounds[2]
  by = ucrb_huc8.total_bounds[1]
  ty  = northern_districts.total_bounds[3]
  data_figure.format_plot(xlim = (lx + xrange*0.435, rx - xrange*0.475), ylim = (by + yrange * 0.55, ty - yrange * 0.275))
  p5 = Polygon([(lx + xrange*0.4375, by + yrange * 0.56), (lx + xrange*0.4375, ty - yrange * 0.285), (lx + xrange*0.455, ty - yrange * 0.285), (lx + xrange*0.455, by + yrange * 0.56)])
  df4 = gpd.GeoDataFrame({'geometry': p5, 'df4':[1,1]})
  df4.crs = ucrb_huc8.crs
  data_figure.plot_scale(df4, 'depth', type = 'polygons', solid_color = 'beige', solid_alpha = 0.4, linewidth_size = 1.0, outline_color = 'goldenrod')
  data_figure.add_colorbar([0.15, 0.165, 0.075, 0.65], [0, 50000], ['0', '50'], 'Total Leases\n(tAF)', colorscale = 'OrRd')
  legend_element = [Patch(facecolor='forestgreen', edgecolor='black', label='Upper Colorado\nRiver Basin', alpha = 0.4),
                  Patch(facecolor='beige', edgecolor='black', label='Colorado - Big\nThompson Service Area', alpha = 0.4),
                  Line2D([0], [0], markerfacecolor='indianred', markeredgecolor='black',  lw = 0, marker = 'o', markersize = 15, label='Transboundary Diversion'),
                  Line2D([0], [0], markerfacecolor='steelblue', markeredgecolor='black',  lw = 0, marker = 'o', markersize = 15, label='Lake Granby')]

if plot_type == 'fig4':
  legend_location = 'upper left'
  data_figure.format_plot(xlim = (ucrb_huc8.total_bounds[0] - xrange*0.01, ucrb_huc8.total_bounds[2] + xrange*0.01), ylim = (ucrb_huc8.total_bounds[1], ucrb_huc8.total_bounds[3]))
  legend_element = [Circle((0,0), facecolor = 'gold', edgecolor = 'black', label = 'Lake Granby'),
                    Circle((0,0), facecolor = 'navy', edgecolor = 'black', label = 'Lease Sellers')]
                  
  lx = ucrb_huc8.total_bounds[0]
  rx = ucrb_huc8.total_bounds[2]
  by = ucrb_huc8.total_bounds[1]
  ty  = ucrb_huc8.total_bounds[3]
  xrange = ucrb_huc8.total_bounds[2] - ucrb_huc8.total_bounds[0]
  yrange = ucrb_huc8.total_bounds[3] - ucrb_huc8.total_bounds[1]
  p5 = Polygon([(lx + xrange*0.0, by + yrange * 0.485), (lx + xrange*0.0, by + yrange * 0.8075), (lx + xrange*0.29, by + yrange * 0.8075), (lx + xrange*0.29, by + yrange * 0.485)])
  df4 = gpd.GeoDataFrame({'geometry': p5, 'df4':[1,1]})
  df4.crs = ucrb_huc8.crs
  data_figure.plot_scale(df4, 'depth', type = 'polygons', solid_color = 'beige', solid_alpha = 0.3, linewidth_size = 0.0)
  data_figure.add_colorbar([0.19, 0.5375, 0.025, 0.10], [0, 50000], ['0%', '100%'], 'Facilitator\nOption Utilization', colorscale = 'Reds')

legend_properties = {'family':'Gill Sans MT','weight':'bold','size':14}
data_figure.add_legend(legend_location, legend_element, legend_properties)
plt.savefig('Shapefiles_UCRB/' + plot_type + '.png', dpi = 300, bbox_inches = 'tight', pad_inches = 0.0)

         