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

input_data_dictionary = crss.create_input_data_dictionary('B', 'A')

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

fig, ax = plt.subplots(figsize = (8,12))
current_water_use = 0.0
counter = 0
running_area = 0.0
for x in sorted_crops:
  total_cost = marginal_net_benefits[x] / (et_requirements[x] / 12.0)
  if total_cost > 10000.0:
    total_cost -= 9000.0
  elif total_cost > 5500.0:
    total_cost -= 5400.0
  total_area = overall_crop_areas[x] / 1000.0
  print(x, end = " ")
  print(total_area, end = " ")
  print(total_cost)
  ax.fill_between([running_area, running_area + total_area], np.zeros(2), [total_cost, total_cost], facecolor = 'indianred', edgecolor = 'black', linewidth = 2.0)
  counter += 1.5
  running_area += total_area
ax.set_yticks([0.0, 200.0, 400.0, 600.0, 800.0, 1000.0, 1200.0, 1400.0])
ax.set_yticklabels(['$0', '$200', '$5600', '$5800', '$6000', '$10000', '$10200', '$10400'])
counter = 0.5
label_list = ['Grapes', 'Orchard', 'Vegetables', 'Bluegrass', 'Sorghum', 'Alfalfa', 'Corn', 'Pasture', 'Wheat', 'Beans', 'Grain', 'Barley']
ax.set_ylabel('Fallow cost ($/AF)', fontsize = 28, weight = 'bold', fontname = 'Gill Sans MT')
ax.set_xlabel('Basinwide Planting\n(1000 acres)', fontsize = 28, weight = 'bold', fontname = 'Gill Sans MT')
ax.set_ylim([0.0, 1400.0])
for item in (ax.get_xticklabels()):
  item.set_fontsize(20)
  item.set_fontname('Gill Sans MT')
for axesnum in range(0, 3):
  for item in (ax.get_yticklabels()):
    item.set_fontsize(20)
    item.set_fontname('Gill Sans MT')

plt.savefig('Shapefiles_UCRB/crop_types.png', dpi = 300, bbox_inches = 'tight', pad_inches = 0.0)
