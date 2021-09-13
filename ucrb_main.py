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

project_folder = 'UCRB_analysis-master/'

#Input Data Names
input_data_dictionary = {}
input_data_dictionary['hydrography'] = project_folder + 'Shapefiles_UCRB/NHDPLUS_H_1401_HU4_GDB.gdb'
input_data_dictionary['structures'] = project_folder + 'Shapefiles_UCRB/Div_5_structures.shp'
input_data_dictionary['snow'] = project_folder + 'Adaptive_experiment/Snow_Data/'
input_data_dictionary['structure_demand'] = project_folder + 'Adaptive_experiment/input_files/cm2015B.ddm'
input_data_dictionary['structure_rights'] = project_folder + 'Adaptive_experiment/input_files/cm2015B.ddr'
input_data_dictionary['downstream'] = project_folder + 'Adaptive_experiment/input_files/cm2015.rin'
input_data_dictionary['historical_reservoirs'] = project_folder + 'Adaptive_experiment/input_files/cm2015.eom'
input_data_dictionary['reservoir_storage'] = project_folder + 'Adaptive_experiment/output_files/cm2015B.xre'
input_data_dictionary['deliveries'] = project_folder + 'Adaptive_experiment/output_files/cm2015B.xdd'
input_data_dictionary['calls'] = project_folder + 'Adaptive_experiment/output_files/cm2015B.xca'
input_data_dictionary['HUC4'] = ['1401',]
input_data_dictionary['HUC8'] = ['14010001', '14010002', '14010003', '14010004', '14010005']

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

#Create Basin class with the same extent as StateMod Basin
ucrb = Basin(input_data_dictionary)
year_start = 1908
year_end = 2013

##Read water rights links to structures, structures links to stream network
rights_data = ucrb.read_text_file(input_data_dictionary['structure_rights'])
downstream_data = ucrb.read_text_file(input_data_dictionary['downstream'])
ucrb.read_rights_data(rights_data)
ucrb.read_downstream_structure(downstream_data)
ucrb.create_rights_stack()

##Historical Data for Adams Tunnel
reservoir_list = ['5104055', '5103710', '5103695']#GRANBY, WILLOW CREEK, SHADOW MNT
reservoir_names = ['GRANBY', 'WILLOW CREEK', 'SHADOW MNT']
adams_tunnel = '5104634'
lake_granby = '5104055'

reservoir_data = crss.read_text_file(input_data_dictionary['reservoir_storage'])
historical_reservoir_data = crss.read_text_file(input_data_dictionary['historical_reservoirs'])
historical_reservoir_timeseries = crss.read_historical_reservoirs(historical_reservoir_data, reservoir_list, year_start, year_end)
simulated_reservoir_timeseries = crss.read_simulated_reservoirs(reservoir_data, reservoir_list, year_start, year_end)

#Make Historical Plots - Storage Validation
reservoir_figure = Plotter(project_folder + 'Adaptive_experiment/results/reservoir_validation_CBT.png', nr = 3)
reservoir_figure.plot_reservoir_figures(historical_reservoir_timeseries, simulated_reservoir_timeseries, reservoir_names)
del reservoir_figure

#Make Historical Plots - Storage vs Diversions
delivery_data = crss.read_text_file(input_data_dictionary['deliveries'])
structure_list = ['5104634',]
simulated_diversion_timeseries = crss.read_simulated_diversions(delivery_data, structure_list, year_start, year_end)

reservoir_figure = Plotter(project_folder + 'Adaptive_experiment/results/snowpack_diversions.png', nr = 2)
reservoir_figure.plot_reservoir_simulation(simulated_reservoir_timeseries, simulated_diversion_timeseries, '5104055', '5104634', ['2', '3', '1'])
del reservoir_figure

#Make Historical Plots - Snowpack Indices vs. available water
structure_list = ['5104055',]
simulated_release_timeseries = crss.read_simulated_control_release(delivery_data, structure_list, year_start, year_end)
reservoir_figure = Plotter(project_folder + 'Adaptive_experiment/results/flow_past_station.png', nr = 3)
snow_coefs = reservoir_figure.plot_release_simulation(simulated_release_timeseries, ucrb.basin_snowpack['14010001'], simulated_diversion_timeseries, '5104055', '5104634', 1950, 2013)
del reservoir_figure

#Make Historical Plots - Snowpack indices vs. available water, grouped by month & current controlling call
reservoir_figure = Plotter(project_folder + 'Adaptive_experiment/results/controlled_flow_past_station.png', nr = 3, nc = 4)
snow_coefs = reservoir_figure.plot_release_simulation_controlled(simulated_release_timeseries, ucrb.basin_snowpack['14010001'], simulated_diversion_timeseries, '5104055', '5104634', 1950, 2013)
del reservoir_figure

#Make Historical Plots - Simulated 'total water available' for Adams Tunnel
downstream_data = crss.read_text_file(input_data_dictionary['downstream'])
column_lengths = [12, 24, 13, 17, 4]
rights_stack = crss.read_rights(downstream_data, column_lengths)
reservoir_figure = Plotter(project_folder + 'Adaptive_experiment/results/available_water.png')
reservoir_figure.plot_available_water(simulated_reservoir_timeseries, ucrb.basin_snowpack['14010001'], simulated_diversion_timeseries, snow_coefs, '5104055', '5104634', 1950, 2013)
del reservoir_figure

ucrb.create_new_simulation(input_data_dictionary, start_year, end_year)
template_filename = design + '/Input_files/cm2015B_template.rsp'
demand_filename = design + '/Input_files/cm2015B_A.ddm'
control_filename = design + '/Input_files/cm2015.ctl'
d = {}
d['DDM'] = 'cm2015B_A.ddm'
d['CTL'] = 'cm2015_A.ctl'

T = open(template_filename, 'r')
template_RSP = Template(T.read())
S1 = template_RSP.safe_substitute(d)
f1 = open(design+'/Experiment_files/cm2015B_A.rsp', 'w')
f1.write(S1)    
f1.close()
for year in range(start_year, end_year):
  writenewDDM(demand_filename, structure_list, reduced_demand, new_demand, structure_receive, year_change, month_change)
  os.system("StateMod_Model_15.exe cm2015B_A -simulate")



