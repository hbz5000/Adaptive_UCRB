import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from osgeo import gdal
import rasterio
from shapely.geometry import Point, Polygon, LineString
from matplotlib.patches import Patch, Circle
from matplotlib.lines import Line2D
import geopandas as gpd
import fiona
from matplotlib.colors import ListedColormap
import matplotlib.pylab as pl
from skimage import exposure
import seaborn as sns
import sys
import scipy.stats as stats
from datetime import datetime, timedelta
import matplotlib.ticker as mtick
import matplotlib.dates as mdates

class Plotter():

  def __init__(self, figure_name, nr = 1, nc = 0, figsize = (15,15)):
    self.sub_rows = nr
    self.sub_cols = nc
    self.figure_name = figure_name
    if self.sub_cols == 0:
      self.fig, self.ax = plt.subplots(self.sub_rows, figsize = figsize)
      if self.sub_rows == 1:
        self.type = 'single'
        self.ax.grid(False)
      else:
        self.type = '1d'
    else:
      self.fig, self.ax = plt.subplots(nrows = self.sub_rows, ncols = self.sub_cols, figsize = figsize)
      self.type = '2d'
    plt.tight_layout()

  def plot_historical_exports(self, structure_deliveries, structure_deliveries_new, snowpack_data, structure_id, tunnel_name, start_year, end_year):
  
    month_list = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    color_list = sns.color_palette('RdBu', 10)
    annual_deliveries = []
    annual_deliveries_new = []
    
    total_annual_deliveries = 0.0
    for index, row in structure_deliveries.iterrows():
      if index.year >= start_year and index.year <= end_year:
        if index.month == 10 and total_annual_deliveries > 0.0:
          annual_deliveries.append(total_annual_deliveries)
          total_annual_deliveries = 0.0
        if structure_id == '3604684':
          total_annual_deliveries += (row[structure_id] + row['5104655'])/1000.0
        else:
          total_annual_deliveries += row[structure_id]/1000.0   
        
    annual_deliveries.append(total_annual_deliveries)        
    annual_deliveries = np.asarray(annual_deliveries)

    total_annual_deliveries = 0.0
    for index, row in structure_deliveries_new.iterrows():
      if index.year >= start_year and index.year <= end_year:
        if index.month == 10 and total_annual_deliveries > 0.0:
          annual_deliveries_new.append(total_annual_deliveries)
          total_annual_deliveries = 0.0
        if structure_id == '3604684':
          total_annual_deliveries += (row[structure_id] + row['5104655'])/1000.0
        else:
          total_annual_deliveries += row[structure_id]/1000.0   
    annual_deliveries_new.append(total_annual_deliveries)        
    annual_deliveries_new = np.asarray(annual_deliveries_new)
    max_delivery = max(np.max(annual_deliveries), np.max(annual_deliveries_new))
    min_delivery = min(np.min(annual_deliveries), np.min(annual_deliveries_new))
    range_use = max_delivery - min_delivery
    pos = np.linspace(min_delivery, max_delivery, 101)
    kde_est = stats.gaussian_kde(annual_deliveries)
    kde_est2 = stats.gaussian_kde(annual_deliveries_new)
    mean_pre = np.mean(annual_deliveries)
    mean_post = np.mean(annual_deliveries_new)
    self.ax[0].fill_between(pos, kde_est(pos), edgecolor = 'black', alpha = 1.0, facecolor = 'beige')
    self.ax[0].fill_between(pos, kde_est2(pos), edgecolor = 'black', alpha = 1.0, facecolor = 'beige')
    self.ax[0].fill_between(pos, kde_est(pos), edgecolor = 'black', alpha = 0.6, facecolor = 'purple')
    self.ax[0].fill_between(pos, kde_est2(pos), edgecolor = 'black', alpha = 0.6, facecolor = 'forestgreen')
    self.ax[0].fill_between([mean_pre - range_use * 0.0075, mean_pre + range_use * 0.0075], [0.0, 0.0], [np.max(kde_est(pos)), np.max(kde_est(pos))], linewidth = 3.0, edgecolor = 'black', facecolor = 'purple')
    self.ax[0].fill_between([mean_post - range_use * 0.0075, mean_post + range_use * 0.0075], [0.0, 0.0], [np.max(kde_est2(pos)),np.max(kde_est2(pos))], linewidth = 3.0, edgecolor = 'black', facecolor = 'Forestgreen')
    snowpack_vals = np.zeros(end_year - start_year + 1)
    for yearnum in range(start_year, end_year + 1):
      datetime_val = datetime(yearnum, 9, 1, 0, 0)
      snowpack_vals[yearnum-start_year] = snowpack_data.loc[datetime_val, 'basinwide_average']
    index_sort = np.argsort(snowpack_vals)
    wet_years = []
    normal_years = []
    dry_years = []
    num_vals = len(index_sort) / 3
    counter = 0
    counter_type = 0
    for x in range(0, len(index_sort)):
      index_use = index_sort[x]
      if counter_type == 0:
        dry_years.append(annual_deliveries_new[index_use] - annual_deliveries[index_use])
      elif counter_type == 1:
        normal_years.append(annual_deliveries_new[index_use] - annual_deliveries[index_use])
      else:
        wet_years.append(annual_deliveries_new[index_use] - annual_deliveries[index_use])
      counter += 1
      if counter > num_vals:
        counter = 0
        counter_type += 1
    wet_years = np.asarray(wet_years)
    normal_years = np.asarray(normal_years)
    dry_years = np.asarray(dry_years)
    max_delivery2 = max(max(np.max(wet_years), np.max(dry_years)), np.max(normal_years))
    min_delivery2 = min(min(np.min(wet_years), np.min(dry_years)), np.min(normal_years))
    range_use = max_delivery2 - min_delivery2
    pos2 = np.linspace(min_delivery2, max_delivery2, 101)
    kde_wet = stats.gaussian_kde(wet_years)
    kde_normal = stats.gaussian_kde(normal_years)
    kde_dry = stats.gaussian_kde(dry_years)
    self.ax[1].fill_between(pos2, kde_wet(pos2), edgecolor = 'black', alpha = 1.0, facecolor = 'beige')
    self.ax[1].fill_between(pos2, kde_normal(pos2), edgecolor = 'black', alpha = 1.0, facecolor = 'beige')
    self.ax[1].fill_between(pos2, kde_dry(pos2), edgecolor = 'black', alpha = 1.0, facecolor = 'beige')
    self.ax[1].fill_between(pos2, kde_wet(pos2), edgecolor = 'black', alpha = 1.0, facecolor = 'steelblue')
    self.ax[1].fill_between(pos2, kde_normal(pos2), edgecolor = 'black', alpha = 1.0, facecolor = 'beige')
    self.ax[1].fill_between(pos2, kde_dry(pos2), edgecolor = 'black', alpha = 1.0, facecolor = 'indianred')
    mean_wet = np.mean(wet_years)
    mean_normal = np.mean(normal_years)
    mean_dry = np.mean(dry_years)
    self.ax[1].fill_between([mean_wet - range_use * 0.0075, mean_wet + range_use * 0.0075], [0.0, 0.0], [np.max(kde_wet(pos2)), np.max(kde_wet(pos2))], linewidth = 3.0, edgecolor = 'black', facecolor = 'steelblue')
    self.ax[1].fill_between([mean_normal - range_use * 0.0075, mean_normal + range_use * 0.0075], [0.0, 0.0], [np.max(kde_normal(pos2)),np.max(kde_normal(pos2))], linewidth = 3.0, edgecolor = 'black', facecolor = 'beige')
    self.ax[1].fill_between([mean_dry - range_use * 0.0075, mean_dry + range_use * 0.0075], [0.0, 0.0], [np.max(kde_dry(pos2)),np.max(kde_dry(pos2))], linewidth = 3.0, edgecolor = 'black', facecolor = 'indianred')
    self.ax[0].set_yticks([])
    self.ax[0].set_yticklabels('')
    self.ax[1].set_yticks([])
    self.ax[1].set_yticklabels('')
    self.ax[0].set_ylabel('Probability', fontsize = 24, weight = 'bold', fontname = 'Gill Sans MT')
    self.ax[1].set_ylabel('Probability', fontsize = 24, weight = 'bold', fontname = 'Gill Sans MT')
    if tunnel_name == 'Roberts':
      self.ax[0].set_xlabel('Annual Exports, ' + tunnel_name + ' & Moffat Tunnel (tAF)', fontsize = 24, weight = 'bold', fontname = 'Gill Sans MT')
      self.ax[1].set_xlabel('Change in Exports, ' + tunnel_name + ' & Moffat Tunnel (tAF)', fontsize = 24, weight = 'bold', fontname = 'Gill Sans MT')
    else:
      self.ax[0].set_xlabel('Annual Exports, ' + tunnel_name + ' Tunnel (tAF)', fontsize = 24, weight = 'bold', fontname = 'Gill Sans MT')
      self.ax[1].set_xlabel('Change in Exports, ' + tunnel_name + ' Tunnel (tAF)', fontsize = 24, weight = 'bold', fontname = 'Gill Sans MT')
    
    legend_location = 'upper left'
    legend_element = [Patch(facecolor='purple', edgecolor='black', alpha = 1., label='Historical Baseline'),
                     Patch(facecolor='forestgreen', edgecolor='black', alpha = 1., label='Reservoir Re-operation'),
                     Line2D([0], [0], color='black', lw = 4, label='Distribution Mean')]
    legend_properties = {'family':'Gill Sans MT','weight':'bold','size':18}
    self.ax[0].legend(handles=legend_element, loc=legend_location, prop=legend_properties)
    legend_element2 = [Patch(facecolor='indianred', edgecolor='black', alpha = 1., label='Dry Years'),
                     Patch(facecolor='beige', edgecolor='black', alpha = 1., label='Normal Years'),
                     Patch(facecolor='steelblue', edgecolor='black', alpha = 1., label='Wet Years'),
                     Line2D([0], [0], color='black', lw = 4, label='Distribution Mean')]
    self.ax[1].legend(handles=legend_element2, loc=legend_location, prop=legend_properties)
    for x in range(0, 2):
      for item in (self.ax[x].get_xticklabels()):
        item.set_fontsize(18)
    self.ax[0].set_xlim([min_delivery, max_delivery])
    self.ax[1].set_xlim([min_delivery2, max_delivery2])
    plt.tight_layout()
    plt.savefig('Shapefiles_UCRB/' + self.figure_name, dpi = 300, bbox_inches = 'tight', pad_inches = 0.0)

  def plot_monthly_ranges(self):
    for decile in range(0, 10):
      bottom_line = np.zeros(12)
      top_line = np.zeros(12)
      for mn_cnt, mn in enumerate(month_list):
        decile_index_bottom = int(float(len(monthly_deliveries[mn])) * float(decile)/10.0)
        decile_index_top = int(float(len(monthly_deliveries[mn])) * float(decile + 1)/10.0)
        if decile_index_top == len(monthly_deliveries[mn]):
          decile_index_top -= 1
        bottom_line[mn_cnt] = monthly_deliveries[mn][decile_index_bottom]
        top_line[mn_cnt] = monthly_deliveries[mn][decile_index_top]
      self.ax[1].fill_between(np.arange(len(month_list)), bottom_line, top_line, color = 'beige', alpha = 1.0)
      self.ax[1].fill_between(np.arange(len(month_list)), bottom_line, top_line, color = color_list[decile], alpha = 0.7)

  def plot_transfer_tradeoffs(self):
    thresh_550 = pd.read_csv('all_changes_0.csv')
    thresh_600 = pd.read_csv('all_changes_1.csv')
    thresh_650 = pd.read_csv('all_changes_2.csv')
    thresh_700 = pd.read_csv('all_changes_3.csv')

    freq_550 = pd.read_csv('freq_changes_0.csv')
    freq_600 = pd.read_csv('freq_changes_1.csv')
    freq_650 = pd.read_csv('freq_changes_2.csv')
    freq_700 = pd.read_csv('freq_changes_3.csv')
    plot_labels = ['550', '600', '650', '700']
    total_leases = np.zeros(4)
    mf_impact = np.zeros(4)
    other_impact = np.zeros(4)
    freq_tot = np.zeros(4)
    freq_mf = np.zeros(4)
    freq_other = np.zeros(4)
    x_cnt = 0
    for lease_use, freq_use in zip([thresh_550, thresh_600, thresh_650, thresh_700], [freq_550, freq_600, freq_650, freq_700]):
      total_leases[x_cnt] = lease_use.loc[0, '0']
      mf_impact[x_cnt] = lease_use.loc[2, '0']
      other_impact[x_cnt] = (lease_use.loc[1, '0'] + lease_use.loc[3, '0'])
    
      freq_tot[x_cnt] = freq_use.loc[0, '0']
      freq_mf[x_cnt] = freq_use.loc[1, '0']
      freq_other[x_cnt] = freq_use.loc[2, '0']
      
      x_cnt += 1
    color_list = sns.color_palette('RdYlBu', 3)
    self.ax.plot(freq_tot, total_leases, color = 'indianred', linewidth = 5.0, zorder = 2) 
    self.ax.scatter(freq_tot, total_leases, c = 'indianred', s = 250, edgecolors = 'black', zorder = 10)
    for fr, vol, lab in zip(freq_tot, total_leases, plot_labels):
      if lab == '700':
        self.ax.text(fr + np.max(freq_tot) * 0.075, vol - 0.05 * np.max(total_leases), 'Contract\nTrigger:\n' + lab + ' tAF', fontsize = 18, weight = 'bold', fontname = 'Gill Sans MT',verticalalignment='center',
            horizontalalignment='center', zorder = 20)
      else:
        self.ax.text(fr - np.max(freq_tot) * 0.075, vol + 0.05 * np.max(total_leases), 'Contract\nTrigger:\n' + lab + ' tAF', fontsize = 18, weight = 'bold', fontname = 'Gill Sans MT',verticalalignment='center',
            horizontalalignment='center', zorder = 20)

    self.ax.plot(freq_mf, mf_impact, color = 'steelblue', linewidth = 5.0, zorder = 2) 
    self.ax.scatter(freq_mf, mf_impact, c = 'steelblue', s = 250, edgecolors = 'black', zorder = 10) 

    self.ax.plot(freq_other, other_impact, color = 'forestgreen', linewidth = 5.0, zorder = 2) 
    self.ax.scatter(freq_other, other_impact, c = 'forestgreen', s = 250, edgecolors = 'black', zorder = 10) 

    legend_location = 'upper left'
    legend_element = [Patch(facecolor='indianred', edgecolor='black', label='Transbasin Diversion'),
                     Patch(facecolor='steelblue', edgecolor='black', label='Minimum Flow Impacts'),
                     Patch(facecolor='forestgreen', edgecolor='black', label='Other Third Parties')]
    self.ax.plot([0, 10], [0.0, 0.0], color = 'black', linewidth = 2.0, zorder = 1)       
    self.ax.set_xlim([0,10])    
    legend_properties = {'family':'Gill Sans MT','weight':'bold','size':24}
    self.ax.legend(handles=legend_element, loc=legend_location, prop=legend_properties, ncol = 1)
    self.ax.set_ylabel('Total Diversion Change (tAF)', fontsize = 24, weight = 'bold', fontname = 'Gill Sans MT')
    self.ax.set_xlabel('Impact Frequency (maximum per 10-year period)', fontsize = 24, weight = 'bold', fontname = 'Gill Sans MT')
    for item in (self.ax.get_xticklabels()):
      item.set_fontsize(20)
    for item in (self.ax.get_yticklabels()):
      item.set_fontsize(20)
    plt.savefig('Shapefiles_UCRB/' + self.figure_name, dpi = 300, bbox_inches = 'tight', pad_inches = 0.0)

  def plot_buyout_mitigation(self, station_no, baseline_revenues, annual_water_index, start_year, buyout_price):
    print(baseline_revenues)
    mitigation_dict = {}
    exclusive_mitigation = {}
    color_list = sns.color_palette('rocket', 4)
    for s_cnt, folder_name in enumerate(['550', '600', '650', '700']):
      mitigation_dict[folder_name] = np.zeros(len(baseline_revenues))
      exclusive_mitigation[folder_name] = {}
      exclusive_mitigation[folder_name]['buyout'] = []
      exclusive_mitigation[folder_name]['shortfall'] = []
      structure_buyouts = pd.read_csv('results_' + folder_name + '/buyouts_2_5104055.csv')
      structure_buyouts['datetime'] = pd.to_datetime(structure_buyouts['date'])
      this_station_buyouts = structure_buyouts[structure_buyouts['structure'] == station_no]
      this_station_buyouts = this_station_buyouts.drop_duplicates(subset = 'datetime')
      for index, row in this_station_buyouts.iterrows():
        mitigation_dict[folder_name][row['datetime'].year - start_year] += row['demand_purchase'] * buyout_price / 1000000.0
        print(folder_name, end = " ")
        print(row['datetime'].year - start_year, end = " ")
        print(row['demand_purchase'], end = " ")
        print(row['demand_purchase'] * buyout_price / 1000000.0, end = " ")
        print(mitigation_dict[folder_name][row['datetime'].year - start_year])
    exclusive_mitigation['none'] = {}  
    exclusive_mitigation['none']['buyout'] = []  
    exclusive_mitigation['none']['shortfall'] = []  
    for year_num in range(0, len(baseline_revenues)):
      no_buyout = True
      for folder_name in ['550', '600', '650', '700']:
        if no_buyout:
          if mitigation_dict[folder_name][year_num] > 0.0:
            exclusive_mitigation[folder_name]['buyout'].append(mitigation_dict[folder_name][year_num])
            exclusive_mitigation[folder_name]['shortfall'].append(baseline_revenues[year_num]/1000000.0)
            no_buyout = False
      if no_buyout:
        exclusive_mitigation['none']['buyout'].append(0.0)
        exclusive_mitigation['none']['shortfall'].append(baseline_revenues[year_num]/1000000.0)
        
    for folder_name, color_count in zip(['700', '650', '600', '550'], [3, 2, 1, 0]):
      self.ax.scatter(exclusive_mitigation[folder_name]['shortfall'], exclusive_mitigation[folder_name]['buyout'], c = color_list[color_count], s = 250, edgecolors = 'black', linewidths = 1.0, clip_on = False, zorder = 10)
    self.ax.scatter(exclusive_mitigation['none']['shortfall'], exclusive_mitigation['none']['buyout'], c = 'none', s = 250, edgecolors = 'black', linewidths = 2.0, zorder = 1)
    
    legend_location = 'upper left'
    legend_element = [Line2D([0], [0], color='white', marker = 'o', markerfacecolor = color_list[0], markeredgecolor = 'black', markersize = 25, label='550 tAF Threshold'),
                      Line2D([0], [0], color='white', marker = 'o', markerfacecolor = color_list[1], markeredgecolor = 'black', markersize = 25,  label='600 tAF Threshold'),
                      Line2D([0], [0], color='white', marker = 'o', markerfacecolor = color_list[2], markeredgecolor = 'black', markersize = 25,  label='650 tAF Threshold'),
                      Line2D([0], [0], color='white', marker = 'o', markerfacecolor = color_list[3], markeredgecolor = 'black', markersize = 25,  label='700 tAF Threshold'),
                      Line2D([0], [0], color='white', marker = 'o', markerfacecolor = 'white', markeredgecolor = 'black', markersize = 25,  label='No Buyout')]
    legend_properties = {'family':'Gill Sans MT','weight':'bold','size':20}
    self.ax.legend(handles=legend_element, loc=legend_location, prop=legend_properties)
    self.ax.set_xlabel('Revenue Shortfalls ($MM)', fontsize = 24, weight = 'bold', fontname = 'Gill Sans MT')
    self.ax.set_ylabel('Buyout Payments ($MM)', fontsize = 24, weight = 'bold', fontname = 'Gill Sans MT')
    self.ax.set_xlim([np.min(baseline_revenues) * 0.9/ 1000000.0, np.max(baseline_revenues) * 1.1/1000000.0])
    for item in (self.ax.get_xticklabels()):
      item.set_fontsize(20)
      item.set_fontname('Gill Sans MT')
    for item in (self.ax.get_yticklabels()):
      item.set_fontsize(20)
      item.set_fontname('Gill Sans MT')
    plt.savefig('Shapefiles_UCRB/' + self.figure_name, dpi = 300, bbox_inches = 'tight', pad_inches = 0.0)
    
  def plot_structure_changes_by_wyt(self, structures_objects, downstream_data, plot_type, axis_breaks, structure_use_list, name_use_list, purchase_transfers = 'none', buyout_transfers = 'none', purchase_list = 'none', buyouts_list = 'none', show_partners = 'all', iteration_no = 3):
    station_id_column_length = 12
    station_id_column_start = 0
    location_name = []
    label_name = []
    label_name_2 = {}
    total_diversions = 0.0
    irr_tp = 0.0
    mf_tp = 0.0
    muni_tp = 0.0
    min_change = 0.0
    max_change = 0.0
    color_list = sns.color_palette('rocket', 4)
    for x, y in zip(structure_use_list, name_use_list):
      label_name_2[x] = y
    
    for j in range(0,len(downstream_data)):
      if downstream_data[j][0] != '#':
        first_line = int(j * 1)
        break
    cumulative_change = 0.0
    counter1 = len(downstream_data)
    total_payments = np.zeros(len(downstream_data))
    for j in range(first_line, len(downstream_data)):
      station_id = str(downstream_data[j][station_id_column_start:station_id_column_length].strip())
      if station_id == '70_ADC050':
        label_name.append('Cameo')
        location_name.append(counter1)
      elif station_id == '5104055':
        label_name.append('Lake Granby')
        location_name.append(counter1)
      elif station_id == '5300584':
        label_name.append('Shoshone')
        location_name.append(counter1)
      if station_id in structures_objects:
        this_structure_filled = structures_objects[station_id].baseline_filled
        change_in_delivery = structures_objects[station_id].average_change[plot_type]
        printed_value = change_in_delivery * 1.0
          
        if this_structure_filled > 0.0:
          use_column = True
          if 'Irrigation' in structures_objects[station_id].structure_types:
            color_use = 'forestgreen'
          elif 'Minimum Flow' in structures_objects[station_id].structure_types:        
            color_use = 'steelblue'
          elif 'Export' in structures_objects[station_id].structure_types:
            color_use = 'indianred'
          elif 'Municipal' in structures_objects[station_id].structure_types:
            color_use = 'goldenrod'
          else:
            use_column = False
          if use_column:
            if station_id in structure_use_list:
              total_diversions += change_in_delivery
            elif station_id in purchase_list:
              skip_list = True
            elif color_use == 'forestgreen':
              irr_tp += change_in_delivery
            elif color_use == 'steelblue':
              mf_tp += change_in_delivery
            elif color_use == 'indianred' or color_use == 'goldenrod':
              muni_tp += change_in_delivery
            #for ab, ax_loc in zip([plot_type + '_1', plot_type + '_2'], [10.0, 6.25]):
              #if change_in_delivery - axis_breaks[ab] > ax_loc:
                #change_in_delivery -= axis_breaks[ab]
                #break
            if show_partners == 'all':  
              self.ax[1].fill_between([counter1-1, counter1], [0.0, 0.0], [this_structure_filled, this_structure_filled], color = color_use, alpha = 1.0, linewidth = 0.0)

              self.ax[0].fill_between([counter1-1, counter1], [0.0, 0.0], [change_in_delivery, change_in_delivery], color = color_use, alpha = 1.0, linewidth = 2.0)
              counter1 -= 1
              if station_id in structure_use_list:
                if change_in_delivery < 0.0:
                  self.ax[0].text(counter1 - 5, change_in_delivery + 2.5, label_name_2[station_id] + '\n' + str(int(printed_value)) + ' tAF', horizontalalignment='right', verticalalignment='top',fontsize = 20, weight = 'bold', fontname = 'Gill Sans MT')
                else:
                  self.ax[0].text(counter1 - 5, min(change_in_delivery + 1.5, 12.0), label_name_2[station_id] + '\n' + str(int(printed_value)) + ' tAF', horizontalalignment='right', verticalalignment='top',fontsize = 20, weight = 'bold', fontname = 'Gill Sans MT')

            elif show_partners == 'revenue':
              if station_id in purchase_list or station_id in structure_use_list:
                skipthis = True
                self.ax[2].fill_between([counter1-1, counter1], [0.0, 0.0], [this_structure_filled, this_structure_filled], color = color_use, alpha = 1.0, linewidth = 0.0)
              else:
                self.ax[2].fill_between([counter1-1, counter1], [0.0, 0.0], [this_structure_filled, this_structure_filled], color = color_use, alpha = 1.0, linewidth = 0.0)
                self.ax[0].fill_between([counter1-1, counter1], [0.0, 0.0], [change_in_delivery, change_in_delivery], color = color_list[iteration_no], alpha = 1.0, linewidth = 2.0)
                min_change = min(min_change, change_in_delivery)
                max_change = max(max_change, change_in_delivery)
                
                purchases = purchase_transfers[purchase_transfers['structure'] == station_id]
                buyouts = buyout_transfers[buyout_transfers['structure'] == station_id]
                buyouts = buyouts.drop_duplicates(subset = 'datetime')
                for index, row in purchases.iterrows():
                  for xxx in range(0, counter1):              
                    total_payments[xxx] += row['demand'] * row['consumptive'] * structures_objects[station_id].purchase_price / 1000000.0
                for index, row in buyouts.iterrows():               
                  for xxx in range(0, counter1):              
                    total_payments[xxx] += row['demand_purchase'] * 10.0 / 1000000.0
                #if change_in_delivery < -10.0 and iteration_no == 3:
                  #self.ax[0].text(counter1 - 6, change_in_delivery + 2.5, str(station_id) + '\n' + str(int(printed_value)) + ' tAF', horizontalalignment='right', verticalalignment='top',fontsize = 20, weight = 'bold', fontname = 'Gill Sans MT')
              counter1 -= 1
            elif station_id in purchase_list or station_id in buyouts_list:
              if show_partners == 'partners':
                self.ax[0].fill_between([counter1-1, counter1], [0.0, 0.0], [change_in_delivery, change_in_delivery], color = color_use, alpha = 1.0, linewidth = 2.0)
                self.ax[1].fill_between([counter1-1, counter1], [0.0, 0.0], [this_structure_filled, this_structure_filled], color = color_use, alpha = 1.0, linewidth = 0.0)
              
              if station_id in structure_use_list:
                self.ax[0].fill_between([counter1-1, counter1], [0.0, 0.0], [change_in_delivery, change_in_delivery], color = color_use, alpha = 1.0, linewidth = 2.0)
                self.ax[1].fill_between([counter1-1, counter1], [0.0, 0.0], [this_structure_filled, this_structure_filled], color = color_use, alpha = 1.0, linewidth = 0.0)
                if change_in_delivery < 0.0:
                  self.ax[0].text(counter1 - 6, change_in_delivery + 2.5, label_name_2[station_id] + '\n' + str(int(printed_value)) + ' tAF', horizontalalignment='right', verticalalignment='top',fontsize = 20, weight = 'bold', fontname = 'Gill Sans MT')
                else:
                  self.ax[0].text(counter1 - 6, min(change_in_delivery + 1.5, 12.0), label_name_2[station_id] + '\n' + str(int(printed_value)) + ' tAF', horizontalalignment='right', verticalalignment='top',fontsize = 20, weight = 'bold', fontname = 'Gill Sans MT')
              counter1 -= 1
            else:
              if show_partners == 'thirdparty':
                self.ax[0].fill_between([counter1-1, counter1], [0.0, 0.0], [change_in_delivery, change_in_delivery], color = color_use, alpha = 1.0, linewidth = 2.0)
                self.ax[1].fill_between([counter1-1, counter1], [0.0, 0.0], [this_structure_filled, this_structure_filled], color = color_use, alpha = 1.0, linewidth = 0.0)
              if station_id in structure_use_list:
                self.ax[0].fill_between([counter1-1, counter1], [0.0, 0.0], [change_in_delivery, change_in_delivery], color = color_use, alpha = 1.0, linewidth = 2.0)
                self.ax[1].fill_between([counter1-1, counter1], [0.0, 0.0], [this_structure_filled, this_structure_filled], color = color_use, alpha = 1.0, linewidth = 0.0)
                if change_in_delivery < 0.0:
                  self.ax[0].text(counter1 - 6, change_in_delivery + 2.5, label_name_2[station_id] + '\n' + str(int(printed_value)) + ' tAF', horizontalalignment='right', verticalalignment='top',fontsize = 20, weight = 'bold', fontname = 'Gill Sans MT')
                else:
                  self.ax[0].text(counter1 - 6, min(change_in_delivery + 1.5, 12.0), label_name_2[station_id] + '\n' + str(int(printed_value)) + ' tAF', horizontalalignment='right', verticalalignment='top',fontsize = 20, weight = 'bold', fontname = 'Gill Sans MT')
              counter1 -= 1
    if show_partners == 'revenue':
      self.ax[1].fill_between(np.arange(len(downstream_data)),np.zeros(len(downstream_data)), total_payments, facecolor = 'beige', alpha = 1.0, linewidth = 2.0)
      self.ax[1].fill_between(np.arange(len(downstream_data)),np.zeros(len(downstream_data)), total_payments, facecolor = color_list[iteration_no], edgecolor = 'black', alpha = 0.8, linewidth = 2.5)
      self.ax[0].text((len(downstream_data) - counter1) * 0.66 + counter1, -40, 'Minimum Flow\nColorado River', horizontalalignment='center', verticalalignment='center',fontsize = 20, weight = 'bold', fontname = 'Gill Sans MT')
      
    for l_name, label_loc in zip(label_name, location_name):
      self.ax[2].scatter([label_loc,], [0.0,], c = 'black', s = 25)
      self.ax[2].text(label_loc, -0.1, l_name, horizontalalignment='center', verticalalignment='center', fontsize = 20, weight = 'bold', fontname = 'Gill Sans MT')
    for x in range(0, 3):
      self.ax[x].set_xticks([])
      self.ax[x].set_xticklabels('')
      self.ax[x].set_xlim([counter1, len(downstream_data)])
    self.ax[0].set_ylabel('Third-Party\nImpacts\n(tAF)', fontsize = 24, weight = 'bold', fontname = 'Gill Sans MT')
    if show_partners == 'revenue':
      self.ax[1].set_ylabel('Buyout\nPayments\n($MM)', fontsize = 24, weight = 'bold', fontname = 'Gill Sans MT')
      if iteration_no == 3:
        self.ax[1].set_ylim([0.0, np.max(total_payments) * 1.1])
        self.ax[0].set_ylim([min_change * 1.1, max_change * 1.1])
      self.ax[2].set_ylabel('In-Priority\nDemands (%)', fontsize = 24, weight = 'bold', fontname = 'Gill Sans MT')
      self.ax[2].set_ylim([0.0, 1.0])
      self.ax[2].yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1.0, decimals = None))
    else:
      self.ax[1].set_ylabel('In-Priority\nDemands (%)', fontsize = 24, weight = 'bold', fontname = 'Gill Sans MT')
      self.ax[1].set_ylim([0.0, 1.0])
      self.ax[1].yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1.0, decimals = None))
      self.ax[0].set_yticks([-30.0, -20.0, -10.0, 0.0, 10.0])
      self.ax[0].set_ylim([-32.0, 12.0])
    legend_location = 'lower right'
    legend_element = [Patch(facecolor='forestgreen', edgecolor='black', label='Irrigation'),
                     Patch(facecolor='steelblue', edgecolor='black', label='Min Flow'),
                     Patch(facecolor='indianred', edgecolor='black', label='Export'),
                     Patch(facecolor='goldenrod', edgecolor='black', label='M&I')]
    legend_location2 = 'upper right'
    legend_element2 = [Patch(facecolor=color_list[0], edgecolor='black', label='550 tAF Threshold'),
                     Patch(facecolor=color_list[1], edgecolor='black', label='600 tAF Threshold'),
                     Patch(facecolor=color_list[2], edgecolor='black', label='650 tAF Threshold'),
                     Patch(facecolor=color_list[3], edgecolor='black', label='700 tAF Threshold')]
    legend_properties = {'family':'Gill Sans MT','weight':'bold','size':20}
    self.ax[2].legend(handles=legend_element, loc=legend_location, prop=legend_properties, ncol = 4)
    self.ax[1].legend(handles=legend_element2, loc=legend_location2, prop=legend_properties, ncol = 2)
    self.ax[0].legend(handles=legend_element2, loc='lower left', prop=legend_properties, ncol = 2)
    self.ax[0].plot([0, counter1], [0.0, 0.0], color = 'black', linewidth = 0.5)
    #self.ax[0].set_yticklabels(['-5.0', '-2.5', '0.0', '2.5', str(min(axis_breaks[plot_type + '_1'], axis_breaks[plot_type + '_2']) + 6.25), str(max(axis_breaks[plot_type + '_1'], axis_breaks[plot_type + '_2']) + 10.)])
    tradeoff_df = pd.DataFrame(np.asarray([total_diversions, irr_tp, mf_tp, muni_tp]))
    tradeoff_df.to_csv('all_changes_' + str(iteration_no) + '.csv')
    for x in range(0,3):
      for item in (self.ax[x].get_yticklabels()):
        item.set_fontsize(18)
    if iteration_no == 0:
      plt.savefig('Shapefiles_UCRB/' + self.figure_name, dpi = 300, bbox_inches = 'tight', pad_inches = 0.0)

  def plot_simple_risk(self, animation_pane):
    export_diversions = pd.read_csv('input_files/total_export_deliveries.csv')
    total_exports = np.asarray(export_diversions['3604684_baseline'] + export_diversions['5104655_baseline'])
    if animation_pane < 2:
      min_val = np.min(total_exports)
      max_val = np.max(total_exports)
      range_val = max_val - min_val
      pos = np.linspace(min_val, max_val, 101)
      kde_export = stats.gaussian_kde(total_exports)
      pos2 = np.linspace(min_val, 110.0, 101)
      pos3 = np.linspace(110.0, max_val, 101)
      total_shortfall_years = np.sum(total_exports < 110.0) / float(len(total_exports))
      print(total_shortfall_years)
    if animation_pane >= 2:
      financial_risk = np.zeros(len(total_exports))
      for x in range(0, len(total_exports)):
        financial_risk[x] = (max(110.0 - total_exports[x], 0.0) * (1.612 - .38))
      financial_risk = np.asarray(financial_risk)
      min_val = np.min(financial_risk)
      max_val = np.max(financial_risk)
      range_val = max_val - min_val
      if animation_pane == 2:
        pos = np.linspace(min_val, max_val, 101)
        kde_export = stats.gaussian_kde(financial_risk)
      elif animation_pane == 3:
        sorted_risk = np.sort(financial_risk)
        five_percent = sorted_risk[len(sorted_risk) - int(0.05 * len(sorted_risk))]
        pos = np.linspace(min_val, max_val, 101)
        pos2 = np.linspace(min_val, five_percent, 101)
        pos3 = np.linspace(five_percent, max_val, 101)
        kde_export = stats.gaussian_kde(financial_risk)
        
    self.ax.fill_between(pos, kde_export(pos), edgecolor = 'black', linewidth = 3.0, alpha = 1.0, facecolor = 'beige')
    if animation_pane == 0:
      self.ax.fill_between(pos, kde_export(pos), edgecolor = 'black', linewidth = 3.0, alpha = 1.0, facecolor = 'steelblue')
      y_lab = 'Annual Transbasin Diversions, Denver Water (tAF)'
    elif animation_pane == 1:
      self.ax.fill_between(pos2, kde_export(pos2), edgecolor = 'black', linewidth = 3.0, alpha = 1.0, facecolor = 'indianred')
      self.ax.fill_between(pos3, kde_export(pos3), edgecolor = 'black', linewidth = 3.0, alpha = 1.0, facecolor = 'steelblue')
      y_lab = 'Annual Transbasin Diversions, Denver Water (tAF)'
    elif animation_pane == 2:
      self.ax.fill_between(pos, kde_export(pos), edgecolor = 'black', linewidth = 3.0, alpha = 1.0, facecolor = 'steelblue')
      y_lab = 'Lost Revenue, Denver Water ($M)'
    elif animation_pane == 3:
      self.ax.fill_between(pos2, kde_export(pos2), edgecolor = 'black', linewidth = 3.0, alpha = 1.0, facecolor = 'steelblue')
      self.ax.fill_between(pos3, kde_export(pos3), edgecolor = 'black', linewidth = 3.0, alpha = 1.0, facecolor = 'indianred')
      self.ax.plot([five_percent, five_percent], [0.0, np.max(kde_export(pos))], linewidth = 3.0, linestyle = '--', color = 'black')
      print(five_percent)
      y_lab = 'Lost Revenue, Denver Water ($M)'
    

    self.ax.set_yticks([])
    self.ax.set_yticklabels('')
    self.ax.set_ylabel('Probability', fontsize = 32, weight = 'bold', fontname = 'Gill Sans MT')
    self.ax.set_xlabel(y_lab, fontsize = 32, weight = 'bold', fontname = 'Gill Sans MT')
    self.ax.set_xlim([min_val, max_val])
    self.ax.set_ylim([0.0,np.max(kde_export(pos))])
    for item in (self.ax.get_xticklabels() + self.ax.get_xticklabels()):
      item.set_fontsize(28)
      item.set_fontname('Gill Sans MT')
    if animation_pane == 1:
      legend_location = 'upper left'
      legend_element = [Patch(facecolor='steelblue', edgecolor='black', label='No Shortfall Years'),
                     Patch(facecolor='indianred', edgecolor='black', label='Shortfall Years')]
      legend_properties = {'family':'Gill Sans MT','weight':'bold','size':28}
      self.ax.legend(handles=legend_element, loc=legend_location, prop=legend_properties)
    if animation_pane == 3:
      legend_location = 'upper right'
      legend_element = [Patch(facecolor='indianred', edgecolor='black', label='Worst 5% Years')]
      legend_properties = {'family':'Gill Sans MT','weight':'bold','size':28}
      self.ax.legend(handles=legend_element, loc=legend_location, prop=legend_properties)
    
    plt.savefig('Shapefiles_UCRB/' + self.figure_name, dpi = 300, bbox_inches = 'tight', pad_inches = 0.0)

  def plot_cost_per_af(self, cost_per_af, total_exports, iteration_no, color_map, xlimit):
    #rpp = [22.12, 20.25, 10.0 ,32.5, 46.5, 81.0, 126.0, 160.5]
    #year_rpp = [2010, 2011, 2012, 2015, 2016, 2017, 2018, 2019]
    rpp = pd.read_csv('input_files/northern_regional_pool.csv')
    color_list = sns.color_palette(color_map, 4)
    volume_weighted_dist = []
    min_val = np.min(cost_per_af)
    max_val = np.max(cost_per_af)
    pos = np.linspace(np.min(cost_per_af), min(np.max(cost_per_af),xlimit), 101)
    for cost_af, ex_inc in zip(cost_per_af, total_exports):
      for x in range(0, int(ex_inc/100.0)):
        volume_weighted_dist.append(cost_af)
    volume_weighted_dist = np.asarray(volume_weighted_dist)
    kde_weighted_cost = stats.gaussian_kde(volume_weighted_dist)
    self.ax.fill_between(pos, kde_weighted_cost(pos), edgecolor = 'black', linewidth = 3.0, alpha = 0.9, facecolor = color_list[iteration_no])
    
#    for yearnum in range(2010, 2020):
#      this_year_rpp = rpp[rpp['year'] == yearnum]
#      if len(this_year_rpp) > 1:
#        min_val = np.min(this_year_rpp['bid'])
#        max_val = np.max(this_year_rpp['bid'])
#        for index_rpp, row_rpp in this_year_rpp.iterrows():
#          self.ax.plot([row_rpp['bid'], row_rpp['bid']], [0.0, np.max(kde_weighted_cost(pos))* 1.1], linewidth = 3.0, color = color_list[yearnum - 2010])
    
    self.ax.set_yticks([])
    self.ax.set_yticklabels('')
    self.ax.set_ylabel('Probability', fontsize = 32, weight = 'bold', fontname = 'Gill Sans MT')
    self.ax.set_xlabel('Transfer Price ($/AF)', fontsize = 32, weight = 'bold', fontname = 'Gill Sans MT')
    legend_location2 = 'upper right'
    if color_map == 'rocket':
      legend_element2 = [Patch(facecolor=color_list[0], edgecolor='black', label='550 tAF Threshold'),
                       Patch(facecolor=color_list[1], edgecolor='black', label='600 tAF Threshold'),
                       Patch(facecolor=color_list[2], edgecolor='black', label='650 tAF Threshold'),
                       Patch(facecolor=color_list[3], edgecolor='black', label='700 tAF Threshold')]
    else:
      legend_element2 = [Patch(facecolor=color_list[0], edgecolor='black', label='Lease 1.5x, Buyout $20/AF'),
                       Patch(facecolor=color_list[1], edgecolor='black', label='Lease 2.5x, Buyout $20/AF'),
                       Patch(facecolor=color_list[2], edgecolor='black', label='Lease 1.5x, Buyout $40/AF'),
                       Patch(facecolor=color_list[3], edgecolor='black', label='Lease 2.5x, Buyout $40/AF')]

    legend_properties = {'family':'Gill Sans MT','weight':'bold','size':24}
    self.ax.legend(handles=legend_element2, loc=legend_location2, prop=legend_properties, ncol = 1)
    
 #   self.fig.subplots_adjust(right=0.9)
 #   cbar_ax = self.fig.add_axes([0.9187, 0.15, 0.025, 0.7])
 #   sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=plt.Normalize(vmin=2010, vmax=2019))
 #   clb1 = plt.colorbar(sm, cax = cbar_ax, ticks=[2010, 2013, 2016, 2019])
 #   clb1.ax.set_yticklabels(['2010', '2013', '2016', '2019']) 
 #   clb1.ax.invert_yaxis()
 #   clb1.ax.tick_params(labelsize=20)
 #  clb1.ax.set_ylabel('Regional Pool Bids', rotation=90, fontsize = 24, fontname = 'Gill Sans MT', fontweight = 'bold', labelpad = 15)
 #   for item in clb1.ax.yaxis.get_ticklabels():
 #     item.set_fontname('Gill Sans MT')  
 #     item.set_fontsize(20)
    self.ax.set_xlim([0.0, xlimit])
    
    for item in (self.ax.get_xticklabels() + self.ax.get_yticklabels()):
      item.set_fontsize(20)
      item.set_fontname('Gill Sans MT')
    if color_map == 'rocket':
      if iteration_no == 3:
        self.ax.set_ylim([0.0, np.max(kde_weighted_cost(pos)) * 1.1])
      if iteration_no == 1:
        plt.savefig('Shapefiles_UCRB/' + self.figure_name, dpi = 300, bbox_inches = 'tight', pad_inches = 0.0)
    else:
      if iteration_no == 0:
        self.ax.set_ylim([0.0, np.max(kde_weighted_cost(pos)) * 1.1])
      if iteration_no == 3:
        plt.savefig('Shapefiles_UCRB/' + self.figure_name, dpi = 300, bbox_inches = 'tight', pad_inches = 0.0)
        
  def plot_structure_changes(self, percent_filled, downstream_data, buyout_list, purchase_list):
    
    sorted_changes = percent_filled.sort_values(by=['pct_filled_baseline'])
    #self.ax.fill_between(np.arange(len(sorted_changes['pct_filled_baseline'])), np.zeros(len(sorted_changes['pct_filled_baseline'])), sorted_changes['pct_filled_baseline'], color = 'black', alpha = 0.4) 
    counter_deliveries = 0.0
    #for index, row in sorted_changes.iterrows():
      #if row['change'] > 0.0:
        #if row['structure_name'] == '5104634':
          #self.ax.fill_between([counter_deliveries, counter_deliveries + 1], np.zeros(2), [row['change'], row['change']], color = 'forestgreen', alpha = 1.0, linewidth = 0.0)
          #counter_deliveries += 1
      #elif row['change'] < 0.0:
        #if row['structure_name'] == '5104634':
          #self.ax.fill_between([counter_deliveries, counter_deliveries + 1], [row['change'], row['change']], np.zeros(2), color = 'forestgreen', alpha = 1.0, linewidth = 0.0)
          #counter_deliveries += 1
    counter1 = 0
    station_id_column_length = 12
    station_id_column_start = 0
    location_name = []
    label_name = []
    for j in range(0,len(downstream_data)):
      if downstream_data[j][0] != '#':
        first_line = int(j * 1)
        break
    cumulative_change = 0.0
    for j in range(first_line, len(downstream_data)):
      station_id = str(downstream_data[j][station_id_column_start:station_id_column_length].strip())
      if station_id == '70_ADC050':
        label_name.append('Cameo')
        location_name.append(counter1)
      elif station_id == '5104055':
        label_name.append('Lake Granby')
        location_name.append(counter1)
      elif station_id == '5300584':
        label_name.append('Shoshone')
        location_name.append(counter1)
      
      this_structure_filled = percent_filled[percent_filled['structure_name'] == station_id]
      for index, row in this_structure_filled.iterrows():
        use_column = True
        if row['structure_types'] == 'Irrigation':
          color_use = 'forestgreen'
        elif row['structure_types'] == 'Minimum Flow':        
          color_use = 'steelblue'
        elif row['structure_types'] == 'Export':
          color_use = 'indianred'
        elif row['structure_types'] == 'Municipal':
          color_use = 'goldenrod'
        else:
          use_column = False
        if use_column:
          if row['pct_filled_baseline'] > 0.0:
            print(counter1, end = " ")
            print(row['structure_name'], end = " ")
            print(row['structure_types'], end = " ")
            print(row['pct_filled_baseline'], end =  " ")            
            self.ax[0].fill_between([0.0, row['pct_filled_baseline']], [counter1, counter1], [counter1 + 1, counter1 + 1], color = color_use, alpha = 1.0, linewidth = 0.0)
            n_count_1 = 0
            n_count_2 = 0
            n_count_3 = 0
            p_count_1 = 0
            p_count_2 = 0
            p_count_3 = 0
            for x in range(0, 10):
              try:
                print(int(row['change_' + str(x)]/100.0)/10.0, end = " ")
              except:
                pass
              try:
                if row['cum_change_pos' + str(x)] > 0.0:
                  self.ax[2].fill_between([0.0, row['cum_change_pos' + str(x)]/1000.0], [counter1, counter1], [counter1 + 1, counter1 + 1], color = 'beige', alpha = 1.0, linewidth = 0.0)
                  self.ax[2].fill_between([0.0, row['cum_change_pos' + str(x)]/1000.0], [counter1, counter1], [counter1 + 1, counter1 + 1], color = 'sienna', alpha = max(min(float(p_count_1+1) / 10.0, 1.0), 0.0), linewidth = 0.0)
                  p_count_1 += 1
              except:
                pass
              try:
                if row['cum_change_neg' + str(x)] < 0.0:
                  self.ax[2].fill_between([row['cum_change_neg' + str(x)]/1000.0, 0.0], [counter1, counter1], [counter1 + 1, counter1 + 1], color = 'beige', alpha = 1.0, linewidth = 0.0)
                  self.ax[2].fill_between([row['cum_change_neg' + str(x)]/1000.0, 0.0], [counter1, counter1], [counter1 + 1, counter1 + 1], color = 'sienna', alpha = max(min(float(n_count_1+1) / 10.0, 1.0), 0.0), linewidth = 0.0)
                  n_count_1 += 1
              except:
                pass
              try:
                if row['change_' + str(x)] < 0.0:
                  self.ax[1].fill_between([row['change_' + str(x)]/1000.0, 0.0], [counter1, counter1], [counter1 + 1, counter1 + 1], color = 'beige', alpha = 1.0)
                  self.ax[1].fill_between([row['change_' + str(x)]/1000.0, 0.0], [counter1, counter1], [counter1 + 1, counter1 + 1], color = color_use, alpha = max(min(float(n_count_2+1) / 10.0, 1.0), 0.0))
                  n_count_2 += 1
              except:
                pass
              try:
                if row['revenue_' + str(x)] < 0.0:
                  self.ax[3].fill_between([row['revenue_' + str(x)]/1000000.0, 0.0], [counter1, counter1], [counter1 + 1, counter1 + 1], color = 'beige', alpha = 1.0)
                  self.ax[3].fill_between([row['revenue_' + str(x)]/1000000.0, 0.0], [counter1, counter1], [counter1 + 1, counter1 + 1], color = color_use, alpha = max(min(float(n_count_3+1) / 10.0, 1.0), 0.0))
                  n_count_3 += 1
              except:
                pass
            for x in range(9, -1, -1):
              try:                 
                if row['change_' + str(x)] > 0.0:
                  self.ax[1].fill_between([0.0, row['change_' + str(x)]/1000.0], [counter1, counter1], [counter1 + 1, counter1 + 1], color = 'beige', alpha = 1.0, linewidth = 0.0)
                  self.ax[1].fill_between([0.0, row['change_' + str(x)]/1000.0], [counter1, counter1], [counter1 + 1, counter1 + 1], color = color_use, alpha = max(min(float(p_count_2+1) / 10.0, 1.0), 0.0))
                  p_count_2 += 1
              except:
                pass
              try:                 
                if row['revenue_' + str(x)] > 0.0:
                  if row['revenue_' + str(x)] > 25000000:
                    value_use = (row['revenue_' + str(x)] - 50000000)/1000000.0
                  else:
                    value_use = row['revenue_' + str(x)]/1000000.0
                  
                  self.ax[3].fill_between([0.0, value_use], [counter1, counter1], [counter1 + 1, counter1 + 1], color = 'beige', alpha = 1.0, linewidth = 0.0)
                  self.ax[3].fill_between([0.0, value_use], [counter1, counter1], [counter1 + 1, counter1 + 1], color = color_use, alpha = max(min(float(p_count_3+1) / 10.0, 1.0), 0.0))
                  p_count_3 += 1
              except:
                pass
            counter1 += 1
            print()
    for l_name, label_loc in zip(label_name, location_name):
      self.ax[0].scatter([0.0,], [label_loc,], c = 'black', s = 25)
      self.ax[0].text(-0.05, label_loc,l_name, horizontalalignment='right', verticalalignment='center', fontsize = 20, weight = 'bold', fontname = 'Gill Sans MT')
    for x in range(0, 4):
      self.ax[x].set_yticks([])
      self.ax[x].set_yticklabels('')
      self.ax[x].set_ylim([0, counter1])
    self.ax[0].set_xlabel('Percent Historical\nDemands Met', fontsize = 24, weight = 'bold', fontname = 'Gill Sans MT')
    self.ax[1].set_xlabel('Structure Diversion\nChange (tAF)', fontsize = 24, weight = 'bold', fontname = 'Gill Sans MT')
    self.ax[2].set_xlabel('Cumulative Diversion\nChange (tAF)', fontsize = 24, weight = 'bold', fontname = 'Gill Sans MT')
    self.ax[3].set_xlabel('Revenue Change\n($MM)', fontsize = 24, weight = 'bold', fontname = 'Gill Sans MT')
    #self.ax[0][0].set_ylim([-100000.0, 200000.0])
    #self.ax[1][0].set_ylim([-100000.0, 200000.0])
    #self.ax[0][1].set_ylim([-100000.0, 200000.0])
    #self.ax[1][1].set_ylim([-100000.0, 200000.0])
    
    #self.ax[0][0].set_xticks([])
    #self.ax[0][0].set_xticklabels('')
    #self.ax[0][1].set_xticks([])
    #self.ax[0][1].set_xticklabels('')
    #self.ax[1][1].set_yticks([])
    #self.ax[1][1].set_yticklabels('')
    #self.ax[0][1].set_yticks([])
    #self.ax[0][1].set_yticklabels('')
    
    #self.ax[0][0].set_ylabel('Average Annual Revenue Change ($/AF)', fontsize = 24, weight = 'bold', fontname = 'Gill Sans MT')
    #self.ax[1][0].set_ylabel('Average Annual Revenue Change ($/AF)', fontsize = 24, weight = 'bold', fontname = 'Gill Sans MT')
    #self.ax[1][0].set_xlabel('Average Annual Change\nin Water Use (AF/year)', fontsize = 24, weight = 'bold', fontname = 'Gill Sans MT')
    #self.ax[1][1].set_xlabel('Average Annual Change\nin Water Use (AF/year)', fontsize = 24, weight = 'bold', fontname = 'Gill Sans MT')
    legend_location = 'lower right'
    #legend_location_alt = 'upper right'
    legend_element = [Patch(facecolor='forestgreen', edgecolor='black', label='Irrigation'),
                     Patch(facecolor='steelblue', edgecolor='black', label='Min Flow'),
                     Patch(facecolor='indianred', edgecolor='black', label='Export'),
                     Patch(facecolor='goldenrod', edgecolor='black', label='M&I')]
    #legend_element2 = [Patch(facecolor='indianred', edgecolor='black', label='Purchase Partners')]
    #legend_element3 = [Patch(facecolor='steelblue', edgecolor='black', label='Buyout Partners')]
    #legend_element4 = [Patch(facecolor='black', edgecolor='black', label='Uninvolved Parties')]
    legend_properties = {'family':'Gill Sans MT','weight':'bold','size':14}
    self.ax[3].legend(handles=legend_element, loc=legend_location, prop=legend_properties)
    
    legend_location2 = 'lower left'
    #legend_location_alt = 'upper right'
    legend_element2 = [Patch(facecolor='sienna', edgecolor='black', alpha = 0.1, label='2% Prob.'),
                     Patch(facecolor='sienna', edgecolor='black', alpha = 0.3, label='6% Prob.'),
                     Patch(facecolor='sienna', edgecolor='black', alpha = 0.5, label='10% Prob.'),
                     Patch(facecolor='sienna', edgecolor='black', alpha = 1.0, label='20% Prob.')]
    #legend_element2 = [Patch(facecolor='indianred', edgecolor='black', label='Purchase Partners')]
    #legend_element3 = [Patch(facecolor='steelblue', edgecolor='black', label='Buyout Partners')]
    #legend_element4 = [Patch(facecolor='black', edgecolor='black', label='Uninvolved Parties')]
    legend_properties = {'family':'Gill Sans MT','weight':'bold','size':14}
    self.ax[2].legend(handles=legend_element2, loc=legend_location2, prop=legend_properties)
    for x in range(1, 4):
      self.ax[x].plot([0.0, 0.0], [0, counter1], color = 'black', linewidth = 0.5)
    #self.ax[0][1].legend(handles=legend_element2, loc=legend_location, prop=legend_properties)
    #self.ax[1][0].legend(handles=legend_element3, loc=legend_location, prop=legend_properties)
    #self.ax[1][1].legend(handles=legend_element4, loc=legend_location, prop=legend_properties)
    #for x in range(0,2):
      #for y in range(0,2):
        #self.ax[x][y].plot(np.zeros(2), [-100000.0, 200000.0], color = 'black', linestyle = '--', linewidth = 2.0)
        #self.ax[x][y].plot([-250000.0, 500000.0], np.zeros(2), color = 'black', linestyle = '--', linewidth = 2.0)
        #self.ax[x][y].set_xlim([-1500, 500])
        #for item in (self.ax[x][y].get_xticklabels()):
          #item.set_fontsize(16)
        #for item in (self.ax[x][y].get_yticklabels()):
          #item.set_fontsize(16)

    plt.savefig('Shapefiles_UCRB/' + self.figure_name, dpi = 300, bbox_inches = 'tight', pad_inches = 0.0)
    plt.show()
    
  def plot_informal_purchases(self, station_id):
    purchased_water = pd.read_csv('results/change_points1_' + station_id +'.csv', index_col = 0)
    purchased_water = purchased_water.drop_duplicates()
    buyout_water = pd.read_csv('results/change_points2_' + station_id +'.csv', index_col = 0)
    buyout_water = buyout_water.drop_duplicates()
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
      if row['month'] < 3:
        date_val = datetime(row['year'], row['month'] + 10, 1, 0, 0)
      else:
        date_val = datetime(row['year'], row['month'] - 2, 1, 0, 0)
      if date_val > prev_date:
        informal_transfer.loc[prev_date, 'purchased'] = total_date_purchase
        prev_demand = 0.0
        total_date_purchase = 0.0
        if row['month'] < 3:
          prev_date = datetime(row['year'], row['month'] + 10, 1, 0, 0)
        else:
          prev_date = datetime(row['year'], row['month'] - 2, 1, 0, 0)

      if row['demand'] > prev_demand:
        if row['structure'] in structure_purchase_list:
          structure_purchase_list[row['structure']].append(row['delivery'] - prev_demand)
        else:
          structure_purchase_list[row['structure']] = []
          structure_purchase_list[row['structure']].append(row['delivery'] - prev_demand)
        total_date_purchase += row['delivery']
        prev_demand = row['demand'] * 1.0

    total_date_buyout = 0.0
    prev_date = datetime(1950, 1, 1, 0, 0)
    for index, row in buyout_water.iterrows():
      if row['month'] < 3:
        date_val = datetime(row['year'], row['month'] + 10, 1, 0, 0)
      else:
        date_val = datetime(row['year'], row['month'] - 2, 1, 0, 0)
      
      if date_val > prev_date:
        informal_transfer.loc[prev_date, 'buyouts'] = total_date_buyout
        prev_demand = 0.0
        total_date_buyout = 0.0
        if row['month'] < 3:
          prev_date = datetime(row['year'], row['month'] + 10, 1, 0, 0)
        else:
          prev_date = datetime(row['year'], row['month'] - 2, 1, 0, 0)

      if row['demand'] > 0.0:
        if row['structure'] in structure_buyout_list:
          structure_buyout_list[row['structure']].append(row['demand'])
        else:
          structure_buyout_list[row['structure']] = []
          structure_buyout_list[row['structure']].append(row['demand'])
        total_date_buyout += row['delivery']
    informal_transfer.to_csv('results/timeseries_informal_transfer.csv')
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
  def plot_cost_distribution(self, res_station, simulated_values, simulated_values2, informal_transfer):
    cost_list = []
    total_volume = []
    for index, row in informal_transfer.iterrows():
      total_cost = row['buyouts'] * 25.0 + row['purchased'] * 200.0
      total_benefit = simulated_values2.loc[index, res_station + '_diversions'] - simulated_values.loc[index, res_station + '_diversions']
      if index.year > 2012:
        break
      if total_benefit > 0.0 and pd.notna(total_benefit) and total_cost > 0.0:
        print(total_benefit, end = " ")
        print(total_cost, end = " ")
        print(float(total_cost/total_benefit))
        for xx in range(0, int(total_benefit/100.0)):
          cost_list.append(float(total_cost/total_benefit))
        total_volume.append(total_benefit)
    #cost_list = np.asarray(cost_list)
    #total_volume = np.asarray(total_volume)
    #cumulative_volume = np.zeros(len(total_volume))
    #sorted_order = np.argsort(cost_list)
    #sorted_cost = cost_list[sorted_order]
    #sorted_volume = total_volume[sorted_order]
    #cv = 0.0
    #for x in range(0, len(total_volume)):
      #cv += sorted_volume[x]
      #cumulative_volume[x] = cv * 1.0
    
    pos = np.linspace(0, 2000.0, 101)
    kde_est = stats.gaussian_kde(cost_list)
    self.ax.set_xlim([0, 2000.0])
    self.ax.set_ylim([0, 0.0008])

    self.ax.fill_between(pos, kde_est(pos), edgecolor = 'black', alpha = 0.6, facecolor = 'steelblue')
    #self.ax.fill_between(pos[:25], kde_est(pos[:25]), edgecolor = 'black', alpha = 1.0, facecolor = 'indianred')
    self.ax.set_yticks([])
    self.ax.set_yticklabels('')
    self.ax.set_xlabel('Total Informal Transfer Cost ($/AF)', fontsize = 20, weight = 'bold', fontname = 'Gill Sans MT')
    
    for item in (self.ax.get_xticklabels()):
      item.set_fontsize(16)
      item.set_fontname('Gill Sans MT')
    
    plt.savefig(self.figure_name, dpi = 300, bbox_inches = 'tight', pad_inches = 0.0)
    plt.close()
    
    
  def plot_reservoir_figures(self, historical_values, simulated_values, simulated_values2, reservoir_names,informal_transfer):
    counter = 0
      #self.ax[axis_plot].fill_between(informal_transfer.index, np.zeros(len(informal_transfer.index)), informal_transfer['purchased'] / 1000.0, color = 'darkorange', alpha = 1.0)
      #self.ax[axis_plot].fill_between(informal_transfer.index, informal_transfer['purchased'] / 1000.0, (informal_transfer['buyouts'] + informal_transfer['purchased'])  / 1000.0, color = 'indianred', alpha = 1.0)
    self.ax.bar(informal_transfer.index, (informal_transfer['buyouts'] + informal_transfer['purchased'])  / 1000.0, width = 15.0, color = 'indianred', alpha = 1.0)
    self.ax.bar(informal_transfer.index, informal_transfer['purchased'] / 1000.0, width = 15.0, color = 'darkorange', alpha = 1.0)
    for x, res_name in zip(historical_values, [reservoir_names,]):
      self.ax.bar(simulated_values2.index, (simulated_values2[x + '_diversions'] - simulated_values[x + '_diversions'])/1000.0, width = 15.0, color = 'teal', alpha = 1.0)
    self.ax.set_ylabel('Informal Transfers (tAF)', fontsize = 20, weight = 'bold', fontname = 'Gill Sans MT')
    for item in (self.ax.get_xticklabels()):
      item.set_fontsize(16)
      item.set_fontname('Gill Sans MT')
    for item in (self.ax.get_yticklabels()):
      item.set_fontsize(16)
      item.set_fontname('Gill Sans MT')

    self.ax.set_xlim([datetime(1977, 3, 15, 0, 0), datetime(1977, 8, 15, 0, 0)])
    self.ax.set_ylim([0, 100])
     
    
    
    #self.ax.legend(['Reservoir Storage', 'Snowpack Storage'])
    legend_elements = [Patch(facecolor='teal', edgecolor='black', label='Additional Transbasin Diversions', alpha = 0.7), 
                      Patch(facecolor='darkorange', edgecolor='black', label='Informal Transfer Purchases', alpha = 0.7), 
                      Patch(facecolor='indianred', edgecolor='black', label='Informal Transfer Buyouts', alpha = 0.7)]
                      
    self.ax.legend(handles=legend_elements, loc='upper left', prop={'family':'Gill Sans MT','weight':'bold','size':14}, framealpha = 1.0, ncol = 2)      
    self.ax.xaxis.set_major_locator(mdates.MonthLocator())
    self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    plt.savefig(self.figure_name, dpi = 300, bbox_inches = 'tight', pad_inches = 0.0)
    plt.close()
    
  def plot_reservoir_simulation(self, simulated_values, total_diversions, res_use, div_use, account_plot_order):
    counter = 0
    qalycolors = sns.color_palette('gnuplot_r', 5)
    total_storage = np.zeros(len(simulated_values[res_use]))
    for account_num in account_plot_order:
      self.ax[0].fill_between(simulated_values.index, total_storage, total_storage + simulated_values[res_use + '_account_' + account_num]/1000.0, color = qalycolors[int(account_num)], alpha = 0.7, linewidth = 0.0)
      total_storage += simulated_values[res_use + '_account_' + account_num]/1000.0
    self.ax[1].plot(simulated_values.index, simulated_values[res_use + '_account_1']/1000.0, color = 'steelblue', zorder = 1)
    
    total_div_sum = 0.0
    annual_index = []
    annual_diversions = []
    for x in range(0, len(total_diversions.index)):
      if total_diversions.index[x].month == 10:
        total_div_sum = total_diversions.loc[total_diversions.index[x], div_use]
      else:
        total_div_sum += total_diversions.loc[total_diversions.index[x], div_use]
      if total_diversions.index[x].month == 9:
        annual_index.append(total_diversions.index[x])
        annual_diversions.append(total_div_sum)
        
    annual_diversions = np.asarray(annual_diversions)
    annual_index = np.asarray(annual_index)
    self.ax[1].bar(annual_index, annual_diversions/1000.0, width=np.timedelta64(180, 'D'), color = 'indianred', alpha = 0.7, zorder = 2)
    recent_years = annual_index > datetime(1949, 10, 1, 0, 0)
    annual_average = np.mean(annual_diversions[recent_years])/1000.0
    self.ax[1].plot([datetime(1949, 10, 1, 0, 0), simulated_values.index[-1]], [annual_average, annual_average], linewidth = 2.0, color = 'indianred')
    self.ax[0].set_xlim([datetime(1949, 10, 1, 0, 0), simulated_values.index[-1]])
    self.ax[1].set_xlim([datetime(1949, 10, 1, 0, 0), simulated_values.index[-1]])
    self.ax[0].set_ylim([0, 650])
    self.ax[1].set_ylim([0, 550])
    self.ax[0].set_ylabel('Account Storage\n(tAF)', fontsize = 14, weight = 'bold', fontname = 'Gill Sans MT')
    self.ax[1].set_ylabel('CBT Supplies\n(tAF)', fontsize = 14, weight = 'bold', fontname = 'Gill Sans MT')
    legend_elements = [Patch(facecolor=qalycolors[1], edgecolor='black', label='CBT', alpha = 0.7), 
                      Patch(facecolor=qalycolors[2], edgecolor='black', label='Dead Pool', alpha = 0.7), 
                      Patch(facecolor=qalycolors[3], edgecolor='black', label='Instream Flow', alpha = 0.7)]
    self.ax[0].legend(handles=legend_elements, loc='upper right', prop={'family':'Gill Sans MT','weight':'bold','size':8}, framealpha = 1.0, ncol = 3)
    legend_elements = [Line2D([0], [0], color='indianred', lw = 2, label='Diversions'),
                       Line2D([0], [0], color='steelblue', lw = 2, label='Storage')]
    self.ax[1].legend(handles=legend_elements, loc='upper right', prop={'family':'Gill Sans MT','weight':'bold','size':8}, framealpha = 1.0, ncol = 2)      
    plt.savefig(self.figure_name, dpi = 300, bbox_inches = 'tight', pad_inches = 0.0)
    plt.close()
    
    
  def plot_release_simulation(self, release_data, snowpack_data, res_station, year_start, year_end, show_plot = False):
    qalycolors = sns.color_palette('gnuplot_r', 12)
    flow1 = {}
    flow2 = {}
    flow3 = {}
    snowpack = {}
    for x in range(0, 12):
      flow1[x] = []
      flow2[x] = []
      flow3[x] = []
      snowpack[x] = []
      
    for year_num in range(year_start, year_end):
      year_add = 0
      month_start = 10
      for month_num in range(0, 12):
        if month_start + month_num == 13:
          month_start -= 12
          year_add = 1
        datetime_val = datetime(year_num + year_add, month_start + month_num, 1, 0, 0)
        remaining_flow = 0.0
        remaining_usable_flow = 0.0
        remaining_diverted_flow = 0.0
        current_snowpack = snowpack_data.loc[datetime_val, 'basinwide_average']
        snowpack[month_num].append(current_snowpack)
        for lookahead_month in range(month_num, 12):
          if lookahead_month > 2:
            lookahead_datetime = datetime(year_num + 1, lookahead_month - 2, 1, 0, 0)
          else:
            lookahead_datetime = datetime(year_num, lookahead_month + 10, 1, 0, 0)
          remaining_flow += release_data.loc[lookahead_datetime, res_station + '_flow']
          remaining_usable_flow += release_data.loc[lookahead_datetime, res_station + '_diverted'] + release_data.loc[lookahead_datetime, res_station + '_available']
          remaining_diverted_flow += release_data.loc[lookahead_datetime, res_station + '_diverted']
        flow1[month_num].append(remaining_flow)
        flow2[month_num].append(remaining_usable_flow)
        flow3[month_num].append(remaining_diverted_flow)
        self.ax[0].scatter([current_snowpack,], [remaining_flow/1000.0,], color = qalycolors[month_num], s = 20)
        self.ax[1].scatter([current_snowpack,], [remaining_usable_flow/1000.0,], color = qalycolors[month_num], s = 20)
        self.ax[2].scatter([current_snowpack,], [remaining_diverted_flow/1000.0,], color = qalycolors[month_num], s = 20)
    coef = np.zeros((12,2))
    value_range = np.zeros((12,2))
    coef2 = np.zeros((12,2))
    coef3 = np.zeros((12,2))
    for x in range(0, 12):
      coef[x,:] = np.polyfit(np.asarray(snowpack[x]), np.asarray(flow1[x]), 1)
      coef2[x,:] = np.polyfit(np.asarray(snowpack[x]), np.asarray(flow2[x]), 1)
      coef3[x,:] = np.polyfit(np.asarray(snowpack[x]), np.asarray(flow3[x]), 1)
      
      simulated_residuals1 = np.zeros(len(snowpack[x]))
      simulated_residuals2 = np.zeros(len(snowpack[x]))
      simulated_residuals3 = np.zeros(len(snowpack[x]))
      for xx in range(0, len(snowpack[x])):
        simulated_residuals1[xx] = 100.0*(snowpack[x][xx] * coef[x,0] + coef[x, 1] - flow1[x][xx])/flow1[x][xx]
        simulated_residuals2[xx] = 100.0*(snowpack[x][xx] * coef2[x,0] + coef2[x, 1] - flow2[x][xx])/flow2[x][xx]
        simulated_residuals3[xx] = 100.0*(snowpack[x][xx] * coef3[x,0] + coef3[x, 1] - flow3[x][xx])/flow3[x][xx]
      
      value_range[x, 0] = np.min(np.asarray(snowpack[x]))
      value_range[x, 1] = np.max(np.asarray(snowpack[x]))
      
      estimate_small = (coef[x,0] * value_range[x, 0] + coef[x, 1])/1000.0
      estimate_large = (coef[x,0] * value_range[x, 1] + coef[x, 1])/1000.0
      estimate_small2 = (coef2[x,0] * value_range[x, 0] + coef2[x, 1])/1000.0
      estimate_large2 = (coef2[x,0] * value_range[x, 1] + coef2[x, 1])/1000.0
      estimate_small3 = (coef3[x,0] * value_range[x, 0] + coef3[x, 1])/1000.0
      estimate_large3 = (coef3[x,0] * value_range[x, 1] + coef3[x, 1])/1000.0
      
      self.ax[0].plot([value_range[x, 0], value_range[x, 1]], [estimate_small, estimate_large], linewidth = 3, color = qalycolors[x])
      self.ax[1].plot([value_range[x, 0], value_range[x, 1]], [estimate_small2, estimate_large2], linewidth = 3, color = qalycolors[x])
      self.ax[2].plot([value_range[x, 0], value_range[x, 1]], [estimate_small3, estimate_large3], linewidth = 3, color = qalycolors[x])
      
    self.ax[0].set_xlim([0.0, max(snowpack_data['basinwide_average'])])
    self.ax[1].set_xlim([0.0, max(snowpack_data['basinwide_average'])])
    self.ax[2].set_xlim([0.0, max(snowpack_data['basinwide_average'])])
    self.ax[0].set_ylim([0.0, max(flow1[0])/1000.0])
    self.ax[1].set_ylim([0.0, max(flow2[0])/1000.0])
    self.ax[2].set_ylim([0.0, max(flow3[0])/1000.0])
    self.ax[2].set_xlabel('Fraction of Average Annual Snowpack Accumulation', fontsize = 18, weight = 'bold', fontname = 'Gill Sans MT')
    month_index = ['OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP']
    self.ax[0].set_xticks([])
    self.ax[0].set_xticklabels('')
    self.ax[1].set_xticks([])
    self.ax[1].set_xticklabels('')
    self.ax[0].set_ylabel('Total Inflow\n(tAF)', fontsize = 18, weight = 'bold', fontname = 'Gill Sans MT')
    self.ax[1].set_ylabel('Total Available Inflow\n(tAF)', fontsize = 18, weight = 'bold', fontname = 'Gill Sans MT')
    self.ax[2].set_ylabel('Total Diversions\n(tAF)', fontsize = 18, weight = 'bold', fontname = 'Gill Sans MT')
    
    #Plot colorbar
    self.fig.subplots_adjust(right=0.9)
    cbar_ax = self.fig.add_axes([0.9187, 0.15, 0.025, 0.7])
    sm = plt.cm.ScalarMappable(cmap='gnuplot_r', norm=plt.Normalize(vmin=0, vmax=11))
    clb1 = plt.colorbar(sm, cax = cbar_ax, ticks=np.arange(0,12))
    clb1.ax.set_yticklabels(month_index) 
    #clb1.ax.invert_yaxis()
    clb1.ax.tick_params(labelsize=16)
    for item in clb1.ax.xaxis.get_ticklabels():
      item.set_fontname('Gill Sans MT')  
    for item in (self.ax[2].get_xticklabels()):
      item.set_fontsize(14)
      item.set_fontname('Gill Sans MT')
    for axesnum in range(0, 3):
      for item in (self.ax[axesnum].get_yticklabels()):
        item.set_fontsize(14)
        item.set_fontname('Gill Sans MT')

    plt.savefig(self.figure_name, dpi = 300, bbox_inches = 'tight', pad_inches = 0.0)
    if show_plot:
      plt.show()
    plt.close()
    
  def plot_release_simulation_controlled(self, release_data, snowpack_data, res_station, tot_coef, year_start, year_end, show_plot = False):
  
    monthly_control = {}
    monthly_control_int = {}
    list_num_dict = {}
    month_index = ['OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP']
    for x in range(1, 13):
      monthly_control[x] = []
      monthly_control_int[x] = []
    for index, row in release_data.iterrows():
      if index > datetime(1950, 10, 1, 0, 0):
        this_row_month = index.month
        monthly_control_int[this_row_month].append(row[res_station + '_location'])
    
    prev_cnt = 0
    for x in range(1,13):
      total_list = list(set(monthly_control_int[x]))
      monthly_control[x] = []
      for cont_loc in total_list:
        num_obs = 0
        for all_loc in monthly_control_int[x]:
          if all_loc == cont_loc:
            num_obs += 1

        if num_obs > 10:
          monthly_control[x].append(cont_loc)
      this_cnt = 0   
      for control_loc in monthly_control[x]:
        if control_loc not in list_num_dict:
          list_num_dict[control_loc] = this_cnt + prev_cnt
          this_cnt += 1
      prev_cnt += this_cnt
    
    plot_counter_x = 0
    plot_counter_y = 0
    num_x = 3
    num_y = 4 
    months_plot = ['FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL']
    plot_x = 3
    plot_y = 2
    coef = {}
    max_remaining_flow = 0.0
    qalycolors = sns.color_palette('gnuplot_r', prev_cnt)
    for month_num in range(0, 12):
      year_add = 0
      month_start = 10
      if month_start + month_num > 12:
        month_start = -2
        year_add = 1

      control_location_list = monthly_control[month_start + month_num]
      flow2 = {}
      snowpack = {}
      for x in control_location_list:
        flow2[x] = []
        snowpack[x] = []

      for year_num in range(year_start, year_end):
        datetime_val = datetime(year_num + year_add, month_start + month_num, 1, 0, 0)
        remaining_usable_flow = 0.0
        current_snowpack = snowpack_data.loc[datetime_val, 'basinwide_average']
        control_location = release_data.loc[datetime_val, res_station + '_location']
        
        for lookahead_month in range(month_num, 12):
          if lookahead_month > 2:
            lookahead_datetime = datetime(year_num + 1, lookahead_month - 2, 1, 0, 0)
          else:
            lookahead_datetime = datetime(year_num, lookahead_month + 10, 1, 0, 0)
          remaining_usable_flow += release_data.loc[lookahead_datetime, res_station + '_diverted'] + release_data.loc[lookahead_datetime, res_station + '_available']
        max_remaining_flow = max(max_remaining_flow, remaining_usable_flow)
        if control_location in flow2:
          snowpack[control_location].append(current_snowpack)
          flow2[control_location].append(remaining_usable_flow)
          color_number = list_num_dict[control_location]
        #  self.ax[plot_counter_x][plot_counter_y].scatter([current_snowpack,], [remaining_usable_flow/1000.0,], c = [qalycolors[color_number],], s = 100)
        #  self.ax[plot_counter_x][plot_counter_y].scatter([current_snowpack,], [remaining_usable_flow/1000.0,], c = 'lightslategray', s = 100)
        #else:
        if month_index[month_num] in months_plot:
          self.ax[plot_counter_x][plot_counter_y].scatter([current_snowpack,], [remaining_usable_flow/1000.0,], c = 'indianred', edgecolor = 'black', s = 250)
                
      coef[month_index[month_num]] = {}
      for cnt_x, control_loc in enumerate(control_location_list):        
        coef[month_index[month_num]][control_loc] = np.polyfit(np.asarray(snowpack[control_loc]), np.asarray(flow2[control_loc]), 1)
      
        value_range = np.zeros(2)
        value_range[0] = np.min(np.asarray(snowpack[control_loc]))
        value_range[1] = np.max(np.asarray(snowpack[control_loc]))
        estimate_small = (coef[month_index[month_num]][control_loc][0] * value_range[0] + coef[month_index[month_num]][control_loc][1])/1000.0
        estimate_large = (coef[month_index[month_num]][control_loc][0] * value_range[1] + coef[month_index[month_num]][control_loc][1])/1000.0
      
        color_number = list_num_dict[control_loc]
        #self.ax[plot_counter_x][plot_counter_y].plot([value_range[0], value_range[1]], [estimate_small, estimate_large], linewidth = 3, color = qalycolors[color_number])

      value_range = np.zeros(2)
      use_snowpack = snowpack_data[snowpack_data.index > datetime(year_start, 10, 1, 0, 0)]
      value_range[0] = np.min(use_snowpack['basinwide_average'])
      value_range[1] = np.max(use_snowpack['basinwide_average'])
      estimate_small = (tot_coef[month_num, 0] * value_range[0] + tot_coef[month_num, 1])/1000.0
      estimate_large = (tot_coef[month_num, 0] * value_range[1] + tot_coef[month_num, 1])/1000.0
      if month_index[month_num] in months_plot:
        self.ax[plot_counter_x][plot_counter_y].plot([value_range[0], value_range[1]], [estimate_small, estimate_large], linewidth = 5, color = 'indianred')

      
      ele_list = []
      #for cl in control_location_list:
        #color_number = list_num_dict[cl]
        #ele_list.append(Line2D([0], [0], markerfacecolor=qalycolors[color_number], markeredgecolor='black',  lw = 0, marker = 'o', markersize = 25, label=cl))
      ele_list.append(Line2D([0], [0], color='indianred', lw =5, label='Historical Trend'))
      if month_index[month_num] in months_plot:
        if plot_counter_y == 0 and plot_counter_x == 1:
          self.ax[plot_counter_x][plot_counter_y].set_ylabel('Remaining Available Water, (tAF)', fontsize = 50, weight = 'bold', fontname = 'Gill Sans MT')
        if plot_counter_y == 0:
          for item in (self.ax[plot_counter_x][plot_counter_y].get_yticklabels()):
            item.set_fontsize(20)
            item.set_fontname('Gill Sans MT')
        else:
          self.ax[plot_counter_x][plot_counter_y].set_yticks([])
          self.ax[plot_counter_x][plot_counter_y].set_yticklabels('')
        if plot_counter_x == (plot_x - 1):
          self.ax[plot_counter_x][plot_counter_y].set_xlabel('Snowpack Frac', fontsize = 35, weight = 'bold', fontname = 'Gill Sans MT')
          for item in (self.ax[plot_counter_x][plot_counter_y].get_xticklabels()):
            item.set_fontsize(20)
            item.set_fontname('Gill Sans MT')
        else:
          self.ax[plot_counter_x][plot_counter_y].set_xticks([])
          self.ax[plot_counter_x][plot_counter_y].set_xticklabels('')
        #if plot_counter_x == 0:
          #self.ax[plot_counter_x][plot_counter_y].text(0.65, 0.7, month_index[month_num], horizontalalignment='center', verticalalignment='center', transform=self.ax[plot_counter_x][plot_counter_y].transAxes, fontsize = 20, weight = 'bold', fontname = 'Gill Sans MT')
        #else:
        self.ax[plot_counter_x][plot_counter_y].text(0.2, 0.85, month_index[month_num], horizontalalignment='center', verticalalignment='center', transform=self.ax[plot_counter_x][plot_counter_y].transAxes, fontsize = 26, weight = 'bold', fontname = 'Gill Sans MT')
        if plot_counter_x == (plot_x - 1) and plot_counter_y == (plot_y - 1):
          self.ax[plot_counter_x][plot_counter_y].legend(handles=ele_list, loc='upper right', prop={'family':'Gill Sans MT','weight':'bold','size':24}, framealpha = 1.0, ncol = 1)      
        plot_counter_y += 1
        if plot_counter_y == plot_y:
          plot_counter_x += 1
          plot_counter_y = 0

    for month_num in range(0, 12):
      max_snowpack = np.max(snowpack_data.loc[snowpack_data.index > datetime(1950, 10, 1, 0, 0), 'basinwide_average'])
    for x in range(0, plot_x):
      for y in range(plot_y):
        self.ax[x][y].set_xlim([0.0, max_snowpack * 1.1])
        self.ax[x][y].set_ylim([0.0, max_remaining_flow * 1.1/1000.0])
        
    
    plt.savefig(self.figure_name, dpi = 300, bbox_inches = 'tight', pad_inches = 0.0)
    if show_plot:
      plt.show()
    plt.close()
    
    return coef


  def plot_available_water(self, reservoir_data, snowpack_data, release_data, delivery_data, snow_coefs, res_station, tunnel_station, year_start, year_end, animation_pane, show_plot = False):
    date_index = []
    total_storage = []
    total_consumed = []
    total_snowpack = []
    month_index = ['OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP']
    annual_water_index = np.ones(year_end-year_start) * 99999999.0
    threshold_color_list = sns.color_palette('rocket', 4)
    for year_num in range(year_start, year_end):
      year_add = 0
      month_start = 10
      already_diverted = 0.0
      for month_num in range(0, 12):
        month_val = month_index[month_num]
        if month_start + month_num == 13:
          month_start -= 12
          year_add = 1
        datetime_val = datetime(year_num + year_add, month_start + month_num, 1, 0, 0)
        available_storage = reservoir_data.loc[datetime_val, res_station]/1000.0
        already_diverted += delivery_data.loc[datetime_val, tunnel_station]/1000.0
        control_location = release_data.loc[datetime_val, res_station + '_location']
        if control_location in snow_coefs[month_val]:
          available_snowmelt = (snowpack_data.loc[datetime_val, 'basinwide_average'] * snow_coefs[month_val][control_location][0] + snow_coefs[month_val][control_location][1])/1000.0
        else:
          available_snowmelt = (snowpack_data.loc[datetime_val, 'basinwide_average'] * snow_coefs[month_val]['all'][0] + snow_coefs[month_val]['all'][1])/1000.0
        date_index.append(datetime_val)
        total_storage.append(available_storage)
        total_snowpack.append(available_storage + available_snowmelt)
        total_consumed.append(already_diverted + available_storage + available_snowmelt)
        if month_start + month_num > 3 and month_start + month_num < 10:
          annual_water_index[year_num - year_start] = min(already_diverted + available_storage + available_snowmelt, annual_water_index[year_num - year_start])
        
    #self.ax.fill_between(date_index, np.zeros(len(total_storage)), total_storage, color = 'steelblue', alpha = 0.7)
    transfer_threshold = 600.0
    self.ax.fill_between(date_index, np.zeros(len(total_storage)), total_storage, color = 'steelblue', edgecolor = 'black', alpha = 0.4)
    legend_elements = [Patch(facecolor='steelblue', edgecolor='black', label='Surface Storage', alpha = 0.7)]
    if animation_pane > 1:
      self.ax.fill_between(date_index, total_storage, total_snowpack, color = 'beige', edgecolor = 'black', alpha = 0.4)
      legend_elements.append(Patch(facecolor='beige', edgecolor='black', label='Snowpack Storage', alpha = 0.7))
    if animation_pane > 2:
      self.ax.fill_between(date_index, total_snowpack, total_consumed, color = 'teal', edgecolor = 'black', alpha = 0.4)
      self.ax.plot(date_index, total_consumed, color = 'indianred', linewidth = 3.0)
      legend_elements.append(Patch(facecolor='teal', edgecolor='black', label='YTD Consumed', alpha = 0.7))
      legend_elements.append(Line2D([0], [0], color='indianred', lw = 3, label='Total Water'))
    if animation_pane > 3:
      self.ax.plot(date_index, np.ones(len(date_index)) * 550.0, color = 'black', linewidth = 4.0, linestyle = '--')
      legend_elements.append(Line2D([0], [0], color='black', lw = 3, linestyle = '--', label='Threshold'))
    if animation_pane > 4:
      for x_cnt, thresh in enumerate([550.0, 600.0, 650.0, 700.0]):
        self.ax.plot(date_index, np.ones(len(date_index)) * thresh, color = threshold_color_list[x_cnt], linewidth = 4.0, linestyle = '--')
    self.ax.set_xlim([date_index[0], date_index[-1]])
    self.ax.set_ylim([0.0, np.max(np.asarray(total_consumed))*1.25])
    self.ax.set_ylabel('Total Available\nSupply (tAF)', fontsize = 28, weight = 'bold', fontname = 'Gill Sans MT')
    
    self.ax.legend(handles=legend_elements, loc='upper left', prop={'family':'Gill Sans MT','weight':'bold','size':18}, framealpha = 1.0, ncol = 5)
    for item in (self.ax.get_xticklabels()):
      item.set_fontsize(20)
      item.set_fontname('Gill Sans MT')
    for item in (self.ax.get_yticklabels()):
      item.set_fontsize(20)
      item.set_fontname('Gill Sans MT')

    plt.savefig(self.figure_name, dpi = 300, bbox_inches = 'tight', pad_inches = 0.0)
    if show_plot:
      plt.show()
    plt.close()
    
    return annual_water_index
    
  def plot_CBT_allocation(self, total_consumed, transfer_threshold, cumulative_losses, year_start, year_end, show_plot = False):
    date_index = []
    month_index = ['OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP']

    for year_num in range(year_start, year_end):
      year_add = 0
      month_start = 10
      already_diverted = 0.0
      for month_num in range(0, 12):
        month_val = month_index[month_num]
        if month_start + month_num == 13:
          month_start -= 12
          year_add = 1
        datetime_val = datetime(year_num + year_add, month_start + month_num, 1, 0, 0)
        date_index.append(datetime_val)
    cbt_allocations = pd.read_csv('input_files/northern_cbt_allocations.csv', index_col = 0)
    date_index_allocations = []
    total_allocations = []
    new_allocations = []
    allocation_min = 150.0
    allocation_max = 350.0
    for index, row in cbt_allocations.iterrows():
      date_index_allocations.append(datetime(index, 4, 1, 0, 0))
      total_allocations.append(float(row['allocation']) * 310.0)
      new_allocations.append(float(row['allocation']) * 310.0 + cumulative_losses[index - 1910]/1000.0)
    counter = 0
    for x in range(0, len(total_consumed)):
      if total_consumed[x] < transfer_threshold and counter == 0:
        start_point = date_index[x]
        counter = 1
      #elif total_consumed[x] > transfer_threshold and counter == 1:
        #self.ax.fill_between([start_point, date_index[x]], [allocation_min, allocation_min], [allocation_max,allocation_max], color = 'indianred', edgecolor = 'black', alpha = 0.4)
        #counter = 0
    self.ax.plot(date_index_allocations, total_allocations, color = 'black', linewidth = 3.0)
    self.ax.plot(date_index_allocations, new_allocations, color = 'indianred', linewidth = 3.0)
    self.ax.set_xlim([date_index[0], date_index[-1]])
    self.ax.set_ylim([allocation_min, allocation_max])
    self.ax.set_ylabel('Total Transbasin Diversions (tAF)', fontsize = 28, weight = 'bold', fontname = 'Gill Sans MT')
    #legend_elements2 = [Patch(facecolor='indianred', edgecolor='black', label='Transfer Months', alpha = 0.7), 
    legend_elements2 = [Line2D([0], [0], color='black', lw = 3, label='Historical Allocation'),
                        Line2D([0], [0], color='indianred', lw = 3, label='Historical Allocations w/Transfers')]
    self.ax.legend(handles=legend_elements2, loc='upper left', prop={'family':'Gill Sans MT','weight':'bold','size':18}, framealpha = 1.0, ncol = 2)
        
    for item in (self.ax.get_xticklabels()):
      item.set_fontsize(20)
      item.set_fontname('Gill Sans MT')
    for item in (self.ax.get_yticklabels()):
      item.set_fontsize(20)
      item.set_fontname('Gill Sans MT')

    plt.savefig(self.figure_name, dpi = 300, bbox_inches = 'tight', pad_inches = 0.0)
    if show_plot:
      plt.show()
    plt.close()


  def plot_physical_supply(self, simulated_releases, informal_transfer_network, station_id, structures_objects, start_year, end_year, show_plot = False):

    sorted_rights_list = structures_objects[station_id].sorted_rights
    start_date = datetime(start_year, 10, 1, 0, 0)
    end_date = datetime(end_year, 9, 30, 0, 0)
    use_simulated_releases = simulated_releases[np.logical_and(simulated_releases.index > start_date, simulated_releases.index < end_date)]
    use_informal_transfer_network = informal_transfer_network[np.logical_and(simulated_releases.index > start_date, simulated_releases.index < end_date)]
    prev_physical = 0.0
    counter = 0
    date_prev = 0
    qalycolors = sns.color_palette('RdYlBu', 100)
    max_qalycolor = np.max(use_informal_transfer_network[sorted_rights_list[0] + '_physical_available'])
    for index, row in use_informal_transfer_network.iterrows():
      
      if index.month > 3 and index.month < 10:
        min_paper = 999999.0
        for sri in sorted_rights_list:
          if structures_objects[station_id].rights_objects[sri].fill_type == 1:
            if row[sri + '_paper_available'] < min_paper:              
              min_paper = min(min_paper, row[sri + '_paper_available'])
              
      if index.month < 4 or index.month > 9:
        min_paper = 999999.0
        for sri in sorted_rights_list:
          if structures_objects[station_id].rights_objects[sri].fill_type == 2:
            if row[sri + '_paper_available'] < min_paper:              
              min_paper = min(min_paper, row[sri + '_paper_available'])
      if counter > 0:
        self.ax.fill_between([date_prev, index], np.zeros(2), [prev_physical/1000.0, row[sorted_rights_list[0] + '_physical_available']/1000.0], color = qalycolors[int(99.0*row[sri + '_physical_available']/max_qalycolor)])
      counter += 1
      date_prev = datetime(index.year, index.month, 1, 0, 0)
      prev_physical = row[sorted_rights_list[0] + '_physical_available'] * 1.0
    self.ax.plot(use_simulated_releases[station_id + '_physical_supply']/1000.0, linewidth = 1.0, color = 'black')
       
    self.ax.set_xlim([datetime(start_year, 10, 1, 0, 0), datetime(end_year, 9, 30, 0, 0)])
    self.ax.set_ylim([0, np.max(use_simulated_releases[station_id + '_physical_supply']) * 1.1 / 1000.0])
    self.ax.xaxis.set_major_locator(mdates.YearLocator(base=10)) 
    self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    self.ax.set_ylabel('Additional Physical Supply (tAF)', fontsize = 20, weight = 'bold', fontname = 'Gill Sans MT')
    for item in (self.ax.get_xticklabels()):
      item.set_fontsize(16)
      item.set_fontname('Gill Sans MT')
    for item in (self.ax.get_yticklabels()):
      item.set_fontsize(16)
      item.set_fontname('Gill Sans MT')
    
    self.fig.subplots_adjust(right=0.9)
    cbar_ax = self.fig.add_axes([0.95, 0.15, 0.025, 0.7])
    sm = plt.cm.ScalarMappable(cmap='RdYlBu', norm=plt.Normalize(vmin=0, vmax=int(np.max(use_simulated_releases[station_id + '_physical_supply'])/1000.0)))
    clb1 = plt.colorbar(sm, cax = cbar_ax, ticks=[0.0, int(np.max(use_simulated_releases[station_id + '_physical_supply'])/1000.0)])
    clb1.ax.set_yticklabels(['0', str(int(np.max(use_simulated_releases[station_id + '_physical_supply'])/1000.0))])
    #clb1.ax.invert_yaxis()
    cbar_ax.set_title('Demand\nBuyouts (tAF)', fontsize = 18, weight = 'bold', fontname = 'Gill Sans MT')

    clb1.ax.tick_params(labelsize=16)
    for item in clb1.ax.xaxis.get_ticklabels():
      item.set_fontname('Gill Sans MT')  
    
    plt.savefig(self.figure_name, dpi = 300, bbox_inches = 'tight', pad_inches = 0.0)
    if show_plot:
      plt.show() 
    plt.close()      
        
  def plot_snow_pdf(self, x_range, snow_vals_fill, snow_vals_shortage, density_height, counter1, counter2):  
    pos = np.linspace(x_range[0], x_range[1], 101)
    if len(np.unique(snow_vals_fill)) > 1:
      kde_fill = stats.gaussian_kde(snow_vals_fill)
      multiplier_fill = density_height / np.max(kde_fill(pos))
      self.ax[counter1][counter2].fill_between(pos, np.ones(len(pos)), multiplier_fill * kde_fill(pos) + np.ones(len(pos)), edgecolor = 'black', alpha = 0.6, facecolor = 'steelblue')
    if len(np.unique(snow_vals_shortage)) > 1:
      kde_shortage = stats.gaussian_kde(snow_vals_shortage)
      multiplier_shortage = -1.0 * density_height / np.max(kde_shortage(pos))
      self.ax[counter1][counter2].fill_between(pos, multiplier_shortage * kde_shortage(pos), np.zeros(len(pos)), edgecolor = 'black', alpha = 0.6, facecolor = 'indianred')
    self.ax[counter1][counter2].set_ylim([0.0 - density_height, 1.0 + density_height])
  
  def plot_index_observation(self, x_range, snow_vals, fill_vals, counter1, counter2, ylab):

    self.ax[counter1][counter2].plot(x_range, [0.0, 0.0], color = 'black', linewidth = 1.5)
    self.ax[counter1][counter2].plot(x_range, [1.0, 1.0], color = 'black', linewidth = 1.5)
    self.ax[counter1][counter2].scatter(snow_vals, fill_vals, color = 'black', s = 25)
    self.ax[counter1][counter2].set_xlim(x_range)
    self.ax[counter1][counter2].set_ylabel(ylab)
    if counter2 == 0:
      self.ax[counter1][counter2].set_yticks([0.0, 1.0])
      self.ax[counter1][counter2].set_yticklabels(['0%', '100%'], fontsize = 10, weight = 'bold', fontname = 'Gill Sans MT')
    else:
      self.ax[counter1][counter2].set_yticks([])
      self.ax[counter1][counter2].set_yticklabels('')
    if counter1 < 2:
      self.ax[counter1][counter2].set_xticks([])
      self.ax[counter1][counter2].set_xticklabels('')
    else:
      self.ax[counter1][counter2].set_xlabel('Normalized Snowpack')

  def plot_index_projection(self, estimated_fill, numsteps, xrange, counter1, counter2):
    self.ax[counter1][counter2].plot(np.linspace(xrange[0], xrange[1], num = numsteps), estimated_fill, linewidth = 2.0, color = 'black')
      
    
  def add_forecast_pdf(self, annual_forecasts, color_map, mn):
    exposure_colors = sns.color_palette(color_map, 6)  
    current_val = 0.0
    current_index = -1
    while current_val == 0.0:
      current_index += 1
      current_val = annual_forecasts[current_index]
    if len(np.unique(annual_forecasts[current_index:])) > 1:
      pos = np.linspace(np.min(annual_forecasts[current_index:]), np.max(annual_forecasts[current_index:]), 101)
      kde_vals = stats.gaussian_kde(annual_forecasts[current_index:])
      norm_mult = np.max(kde_vals(pos))
      self.ax.fill_between(pos, np.zeros(len(pos)), kde_vals(pos)/norm_mult, edgecolor = 'black', alpha = 0.6, facecolor = exposure_colors[mn])  
    
  def set_legend(self, legend_elements, legend_location, counter1, counter2 = 0, legend_title = 'none'):
  
    if legend_title == 'none':
      if self.sub_cols == 0 and self.sub_rows == 1:
        self.ax.legend(handles=legend_elements, loc=legend_location, prop={'family':'Gill Sans MT','weight':'bold','size':6}, framealpha = 1.0)
      elif self.sub_cols == 0:
        self.ax[counter1].legend(handles=legend_elements, loc=legend_location, prop={'family':'Gill Sans MT','weight':'bold','size':6}, framealpha = 1.0)
      else:
        self.ax[counter1][counter2].legend(handles=legend_elements, loc=legend_location, prop={'family':'Gill Sans MT','weight':'bold','size':6}, framealpha = 1.0)
    else:
      if self.sub_cols == 0 and self.sub_rows == 1:
        self.ax.legend(handles=legend_elements, loc=legend_location, prop={'family':'Gill Sans MT','weight':'bold','size':6}, title = legend_title, title_fontsize = 7, framealpha = 1.0)
      elif self.sub_cols == 0:
        self.ax[counter1].legend(handles=legend_elements, loc=legend_location, prop={'family':'Gill Sans MT','weight':'bold','size':6}, title = legend_title, title_fontsize = 7, framealpha = 1.0)
      else:
        self.ax[counter1][counter2].legend(handles=legend_elements, loc=legend_location, prop={'family':'Gill Sans MT','weight':'bold','size':6}, title = legend_title, title_fontsize = 7, framealpha = 1.0)

  def save_fig(self, figure_name, dpi_val = 150, bbox_inches_val = 'tight', pad_inches_val = 0.0):
    plt.tight_layout()
    plt.savefig(figure_name, dpi = int(dpi_val), bbox_inches = bbox_inches_val, pad_inches = pad_inches_val)

  def plot_snow_fnf_relationship(self, basin_snowpack, full_natural_flows, flow_station):
    snow_months = {}
    flow_months = {}
    for x in range(1, 13):
      snow_months[x] = []
      flow_months[x] = []
    counter = 0
    for date_ix in basin_snowpack.index:
      if basin_snowpack.loc[date_ix, 'basinwide_average'] > -1.0 and date_ix in full_natural_flows.index:
        if date_ix.month == 10 or counter == 0:
          cumulative_flow = 0.0
          total_flow = 0.0
          for new_month in range(0, 12):
            total_month = date_ix.month + new_month
            counter += 1
            if total_month > 12:
              total_month -= 12
              total_flow += full_natural_flows.loc[datetime(date_ix.year + 1, total_month, 1, 0, 0), flow_station]/ 1000000.0
            else:
              total_flow += full_natural_flows.loc[datetime(date_ix.year, total_month, 1, 0, 0), flow_station]/ 1000000.0 
        snow_months[date_ix.month].append(basin_snowpack.loc[date_ix, 'basinwide_average'])
        flow_months[date_ix.month].append(total_flow - cumulative_flow)
        cumulative_flow += full_natural_flows.loc[date_ix, flow_station]/ 1000000.0
    qalycolors = sns.color_palette('RdYlBu_r', 12)
    for x in range(1, 13):
      coef = np.polyfit(np.asarray(snow_months[x]), np.asarray(flow_months[x]), 1)
      xmin = np.min(np.asarray(snow_months[x]))
      xmax = np.max(np.asarray(snow_months[x]))
      if x < 10:
        self.ax.scatter(np.asarray(snow_months[x]), np.asarray(flow_months[x]), facecolor = qalycolors[x+2], s = 75, edgecolor = 'black') 
        self.ax.plot([xmin, xmax], [xmin*coef[0] + coef[1], xmax*coef[0] + coef[1]], color = qalycolors[x+2], linewidth = 4.0)
      else:
        self.ax.scatter(np.asarray(snow_months[x]), np.asarray(flow_months[x]), facecolor = qalycolors[x-10], s = 75, edgecolor = 'black', linewidth = 1.0) 
        self.ax.plot([xmin, xmax], [xmin*coef[0] + coef[1], xmax*coef[0] + coef[1]], color = qalycolors[x-10], linewidth = 4.0)
    self.ax.set_xlim([0.0, 1.02*np.max(np.asarray(snow_months[9]))])
    self.ax.set_ylim([0.0, 1.1*np.max(np.asarray(flow_months[10]))])
    self.ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1.0, decimals = None))
    self.ax.set_xlabel('Snowpack, % of mean final accumluation', fontsize = 20, weight = 'bold', fontname = 'Gill Sans MT')
    self.ax.set_ylabel('Remaining Annual Full-Natural-Flow,\nBasinwide (MAF)', fontsize = 20, weight = 'bold', fontname = 'Gill Sans MT')
    self.add_colorbar([0.875, 0.7, 0.025, 0.25], [0,6,11], ['October', 'April', 'September'], colorscale = 'RdYlBu_r')
    for item in (self.ax.get_xticklabels()):
      item.set_fontsize(18)
      item.set_fontname('Gill Sans MT')  
    for item in (self.ax.get_yticklabels()):
      item.set_fontsize(18)
      item.set_fontname('Gill Sans MT')

    plt.tight_layout()
    plt.savefig(self.figure_name, dpi = 300, bbox_inches = 'tight', pad_inches = 0.0)
    plt.show()

  def plot_rights_stack_prob(self, rights_stack_ids, rights_stack_structure_ids, structures_objects, reservoir_list, show_plot = False):
    right_counter = 0
    qalycolors = sns.color_palette('magma_r', 6)
    rights_stack_probs = {}
    first_fill_rights = []
    second_fill_rights = []
    for x in range(0, 12):
      rights_stack_probs[x] = np.zeros(len(rights_stack_ids))
    for r_id, s_id in zip(rights_stack_ids, rights_stack_structure_ids):
      this_right_demands = np.asarray(structures_objects[s_id].rights_objects[r_id].historical_monthly_demand['demand'])
      counter = 0
      total_demand = 0.0
      annual_demands = []
      for xxx in range(0, len(this_right_demands)):
        total_demand += this_right_demands[xxx]
        counter += 1
        if counter == 12:
          annual_demands.append(total_demand)
          counter = 0
          total_demand = 0.0
      annual_demands = np.asarray(annual_demands)
      if right_counter == 0:
        cumulative_stack_demand = np.zeros(len(annual_demands))
      for x in range(0, len(annual_demands)):
        cumulative_stack_demand[x] += annual_demands[x]
      sorted_demands = np.sort(cumulative_stack_demand)
      for x in range(0, 12):
        sorted_index = int(x * len(sorted_demands) / 11.0)
        if sorted_index == len(sorted_demands):
          rights_stack_probs[x][right_counter] = sorted_demands[sorted_index - 1] * 1.0
        else:
          rights_stack_probs[x][right_counter] = sorted_demands[sorted_index] * 1.0
      if s_id in reservoir_list:
        this_fill_right = structures_objects[s_id].rights_objects[r_id].fill_type
        if this_fill_right == 1:
          first_fill_rights.append(right_counter)
        elif this_fill_right == 2:
          second_fill_rights.append(right_counter)
      right_counter += 1
    for x in range(0, 11):
      color_int = int(np.power(np.power(x-5, 2), 0.5))
      self.ax.fill_between(np.arange(len(rights_stack_ids)), rights_stack_probs[x]/1000000.0, rights_stack_probs[x+1]/1000000.0, facecolor = qalycolors[color_int], alpha = 0.8 )
    self.ax.set_xlim([0.0, len(rights_stack_ids)])
    self.ax.set_ylim([0.0, max(rights_stack_probs[10])/1000000.0])
    for x in range(0, len(first_fill_rights)):
      self.ax.plot([first_fill_rights[x], first_fill_rights[x]], [0.0, max(rights_stack_probs[10])/1000000.0], linewidth = 3.0, color = 'indianred')
    for x in range(0, len(second_fill_rights)):
      self.ax.plot([second_fill_rights[x], second_fill_rights[x]], [0.0, max(rights_stack_probs[10])/1000000.0], linewidth = 3.0, linestyle = '--', color = 'steelblue')
    
    self.ax.set_xticks([len(rights_stack_ids) * 0.2, len(rights_stack_ids) * 0.8])
    self.ax.set_xticklabels(['<-- More Senior', 'More Junior -->'])
    self.ax.set_xlabel('Right Priority', fontsize = 20, weight = 'bold', fontname = 'Gill Sans MT')
    self.ax.set_ylabel('Cumulative Basin Demand (MAF)', fontsize = 20, weight = 'bold', fontname = 'Gill Sans MT')
    self.add_colorbar([0.075, 0.55, 0.025, 0.25], [0, 1], ['Median', 'Historical\nExtreme'], colorscale = 'magma_r')
    for item in (self.ax.get_xticklabels()):
      item.set_fontsize(18)
      item.set_fontname('Gill Sans MT')  
    for item in (self.ax.get_yticklabels()):
      item.set_fontsize(18)
      item.set_fontname('Gill Sans MT')

    legend_elements = [Line2D([0], [0], color='indianred', lw = 2, label='First Fill Rights'), Line2D([0], [0], color='steelblue', lw = 2, linestyle = '--', label='Second Fill Rights')]
    self.ax.legend(handles=legend_elements, loc='upper left', prop={'family':'Gill Sans MT','weight':'bold','size':18}, framealpha = 1.0, ncol = 1)      

    plt.savefig(self.figure_name, dpi = 300, bbox_inches = 'tight', pad_inches = 0.0)
    if show_plot:
      plt.show()
    plt.close()
    
    
  def add_colorbar(self, coords, label_loc, label_key, title_name = 'none', colorscale = 'none', nr = 0, nc = 0, fontsize = 12):
    if self.type == 'single':   
      fig = self.ax.get_figure()
    elif self.type == '2d':      
      fig = self.ax[nr][nc].get_figure()
    cax = fig.add_axes([coords[0], coords[1], coords[2], coords[3]])
    cmap = pl.cm.RdBu
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:,-1] = np.linspace(0.9, 1, cmap.N)
    if colorscale == 'none':
      my_cmap = ListedColormap(my_cmap)
    else:
      my_cmap = ListedColormap(sns.color_palette(colorscale).as_hex())
    sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=plt.Normalize(vmin=label_loc[0], vmax=label_loc[-1]))
    # fake up the array of the scalar mappable. Urgh...
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax)
    if title_name == 'none':
      skip =1
    else:
      cax.set_title(title_name, fontsize = fontsize, weight = 'bold', fontname = 'Gill Sans MT')
    cbar.set_ticks(label_loc)
    cbar.ax.set_yticklabels(label_key, fontsize = fontsize, weight = 'bold', fontname = 'Gill Sans MT')
