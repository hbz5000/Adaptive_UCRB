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

class Plotter():

  def __init__(self, figure_name, nr = 1, nc = 0):
    self.sub_rows = nr
    self.sub_cols = nc
    self.figure_name = figure_name
    if self.sub_cols == 0:
      self.fig, self.ax = plt.subplots(self.sub_rows)
      if self.sub_rows == 1:
        self.type = 'single'
        self.ax.grid(False)
      else:
        self.type = '1d'
    else:
      self.fig, self.ax = plt.subplots(self.sub_rows, self.sub_cols)
      self.type = '2d'
    plt.tight_layout()
    
    
  def plot_reservoir_figures(self, historical_values, simulated_values, reservoir_names):
    counter = 0
    for x, res_name in zip(historical_values, reservoir_names):
      self.ax[counter].plot(historical_values[x]/1000.0, color = 'indianred')
      self.ax[counter].plot(simulated_values[x]/1000.0, color = 'steelblue')
      self.ax[counter].set_ylabel(res_name + '\nStorage (tAF)', fontsize = 10, weight = 'bold', fontname = 'Gill Sans MT')
      self.ax[counter].set_xlim([historical_values.index[0], historical_values.index[-1]])
      increment = np.ceil(np.max(simulated_values[x]) / 50000.0) * 10.0
      self.ax[counter].set_ylim([0, np.ceil(np.max(simulated_values[x]/(increment*1000.0))) * increment])
      counter += 1
    legend_elements = [Line2D([0], [0], color='indianred', lw = 2, label='Historical'),
                       Line2D([0], [0], color='steelblue', lw = 2, label='Simulated')]
    self.ax[2].legend(handles=legend_elements, loc='lower left', prop={'family':'Gill Sans MT','weight':'bold','size':8}, framealpha = 1.0, ncol = 2)      

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
    
    
  def plot_release_simulation(self, release_data, snowpack_data, diversion_data, res_station, div_station, year_start, year_end):
    qalycolors = sns.color_palette('gnuplot_r', 12)
    flow1 = {}
    flow2 = {}
    snowpack = {}
    for x in range(0, 12):
      flow1[x] = []
      flow2[x] = []
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
        current_snowpack = snowpack_data.loc[datetime_val, 'basinwide_average']
        snowpack[month_num].append(current_snowpack)
        for lookahead_month in range(month_num, 12):
          if lookahead_month > 2:
            lookahead_datetime = datetime(year_num + 1, lookahead_month - 2, 1, 0, 0)
          else:
            lookahead_datetime = datetime(year_num, lookahead_month + 10, 1, 0, 0)
          remaining_flow += release_data.loc[lookahead_datetime, res_station + '_flow']
          remaining_usable_flow += release_data.loc[lookahead_datetime, res_station + '_flow'] - release_data.loc[lookahead_datetime, res_station + '_controlled'] + release_data.loc[lookahead_datetime, res_station + '_free']
        flow1[month_num].append(remaining_flow)
        flow2[month_num].append(remaining_usable_flow)
        self.ax[0].scatter([current_snowpack,], [remaining_flow/1000.0,], c = qalycolors[month_num], s = 20)
        self.ax[1].scatter([current_snowpack,], [remaining_usable_flow/1000.0,], c = qalycolors[month_num], s = 20)
    coef = np.zeros((12,2))
    value_range = np.zeros((12,2))
    coef2 = np.zeros((12,2))
    for x in range(0, 12):
      coef[x,:] = np.polyfit(np.asarray(snowpack[x]), np.asarray(flow1[x]), 1)
      simulated_residuals = np.zeros(len(snowpack[x]))
      for xx in range(0, len(snowpack[x])):
        simulated_residuals[xx] = 100.0*(snowpack[x][xx] * coef[x,0] + coef[x, 1] - flow1[x][xx])/flow1[x][xx]
      
      value_range[x, 0] = np.min(np.asarray(snowpack[x]))
      value_range[x, 1] = np.max(np.asarray(snowpack[x]))
      estimate_small = (coef[x,0] * value_range[x, 0] + coef[x, 1])/1000.0
      estimate_large = (coef[x,0] * value_range[x, 1] + coef[x, 1])/1000.0
      
      self.ax[0].plot([value_range[x, 0], value_range[x, 1]], [estimate_small, estimate_large], linewidth = 3, color = qalycolors[x])

      coef2[x,:] = np.polyfit(np.asarray(snowpack[x]), np.asarray(flow2[x]), 1)
      simulated_residuals2 = np.zeros(len(snowpack[x]))
      for xx in range(0, len(snowpack[x])):
        simulated_residuals2[xx] = 100.0*(snowpack[x][xx] * coef2[x,0] + coef2[x, 1] - flow2[x][xx])/flow2[x][xx]
      estimate_small2 = (coef2[x,0] * value_range[x, 0] + coef2[x, 1])/1000.0
      estimate_large2 = (coef2[x,0] * value_range[x, 1] + coef2[x, 1])/1000.0

      self.ax[1].plot([value_range[x, 0], value_range[x, 1]], [estimate_small2, estimate_large2], linewidth = 3, color = qalycolors[x])
      
      self.ax[2].scatter(np.ones(len(simulated_residuals)) * (x-0.1), simulated_residuals, s = 10, c = 'indianred')
      self.ax[2].scatter(np.ones(len(simulated_residuals2)) * (x+0.1), simulated_residuals2, s = 10, c = 'steelblue')
    self.ax[0].set_xlim([0.0, max(snowpack_data['basinwide_average'])])
    self.ax[1].set_xlim([0.0, max(snowpack_data['basinwide_average'])])
    month_index = ['OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP']
    self.ax[2].set_xticks(np.arange(0, 12))
    self.ax[2].set_xticklabels(month_index)
    self.ax[2].set_xlim([-0.5, 11.5])
    self.ax[2].set_ylim([-200.0, 200.0])
    self.ax[0].set_ylabel('Total Inflow\n(tAF)', fontsize = 10, weight = 'bold', fontname = 'Gill Sans MT')
    self.ax[1].set_ylabel('Total Available Inflow\n(tAF)', fontsize = 10, weight = 'bold', fontname = 'Gill Sans MT')
    self.ax[2].set_ylabel('Snow Predictor\nResiduals (%)', fontsize = 10, weight = 'bold', fontname = 'Gill Sans MT')
    self.ax[0].set_xlabel('Fraction of Average Annual Snowpack Accumulation', fontsize = 10, weight = 'bold', fontname = 'Gill Sans MT')
    self.ax[1].set_xlabel('Fraction of Average Annual Snowpack Accumulation', fontsize = 10, weight = 'bold', fontname = 'Gill Sans MT')
    self.ax[2].legend(['Total Flow', 'Available Flow'], loc = 'lower left', ncol = 2)
    self.ax[2].plot([-0.5, 11.5], [0.0, 0.0], linewidth = 1.0, color = 'black')
    
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
      item.set_fontsize(10)
      item.set_fontname('Gill Sans MT')

    plt.savefig(self.figure_name, dpi = 300, bbox_inches = 'tight', pad_inches = 0.0)
    plt.close()
    
    return coef2

  def plot_release_simulation_controlled(self, release_data, snowpack_data, diversion_data, res_station, div_station, year_start, year_end):
  
    monthly_control = {}
    monthly_control_int = {}
    list_num_dict = {}
    month_index = ['OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP']
    for x in range(1, 13):
      monthly_control[x] = []
      monthly_control_int[x] = []
    for index, row in release_data.iterrows():
      this_row_month = index.month
      monthly_control_int[this_row_month].append(row[res_station + '_location'])
      
    for x in range(1,13):
      monthly_control[x] = list(set(monthly_control_int[x]))
      for cl_cnt, control_loc in enumerate(monthly_control[x]):
        list_num_dict[control_loc + '_' + str(x)] = cl_cnt
    
    plot_counter_x = 0
    plot_counter_y = 0
    num_x = 3
    num_y = 4 
    coef = {}
    for month_num in range(0, 12):
      year_add = 0
      month_start = 10
      if month_start + month_num > 12:
        month_start = -2
        year_add = 1

      control_location_list = monthly_control[month_start + month_num]
      qalycolors = sns.color_palette('gnuplot_r', len(control_location_list))
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
        
        snowpack[control_location].append(current_snowpack)
        for lookahead_month in range(month_num, 12):
          if lookahead_month > 2:
            lookahead_datetime = datetime(year_num + 1, lookahead_month - 2, 1, 0, 0)
          else:
            lookahead_datetime = datetime(year_num, lookahead_month + 10, 1, 0, 0)
          remaining_usable_flow += release_data.loc[lookahead_datetime, res_station + '_flow'] - release_data.loc[lookahead_datetime, res_station + '_controlled'] + release_data.loc[lookahead_datetime, res_station + '_free']
        flow2[control_location].append(remaining_usable_flow)
        color_number = list_num_dict[control_location + '_' + str(month_start + month_num)]
        self.ax[plot_counter_x][plot_counter_y].scatter([current_snowpack,], [remaining_usable_flow/1000.0,], c = [qalycolors[color_number],], s = 20)
      
      coef[month_index[month_num]] = np.zeros((len(control_location_list),2))
      value_range = np.zeros((len(control_location_list),2))
      for cnt_x, control_loc in enumerate(control_location_list):
        if len(np.unique(np.asarray(snowpack[control_loc]))) > 1 and len(np.unique(np.asarray(flow2[control_loc]))) > 1:
          coef[month_index[month_num]][cnt_x,:] = np.polyfit(np.asarray(snowpack[control_loc]), np.asarray(flow2[control_loc]), 1)
      
          value_range[cnt_x, 0] = np.min(np.asarray(snowpack[control_loc]))
          value_range[cnt_x, 1] = np.max(np.asarray(snowpack[control_loc]))
          estimate_small = (coef[month_index[month_num]][cnt_x,0] * value_range[cnt_x, 0] + coef[month_index[month_num]][cnt_x, 1])/1000.0
          estimate_large = (coef[month_index[month_num]][cnt_x,0] * value_range[cnt_x, 1] + coef[month_index[month_num]][cnt_x, 1])/1000.0
      
          color_number = list_num_dict[control_loc + '_' + str(month_start + month_num)]
          self.ax[plot_counter_x][plot_counter_y].plot([value_range[cnt_x, 0], value_range[cnt_x, 1]], [estimate_small, estimate_large], linewidth = 3, color = qalycolors[color_number])

      
      max_snowpack = 0
      ele_list = []
      print(month_num, end = " ")
      for cl in control_location_list:
        print(cl, end = " ")
        max_snowpack = max(max_snowpack, np.max(np.asarray(snowpack[control_location])))
        color_number = list_num_dict[cl + '_' + str(month_start + month_num)]
        ele_list.append(Line2D([0], [0], markerfacecolor=qalycolors[color_number], markeredgecolor='black',  lw = 0, marker = 'o', markersize = 10, label=cl))
      print()
      self.ax[plot_counter_x][plot_counter_y].set_xlim([0.0, max_snowpack * 1.1])
      self.ax[plot_counter_x][plot_counter_y].set_ylabel('TW, ' + month_index[month_num] + ' (tAF)', fontsize = 10, weight = 'bold', fontname = 'Gill Sans MT')
      self.ax[plot_counter_x][plot_counter_y].set_xlabel('Snowpack Frac', fontsize = 8, weight = 'bold', fontname = 'Gill Sans MT')
      self.ax[plot_counter_x][plot_counter_y].legend(handles=ele_list, loc='upper left', prop={'family':'Gill Sans MT','weight':'bold','size':4}, framealpha = 1.0, ncol = 1)      
      plot_counter_x += 1
      if plot_counter_x == num_x:
        plot_counter_y += 1
        plot_counter_x = 0
        
    
    plt.savefig(self.figure_name, dpi = 300, bbox_inches = 'tight', pad_inches = 0.0)
    plt.show()
    plt.close()
    
    return coef


  def plot_available_water(self, reservoir_data, snowpack_data, diversion_data, snow_coefs, res_station, div_station, year_start, year_end):
    date_index = []
    total_storage = []
    total_water = []
    for year_num in range(year_start, year_end):
      year_add = 0
      month_start = 10
      for month_num in range(0, 12):
        if month_start + month_num == 13:
          month_start -= 12
          year_add = 1
        datetime_val = datetime(year_num + year_add, month_start + month_num, 1, 0, 0)
        available_storage = reservoir_data.loc[datetime_val, res_station + '_account_1']/1000.0
        available_snowmelt = (snowpack_data.loc[datetime_val, 'basinwide_average'] * snow_coefs[month_num,0] + snow_coefs[month_num,1])/1000.0
        date_index.append(datetime_val)
        total_storage.append(available_storage)
        total_water.append(available_storage + available_snowmelt)
    self.ax.fill_between(date_index, np.zeros(len(total_storage)), total_storage, color = 'steelblue', alpha = 0.7)
    self.ax.fill_between(date_index, total_storage, total_water, color = 'paleturquoise', edgecolor = 'black', alpha = 0.7)
    self.ax.set_xlim([date_index[0], date_index[-1]])
    self.ax.set_ylabel('Total CBT Water Available', fontsize = 14, weight = 'bold', fontname = 'Gill Sans MT')
    self.ax.legend(['Reservoir Storage', 'Snowpack Storage'])
    plt.savefig(self.figure_name, dpi = 300, bbox_inches = 'tight', pad_inches = 0.0)
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
