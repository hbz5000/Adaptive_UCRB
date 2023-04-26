import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import geopandas as gpd
import seaborn as sns
from datetime import datetime

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

  def set_plotting_color_palette(self):
    #ensure plotting colors are consistent throughout plots
    colors_use = sns.color_palette('rocket_r', 4)
    color_use_value = {}
    color_use_value['550'] = 3
    color_use_value['600'] = 2
    color_use_value['650'] = 1
    color_use_value['700'] = 0
    return colors_use, color_use_value
    
  def plot_crop_types(self, filename, show_plot = False):
    #figure 2 in the manuscript - crop prices vs. irrigated acreage within the UCRB
    
    #use irrigation shapefile to find total irrigated area of each crop within the UCRB
    irrigation_ucrb = gpd.read_file(filename)
    irrigation_ucrb = irrigation_ucrb.to_crs(epsg = 3857)
    overall_crop_areas = {}
    for index, row in irrigation_ucrb.iterrows():
      #get total irrigated acreage (in thousand acres) for all listed crops
      if row['CROP_TYPE'] in overall_crop_areas:
        overall_crop_areas[row['CROP_TYPE']] += row['ACRES'] / 1000.0
      else:
        overall_crop_areas[row['CROP_TYPE']] = row['ACRES'] / 1000.0
    #crop marginal net benefits ($/acre) come from enterpirse budget reports from the CSU ag extension
    marginal_net_benefits = {}
    marginal_net_benefits['VEGETABLES'] = 506.0
    marginal_net_benefits['ALFALFA'] = 306.0
    marginal_net_benefits['BARLEY'] = 401.0
    marginal_net_benefits['BLUEGRASS'] = 401.0
    marginal_net_benefits['CORN_GRAIN'] = 50.0
    marginal_net_benefits['DRY_BEANS'] = 64.0
    marginal_net_benefits['GRASS_PASTURE'] = 401.0
    marginal_net_benefits['SOD_FARM'] = 401.0
    marginal_net_benefits['SMALL_GRAINS'] = 401.0
    marginal_net_benefits['SORGHUM_GRAIN'] = 401.0
    marginal_net_benefits['WHEAT_FALL'] = 252.0
    marginal_net_benefits['WHEAT_SPRING'] = 252.0
    #et requirements come from .........
    et_requirements = {}
    effective_precip = 3.1
    et_requirements['VEGETABLES'] = 26.2
    et_requirements['ALFALFA'] = 36.0
    et_requirements['BARLEY'] = 22.2
    et_requirements['BLUEGRASS'] = 30.0
    et_requirements['CORN_GRAIN'] = 26.9
    et_requirements['DRY_BEANS'] = 18.1
    et_requirements['GRASS_PASTURE'] = 30.0
    et_requirements['SOD_FARM'] = 30.0
    et_requirements['SMALL_GRAINS'] = 22.2
    et_requirements['SORGHUM_GRAIN'] = 24.5
    et_requirements['WHEAT_FALL'] = 16.1
    et_requirements['WHEAT_SPRING'] = 16.1
    et_requirements['ORCHARD_WITH_COVER'] = 22.2
    et_requirements['ORCHARD_WO_COVER'] = 22.2
    et_requirements['GRAPES'] = 22.2
    #for perennial crops, consider the costs of replacing the tree/vine and how long it takes to return to full production
    #these are revenues - costs for vineyards ($/acre) in each year, compared to a baseline
    grapes_planting_costs = [-6385.0, -2599.0, -1869.0, 754.0, 2012.0, 2133.0, 2261.0] 
    grapes_baseline_revenue = [2261.0, 2261.0, 2261.0, 2261.0, 2261.0, 2261.0, 2261.0]
    total_npv_costs = 0.0
    counter = 0
    #discount cost/revenue by 2.5% per year to get a NPV cost
    for cost, baseline in zip(grapes_planting_costs, grapes_baseline_revenue):
      total_npv_costs +=  (baseline - cost)/np.power(1.025, counter)
      counter += 1
    marginal_net_benefits['GRAPES'] = total_npv_costs
    #these are revenues - costs for a typical fruit orchard (peaches are used here) - considers a 20-year life cycle for trees, and assume fallow of 10-year old trees
    #(i.e., compare the NPV of fallowing now vs. fallowing in 10 years)
    orchard_planting_costs = [-5183.0, -2802.0, -2802.0, 395.0, 5496.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0] 
    orchard_baseline_revenue = [9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, 9398.0, -5183.0, -2802.0, -2802.0, 395.0, 5496.0]
    total_npv_costs = 0.0
    counter = 0
    #calculate difference in NPV across the entire cycle (6 years for grapes, 15 year for peaches)
    for cost, baseline in zip(orchard_planting_costs, orchard_baseline_revenue):
      total_npv_costs +=  (baseline - cost)/np.power(1.025, counter)
      counter += 1
    marginal_net_benefits['ORCHARD_WITH_COVER'] = total_npv_costs
    marginal_net_benefits['ORCHARD_WO_COVER'] = total_npv_costs
    
    #sort crops from highest cost to lowest cost
    water_costs = np.zeros(len(marginal_net_benefits))
    crop_list = []
    counter = 0
    for x in marginal_net_benefits:
      #MNB = $/ac; et = AF/ac; MNB/et = $/AF
      water_costs[counter] = marginal_net_benefits[x] / (et_requirements[x] / 12.0)#et is ac-in, divide by 12 for ac-ft
      crop_list.append(x)
      counter += 1
    #sort crops by $/AF
    sorted_index = np.argsort(water_costs*(-1.0))
    crop_list_new = np.asarray(crop_list)
    sorted_crops = crop_list_new[sorted_index]

    running_area = 0.0#total crop acreage across all crop types
    for x in sorted_crops:
      # $/AF of fallowing for current crop type
      total_cost = marginal_net_benefits[x] / (et_requirements[x] / 12.0)
            
      #acreage irrigated for this crop type      
      total_area = overall_crop_areas[x]
      total_area = total_area * 4.046
      total_cost = total_cost / 1233.0
      #add in line break so peaches ($6000/AF) and grapes ($10000/AF) are on the same plot as annual crops
      if total_cost > 4 and total_cost < 5:
        total_cost = 0.4 + (total_cost - 4)/10.0
      elif total_cost > 5:
        total_cost = 0.7 + (total_cost - 8)/10.0
      #elif total_cost > 5500.0:
        #total_cost = 400.0 + (total_cost - 5500.0)/5.0
      #plot value per acre foot vs. cumulative acreage, with crops ordered from highest value crop to lowest value crop
      self.ax.fill_between([running_area, running_area + total_area], np.zeros(2), [total_cost, total_cost], facecolor = 'indianred', edgecolor = 'black', linewidth = 2.0)
      running_area += total_area

    #format plot
    self.ax.set_yticks([0.0, .1, .2, 0.4, 0.5, 0.7, 0.8])
    self.ax.set_yticklabels(['0', '0.1', '0.2', '4', '5', '8', '9'])
    self.ax.set_ylabel('Cost of Fallowing ($MM per km2)', fontsize = 42, weight = 'bold', fontname = 'Gill Sans MT')
    self.ax.set_xlabel('UCRB Irrigation (km2)', fontsize = 42, weight = 'bold', fontname = 'Gill Sans MT')
    self.ax.set_ylim([0.0, .85])
    self.ax.set_xlim([-10.0, running_area * 1.05])
    for item in (self.ax.get_xticklabels()):
      item.set_fontsize(32)
      item.set_fontname('Gill Sans MT')
    for axesnum in range(0, 3):
      for item in (self.ax.get_yticklabels()):
        item.set_fontsize(32)
        item.set_fontname('Gill Sans MT')

    plt.savefig(self.figure_name, dpi = 300, bbox_inches = 'tight', pad_inches = 0.0)
    if show_plot:
      plt.show()
    plt.close()
    
  def plot_trigger_stages(self, index_use_list = [550.0, 600.0, 650.0, 700.0], plot_sales = False, show_plot = False):
  
    cbi_timeseries = pd.read_csv('cbi_timeseries.csv', index_col = 0)
    cbi_timeseries.index = pd.to_datetime(cbi_timeseries.index)
    cbi_timeseries.sort_index(inplace=True)
    snowpack_vol = np.asarray(cbi_timeseries['snowpack'])
    diversion_vol = np.asarray(cbi_timeseries['diversion'])
    storage_vol = np.asarray(cbi_timeseries['storage'])
    plot_index = pd.to_datetime(cbi_timeseries.index)
    num_timesteps = len(cbi_timeseries['storage'])
    colors_use, color_use_value = self.set_plotting_color_palette()
    legend_elements = []
    labels_use = {}
    labels_use['700'] = 'Stage 1'
    labels_use['650'] = 'Stage 2'
    labels_use['600'] = 'Stage 3'
    labels_use['550'] = 'Stage 4'
    year_use_list = []
    for thresh_use in index_use_list:
      total_benefits_all = 0
      all_leases = pd.read_csv('results_' + str(int(thresh_use)) + '/diversions_5104055.csv')#get record of additional exports (completed leases)
      all_leases['datetime'] = pd.to_datetime(all_leases['date'])
      all_buyouts = pd.read_csv('results_' + str(int(thresh_use)) + '/buyouts_2_5104055.csv')
      all_buyouts = all_buyouts.drop_duplicates(subset = ['demand', 'demand_purchase', 'date'])#buyout data is duplicated because it the same structure can be recorded as 'bought out' by multiple water right leases but only needs to be bought out once
      all_buyouts['datetime'] = pd.to_datetime(all_buyouts['date'])
      all_cr = pd.read_csv('results_' + str(int(thresh_use)) + '/individual_changes.csv')
      all_cr['datetime'] = pd.to_datetime(all_cr['dates'])
      legend_elements.append(Patch(facecolor=colors_use[color_use_value[str(int(thresh_use))]], edgecolor='black', label=labels_use[str(int(thresh_use))], alpha = 1.0))
      for year_use in range(1950, 2014):
        
        leases_year = all_leases[np.logical_and(all_leases['datetime'] >= datetime(year_use, 1, 1, 0, 0), all_leases['datetime'] < datetime(year_use + 1, 1, 1, 0, 0))]
        buyouts_year = all_buyouts[np.logical_and(all_buyouts['datetime'] >= datetime(year_use, 1, 1, 0, 0), all_buyouts['datetime'] < datetime(year_use + 1, 1, 1, 0, 0))]
        cr_year = all_cr[np.logical_and(all_cr['datetime'] >= datetime(year_use, 1, 1, 0, 0), all_cr['datetime'] < datetime(year_use + 1, 1, 1, 0, 0))]
        this_year_lease = np.sum(leases_year['demand']) * (-1.0)
        this_year_cr = np.sum(cr_year['compensatory']) * 1000
        this_year_buyout = np.sum(buyouts_year['demand_purchase'])
        if this_year_lease > 3000 and year_use not in year_use_list:
          year_use_list.append(year_use)
          buyouts_per_lease = 16 * this_year_buyout / (this_year_lease - this_year_cr) - 500
          scaled_leases =(this_year_lease - this_year_cr) / 54.1
          scaled_leases_tot = this_year_lease / 54.1
          total_benefits = (this_year_lease - this_year_cr) * (900.0 - 231.0) / 25000
          total_benefits_all += total_benefits
          self.ax.fill_between([datetime(year_use, 1, 1, 0, 0), datetime(year_use + 1, 1, 1, 0, 0)], [-500, -500], [total_benefits - 500, total_benefits - 500], facecolor = colors_use[color_use_value[str(int(thresh_use))]], edgecolor = 'black', linewidth = 1.0)
          self.ax.fill_between([datetime(year_use, 1, 1, 0, 0), datetime(year_use + 1, 1, 1, 0, 0)], np.zeros(2), [scaled_leases, scaled_leases], facecolor = colors_use[color_use_value[str(int(thresh_use))]], edgecolor = 'black', linewidth = 1.0)
          self.ax.fill_between([datetime(year_use, 1, 1, 0, 0), datetime(year_use + 1, 1, 1, 0, 0)], [scaled_leases, scaled_leases], [scaled_leases_tot, scaled_leases_tot], facecolor = 'steelblue', edgecolor = 'black', linewidth = 1.0)
    cbi_index = storage_vol + snowpack_vol + diversion_vol
    self.ax.plot(plot_index, cbi_index, color = 'black', linewidth = 3.5)
    for x_cnt, thresh in enumerate(index_use_list):
      self.ax.plot(plot_index, np.ones(num_timesteps) * thresh, color = colors_use[color_use_value[str(int(thresh))]], linewidth = 4.0, linestyle = '--')
    
    self.ax.fill_between(plot_index, cbi_index, np.ones(num_timesteps) * 550.0, where = cbi_index <= np.ones(num_timesteps) * 550.0, facecolor = colors_use[color_use_value['550']], interpolate=True)
    self.ax.fill_between(plot_index, np.maximum(np.ones(num_timesteps) * 550.0, cbi_index), np.ones(num_timesteps) * 600.0, where = cbi_index <= np.ones(num_timesteps) * 600.0, facecolor = colors_use[color_use_value['600']], interpolate=True)
    self.ax.fill_between(plot_index, np.maximum(np.ones(num_timesteps) * 600.0, cbi_index), np.ones(num_timesteps) * 650.0, where = cbi_index <= np.ones(num_timesteps) * 650.0, facecolor = colors_use[color_use_value['650']], interpolate=True)
    self.ax.fill_between(plot_index, np.maximum(np.ones(num_timesteps) * 650.0, cbi_index), np.ones(num_timesteps) * 700.0, where = cbi_index <= np.ones(num_timesteps) * 700.0, facecolor = colors_use[color_use_value['700']], interpolate=True)
    self.ax.plot(plot_index, np.zeros(num_timesteps), color = 'black', linewidth = 1.0)
    #format figure
    self.ax.set_xlim([datetime(1951, 1, 1, 0, 0), plot_index[-1]])
    self.ax.set_ylim([0.0, np.max(np.asarray(storage_vol + snowpack_vol + diversion_vol))*1.05])
    self.ax.set_ylabel('         Total Net Benefits       Total Informal Leases      Colorado-Big Thompson       \n     Informal Leasing ($MM)               (km3)                  Water Supply Index (tAF)    ', fontsize = 24, weight = 'bold', fontname = 'Gill Sans MT')
    self.ax.legend(handles=legend_elements, loc='upper left', prop={'family':'Gill Sans MT','weight':'bold','size':24}, framealpha = 1.0, ncol = 4)
    self.ax.set_yticks([-500, -300, -100, 0, 150, 300, 450, 500, 600, 700, 800, 900, 1000])
    self.ax.set_yticklabels([0, 5, 10, 0,  0.01,  0.02, 0.03, 500, 600, 700, 800, 900, 1000])
    self.ax.fill_between(plot_index, np.ones(num_timesteps) * 450, cbi_index, facecolor = 'slategray', linewidth = 0.0, alpha = 0.5)
    self.ax.fill_between([datetime(1952, 1, 1, 0, 0), datetime(1953, 1, 1, 0, 0)], [350, 350], [400, 400], facecolor = 'steelblue', edgecolor = 'black', linewidth = 1.0)
    self.ax.text(datetime(1959, 3, 1, 0, 0),375, 'Compensatory Release', fontsize = 20, weight = 'bold', fontname = 'Gil Sans MT',verticalalignment='center', horizontalalignment='center', zorder = 20)
    
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
    
  def plot_available_water(self, index_use_list = [550.0, 600.0, 650.0, 700.0]):
    cbi_timeseries = pd.read_csv('output_files/cbi_timeseries.csv', index_col = 0)
    plot_index = pd.to_datetime(cbi_timeseries.index)
    snowpack_vol = np.asarray(cbi_timeseries['snowpack'])
    diversion_vol = np.asarray(cbi_timeseries['diversion'])
    storage_vol = np.asarray(cbi_timeseries['storage'])
    num_timesteps = len(cbi_timeseries['storage'])
    colors_use, color_use_value = self.set_plotting_color_palette()
    
    self.ax.fill_between(plot_index, np.zeros(num_timesteps), storage_vol, color = 'steelblue', edgecolor = 'black', alpha = 0.6)
    legend_elements = [Patch(facecolor='steelblue', edgecolor='black', label='Surface Storage', alpha = 0.7)]
    animation_pane = 4
    #snowpack component
    if animation_pane > 1:
      self.ax.fill_between(plot_index, storage_vol, storage_vol + snowpack_vol, color = 'beige', edgecolor = 'black', alpha = 1.0)
      legend_elements.append(Patch(facecolor='beige', edgecolor='black', label='Snowpack Storage', alpha = 0.7))
    #tunnel component
    if animation_pane > 2:
      self.ax.fill_between(plot_index, storage_vol + snowpack_vol, storage_vol + snowpack_vol + diversion_vol, color = 'teal', edgecolor = 'black', alpha = 0.6)
      self.ax.plot(plot_index, storage_vol + snowpack_vol + diversion_vol, color = 'black', linewidth = 3.5)
      legend_elements.append(Patch(facecolor='teal', edgecolor='black', label='YTD Consumed', alpha = 0.7))
      legend_elements.append(Line2D([0], [0], color='indianred', lw = 3, label='Water Supply Index'))
    if animation_pane > 3:
      for x_cnt, thresh in enumerate(index_use_list):
        if len(index_use_list) == 4:
          self.ax.plot(plot_index, np.ones(num_timesteps) * thresh, color = colors_use[colors_use_vals[str(int(thresh))]], linewidth = 4.0, linestyle = '--')
        else:
          self.ax.plot(plot_index, np.ones(num_timesteps) * thresh, color = 'crimson', linewidth = 4.0, linestyle = '--')
        
    #format figure
    self.ax.set_xlim([plot_index[0], plot_index[-1]])
    self.ax.set_ylim([0.0, np.max(np.asarray(storage_vol + snowpack_vol + diversion_vol))*1.2])
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
    
  def plot_option_payments(self, thresh_use_list):
    colors_use, color_use_value = self.set_plotting_color_palette()
    for thresh_use in thresh_use_list:
      option_payments = pd.read_csv('results_' + str(int(thresh_use)) + '/option_payments_facilitators.csv', index_col = 0)
      option_payments = option_payments[np.logical_and(option_payments.index != '3603543', option_payments.index != '3604512')]
      sorted_payment = np.argsort(np.asarray(option_payments['annual payment'])*(-1.0))
      sorted_index = option_payments.index[sorted_payment]
      counter = 0
      for station_use in sorted_index:
        annual_payment = option_payments.loc[station_use, 'annual payment']
        this_loading = (10. * option_payments.loc[station_use, 'loading'] - 1.0) * 100
        self.ax[1].fill_between([counter, counter + 0.85], np.zeros(2), [annual_payment, annual_payment], facecolor = colors_use[color_use_value[thresh_use]], edgecolor = 'black', linewidth = 1.0, alpha = 1.0)
        self.ax[0].plot([counter + 0.425, ], [this_loading, ], marker = 'o', markersize = 25, markerfacecolor = colors_use[color_use_value[thresh_use]], markeredgecolor = 'black', linewidth = 1.0, alpha = 1.0)
        counter += 1
        
    self.ax[0].set_xlim([0, 20])
    self.ax[1].set_ylim([0.0,  110])
    self.ax[1].set_xlim([0, 20])
    self.ax[0].set_ylim([40,  95])
    self.ax[1].set_ylabel('Annual Option Cost ($1000)', fontsize = 28, weight = 'bold', fontname = 'Gill Sans MT')
    self.ax[0].set_ylabel('Total Loading (%)', fontsize = 28, weight = 'bold', fontname = 'Gill Sans MT')
    station_labels = ['Min Flow', 'Min Flow', 'Min Flow', 'Shoshone PP', 'Min Flow', 'Min Flow', 'Min Flow', 'GVP1', 'Min Flow', 'GVP2', 'GVP3', 'Min Flow', 'Min Flow', 'Min Flow', 'Min Flow', 'Min Flow', 'Min Flow', 'Min Flow', 'Min Flow', 'Min Flow']
    self.ax[0].set_xticklabels('')
    self.ax[0].set_xticks([])
    self.ax[1].set_xticklabels(station_labels)
    self.ax[1].set_xticks(np.arange(20) + 0.425)
    legend_elements = [Patch(facecolor=colors_use[color_use_value['550']], edgecolor='black', label='Stage 4 Drought', alpha = 0.7)]
    legend_elements.append(Patch(facecolor=colors_use[color_use_value['600']], edgecolor='black', label='Stage 3 Drought', alpha = 0.7))
    legend_elements.append(Patch(facecolor=colors_use[color_use_value['650']], edgecolor='black', label='Stage 2 Drought', alpha = 0.7))
    legend_elements.append(Patch(facecolor=colors_use[color_use_value['700']], edgecolor='black', label='Stage 1 Drought', alpha = 0.7))
    self.ax[0].legend(handles=legend_elements, loc='upper left', prop={'family':'Gill Sans MT','weight':'bold','size':18}, framealpha = 1.0, ncol = 5)
    for ax_num in range(0, 2):
      for item in (self.ax[ax_num].get_xticklabels()):
        item.set_fontsize(20)
        item.set_fontname('Gill Sans MT')
      for item in (self.ax[ax_num].get_yticklabels()):
        item.set_fontsize(20)
        item.set_fontname('Gill Sans MT')
    plt.xticks(rotation = 90)
    plt.savefig(self.figure_name, dpi = 300, bbox_inches = 'tight', pad_inches = 0.0)
    plt.close()
      
      
  
  def plot_informal_leases(self, thresh_use_list, ani_plot):
    #figure 4 - annual frequency of informal lease purchases by informal leasing threshold
    
    #set color scheme    
    colors_use, color_use_value = self.set_plotting_color_palette()
    bar_width = 0.85#width of bars on plot (max = 1.0)
    freq_scale = 150.0#maximum plotting frequency (x-axis)
    bpl_scale = 150.0
    counter = 3#counter for informal leasing scenarios
    counter_ani = 0
    y_tick_locs = []
    y_tick_labs = []
    y_labels_int = ['Stage 4', 'Stage 3', 'Stage2',  'Stage 1',]
    years_leased = {}
    for thresh_use in thresh_use_list:
      all_leases = pd.read_csv('results_' + str(int(thresh_use)) + '/diversions_5104055.csv')#get record of additional exports (completed leases)
      all_leases['datetime'] = pd.to_datetime(all_leases['date'])
      all_buyouts = pd.read_csv('results_' + str(int(thresh_use)) + '/buyouts_2_5104055.csv')
      all_buyouts = all_buyouts.drop_duplicates(subset = ['demand', 'demand_purchase', 'date'])#buyout data is duplicated because it the same structure can be recorded as 'bought out' by multiple water right leases but only needs to be bought out once
      all_buyouts['datetime'] = pd.to_datetime(all_buyouts['date'])
      all_cr = pd.read_csv('results_' + str(int(thresh_use)) + '/individual_changes.csv')
      all_cr['datetime'] = pd.to_datetime(all_cr['dates'])

      total_leases = np.zeros(2014 - 1950)
      for index, row in all_leases.iterrows():
        total_leases[row['datetime'].year - 1950] += row['demand'] * (-1.0) / 1000.0
      for index, row in all_cr.iterrows():
        if total_leases[row['datetime'].year - 1950] > 0:
          total_leases[row['datetime'].year - 1950] -= row['compensatory']
      only_leases = total_leases[total_leases > 0]
      years_leased[thresh_use] = len(only_leases)
      all_options = pd.read_csv('results_' + thresh_use + '/option_payments_facilitators_updated.csv', index_col = 0)
      all_options = all_options[np.logical_and(all_options.index != '3603543', all_options.index != '3604512')]
      total_annual_option_payment = np.sum(all_options['annual payment']) / np.mean(total_leases)
      
      buyout_per_lease = 5.0 * np.sum(all_buyouts['demand_purchase']) / ( (-1.0) * np.sum(all_leases['demand']) - np.sum(all_cr['compensatory']))
      #plot bar of frequency for a given informal leasing scenario
      self.ax.fill_between([-1 * total_annual_option_payment * 0.5 / freq_scale, 0.0], [float(counter), float(counter)], [float(counter) + bar_width, float(counter) + bar_width], facecolor = 'indianred', edgecolor = 'black', linewidth = 0.5, alpha = 1.0)
      if ani_plot >= 5:
        self.ax.fill_between([0.0, 4.0 * buyout_per_lease/bpl_scale], [float(counter), float(counter)], [float(counter) + bar_width, float(counter) + bar_width], facecolor = 'steelblue', edgecolor = 'black', linewidth = 0.5, alpha = 0.4)
        self.ax.fill_between([-1 * total_annual_option_payment * 2.0 / freq_scale, 0.0], [float(counter), float(counter)], [float(counter) + bar_width, float(counter) + bar_width], facecolor = 'indianred', edgecolor = 'black', linewidth = 0.5, alpha = 0.4)
      if ani_plot >= 4:
        self.ax.fill_between([0.0, 2.0 * buyout_per_lease/bpl_scale], [float(counter), float(counter)], [float(counter) + bar_width, float(counter) + bar_width], facecolor = 'steelblue', edgecolor = 'black', linewidth = 0.5, alpha = 0.7)
        self.ax.fill_between([-1 * total_annual_option_payment / freq_scale, 0.0], [float(counter), float(counter)], [float(counter) + bar_width, float(counter) + bar_width], facecolor = 'indianred', edgecolor = 'black', linewidth = 0.5, alpha = 0.7)
      self.ax.fill_between([0.0, buyout_per_lease/bpl_scale], [float(counter), float(counter)], [float(counter) + bar_width, float(counter) + bar_width], facecolor = 'steelblue', edgecolor = 'black', linewidth = 0.5, alpha = 1.0)
      if counter_ani == ani_plot:
        break
      counter_ani += 1
      counter -= 1
    #format plot
    self.ax.set_ylabel('Lease Option Exercise Threshold                        ', fontsize = 36, weight = 'bold', fontname = 'Gill Sans MT', labelpad = 25)
    self.ax.set_xlabel('Anunal Option Payment (dollars/m3)                                               Average Transaction Cost (dollars/m3)', fontsize = 36, weight = 'bold', fontname = 'Gill Sans MT')
    legend_elements = [] 
    legend_elements.append(Patch(facecolor='indianred', edgecolor='black', label='Up-front option fee (5% return flow uncertainty)', alpha = 1.0))
    if ani_plot >= 4:
      legend_elements.append(Patch(facecolor='indianred', edgecolor='black', label='Up-front option fee (10% return flow uncertainty)', alpha = 0.7))
      if ani_plot == 5:
        legend_elements.append(Patch(facecolor='indianred', edgecolor='black', label='Up-front option fee (20% return flow uncertainty)', alpha = 0.4))    
    legend_elements.append(Patch(facecolor='steelblue', edgecolor='black', label='$5 per acre-foot facilitator fee', alpha = 1.0))
    if ani_plot >= 4:
      legend_elements.append(Patch(facecolor='steelblue', edgecolor='black', label='$10 per acre-foot facilitator fee', alpha = 0.7))
      if ani_plot == 5:
        legend_elements.append(Patch(facecolor='steelblue', edgecolor='black', label='$20 per acre-foot facilitator fee', alpha = 0.4))    
    self.ax.legend(handles=legend_elements, loc='upper left', prop={'family':'Gill Sans MT','weight':'bold','size':36}, framealpha = 1.0, ncol = 2)  
    
    self.ax.plot([0, 0], [0, 4], linewidth = 1.5, color = 'black')
    for x in range(0, min(ani_plot + 1, len(y_labels_int))):
      y_tick_locs.append(float(len(y_labels_int)) - float(x) - (1.0 - bar_width/2.0))
      y_tick_labs.append(y_labels_int[x])
    self.ax.set_yticks(y_tick_locs)
    self.ax.set_ylim([0, 5.25])
    self.ax.set_xlim([-1, 1])
    self.ax.set_xticks([-1*(123.3/freq_scale), -0.5*(123.3/freq_scale), 0.0, 0.5*(123.3/freq_scale), 1.0*(123.3/freq_scale)])
    self.ax.set_xticklabels(['0.1', '0.05', '0', '0.05', '0.1'])
    y_cnt = 0
    #for thresh_use in reversed(thresh_use_list):
      #self.ax.get_yticklabels()[y_cnt].set_color(colors_use[color_use_value[thresh_use]])
      #if y_cnt == ani_plot:
        #break
      #y_cnt += 1

    self.ax.set_yticklabels(y_tick_labs)
    
    for item in (self.ax.get_xticklabels()):
      item.set_fontsize(28)
      item.set_fontname('Gill Sans MT')
    for item in (self.ax.get_yticklabels()):
      item.set_fontsize(36)
      item.set_fontname('Gill Sans MT')
    plt.savefig(self.figure_name, dpi = 300, bbox_inches = 'tight', pad_inches = 0.0)
    plt.close()
    
    return years_leased
  def plot_third_party_impacts(self, thresh_use_list, years_leased, ani_plot = 3, plot_type = 'all'):
    #figure 5 - total leases vs. third party impacts and potential compensatory releases for different informal leasing scenarios
    group_spacing = 2.25#total width of each informal lease scenario group
    bar_width = 0.85#width of individual bars
    intra_group_spacing = 1.0
    max_val = 0.0
    min_val = 0.0
    if ani_plot < 2:
      x_tick_locs = []
      if plot_type == 'single':
        x_tick_labels = ['Without\nCompensatory Releases', 'With Compensatory\nReleases']
        for x_cnt, x_lab in enumerate(x_tick_labels):
          x_tick_locs.append(float(x_cnt) + bar_width / 2.0)
      else:
        x_tick_labels = ['Leases triggered by:\nStage 4 Drought', ]
        x_tick_locs =[(intra_group_spacing + bar_width) / 2.0,]
    else:
      label_int = ['Leases triggered by:\nStage 4 Drought', 'Leases triggered by:\nStage 3 Drought', 'Leases triggered by:\nStage 2 Drought', 'Leases triggered by:\nStage 1 Drought']
      x_tick_locs = []
      x_tick_labels = []
      for x_cnt in range(0, ani_plot - 1):
        x_tick_labels.append(label_int[x_cnt])
        x_tick_locs.append(group_spacing * float(x_cnt) + (intra_group_spacing + bar_width) / 2.0)
        if x_cnt == ani_plot - 2:
          break
    for x_cnt, folder_name in enumerate(thresh_use_list):#informal lease scenario loop
      #read in aggregate leases, shortfalls, and compensatory releases for each leasing scenario
      third_party_impacts = pd.read_csv('results_' + folder_name + '/total_changes.csv', index_col = 0)
      
      start_val = 0
      total_exports = third_party_impacts.loc['Exports', 'change'] / float(years_leased[folder_name])#total leases
      e_losses = third_party_impacts.loc['Environment', 'change'] / float(years_leased[folder_name])#total instream flow shortfall
      con_losses = third_party_impacts.loc['Other', 'change'] / float(years_leased[folder_name])#total consumptive diversion shortfall
      comp_release = third_party_impacts.loc['Compensatory', 'change'] / float(years_leased[folder_name])#compensatory release
      max_val = max(max_val, total_exports)
      min_val = min(min_val, con_losses + e_losses)
      self.ax.fill_between([x_cnt * group_spacing, x_cnt * group_spacing + bar_width], [0.0, 0.0], [total_exports, total_exports], facecolor = 'goldenrod', edgecolor = 'black', linewidth = 1.5, alpha = 0.8)
      if ani_plot < 2:
        if ani_plot > 0:
          self.ax.fill_between([x_cnt * group_spacing, x_cnt * group_spacing + bar_width], [e_losses, e_losses], [0.0, 0.0], facecolor = 'steelblue', edgecolor = 'black', linewidth = 1.5, alpha = 0.8)        
          self.ax.fill_between([x_cnt * group_spacing, x_cnt * group_spacing + bar_width], [con_losses + e_losses,con_losses + e_losses], [e_losses, e_losses], facecolor = 'forestgreen', edgecolor = 'black', linewidth = 1.5, alpha = 0.8)
        if ani_plot > 1:
          self.ax.fill_between([x_cnt * group_spacing + intra_group_spacing, x_cnt * group_spacing + bar_width + intra_group_spacing], [0.0, 0.0], [total_exports - comp_release, total_exports - comp_release], facecolor = 'goldenrod', edgecolor = 'black', linewidth = 1.5, alpha = 0.8)
          self.ax.fill_between([x_cnt * group_spacing + intra_group_spacing, x_cnt * group_spacing + bar_width + intra_group_spacing], [total_exports - comp_release, total_exports - comp_release], [total_exports, total_exports], facecolor = 'indianred', edgecolor = 'black', linewidth = 1.5, alpha = 0.8)
      else:        
        self.ax.fill_between([x_cnt * group_spacing, x_cnt * group_spacing + bar_width], [e_losses, e_losses], [0.0, 0.0], facecolor = 'steelblue', edgecolor = 'black', linewidth = 1.5, alpha = 0.8)
        self.ax.fill_between([x_cnt * group_spacing, x_cnt * group_spacing + bar_width], [con_losses + e_losses,con_losses + e_losses], [e_losses, e_losses], facecolor = 'forestgreen', edgecolor = 'black', linewidth = 1.5, alpha = 0.8)
        self.ax.fill_between([x_cnt * group_spacing + intra_group_spacing, x_cnt * group_spacing + bar_width + intra_group_spacing], [0.0, 0.0], [total_exports - comp_release, total_exports - comp_release], facecolor = 'goldenrod', edgecolor = 'black', linewidth = 1.5, alpha = 0.8)
        self.ax.fill_between([x_cnt * group_spacing + intra_group_spacing, x_cnt * group_spacing + bar_width + intra_group_spacing], [total_exports - comp_release, total_exports - comp_release], [total_exports, total_exports], facecolor = 'indianred', edgecolor = 'black', linewidth = 1.5, alpha = 0.8)
      if x_cnt >= ani_plot - 2:
        break
    #format plot
    self.ax.plot([-0.15, len(thresh_use_list) * group_spacing], [0.0, 0.0], linewidth = 1.5, color = 'black')
    self.ax.set_xlim([-0.15, len(thresh_use_list) * group_spacing])
    self.ax.set_ylim([min_val * 1.05, max_val * 1.05])
    legend_elements = [Patch(facecolor='goldenrod', edgecolor='black', label='C-BT Leases    ', alpha = 1.0),]    
    if ani_plot > 0:
      legend_elements.append(Patch(facecolor='steelblue', edgecolor='black', label='Environmental\nShortfalls', alpha = 1.0))
      legend_elements.append(Patch(facecolor='forestgreen', edgecolor='black', label='Consumptive\nShortfalls', alpha = 1.0))
    if ani_plot > 1:
      legend_elements.append(Patch(facecolor='indianred', edgecolor='black', label='Compensatory\nReleases', alpha = 1.0))
    
    self.ax.legend(handles=legend_elements, loc='lower right', prop={'family':'Gill Sans MT','weight':'bold','size':32}, framealpha = 1.0, ncol = 2)  
    self.ax.set_ylabel('Total Change in Diversions (m3 per year)', fontsize = 36, weight = 'bold', fontname = 'Gill Sans MT')
    
    self.ax.set_xticks(x_tick_locs)
    self.ax.set_xticklabels(x_tick_labels)
    self.ax.set_yticks([0, 6.16, 12.33, 18.49])
    self.ax.set_yticklabels(['0', '5', '10', '15'])    
    for item in (self.ax.get_xticklabels()):
      item.set_fontsize(36)
      item.set_fontname('Gill Sans MT')
    for item in (self.ax.get_yticklabels()):
      item.set_fontsize(36)
    plt.savefig(self.figure_name, dpi = 300, bbox_inches = 'tight', pad_inches = 0.0)
    plt.close()
    
  def plot_informal_price(self, thresh_use, ani_plot, plot_facilitators = False):
    #this plots marginal transaction cost vs. total informal leasing, compared to the transaction costs of a similar volume of formal leases
    color_use = sns.color_palette('rocket', 4)
    #load informal leasing output
    all_diversions = pd.read_csv('results_' + thresh_use + '/diversions_5104055.csv')  
    all_purchases = pd.read_csv('results_' + thresh_use + '/purchases_5104055.csv')  
    all_buyouts = pd.read_csv('results_' + thresh_use + '/buyouts_2_5104055.csv')
    all_buyouts = all_buyouts.drop_duplicates(subset = ['demand', 'demand_purchase', 'date'])#buyout data is duplicated because it the same structure can be recorded as 'bought out' by multiple water right leases but only needs to be bought out once
    for lease_df in [all_diversions, all_purchases, all_buyouts]:
      lease_df['datetime_use'] = pd.to_datetime(lease_df['date'])
      lease_df['year_use'] = pd.DatetimeIndex(lease_df['datetime_use']).year
      lease_df['month_use'] = pd.DatetimeIndex(lease_df['datetime_use']).month
    
    #find total volume of informal leases received by buyer and lease facilitator options exercised in each year
    cost_list = []
    volume_list = []
    year_list = []
    for year in range(1950, 2014):
      #find this years monthly informal lease and facilitator options exercised
      this_month_purchase = all_purchases[all_purchases['year_use'] == year]
      this_month_buyout = all_buyouts[all_buyouts['year_use'] == year]
      this_month_diversion = all_diversions[all_diversions['year_use'] == year]
        
      total_purchases = 0.0
      total_buyouts = 0.0
      total_diversions = 0.0
      #purchases from lease sellers are only for the consumptive portion of the use
      for index, row in this_month_purchase.iterrows():
        total_purchases += row['demand'] * row['consumptive']
      #facilitator purchases are for the entire facilitated demand amount at each structure        
      for index, row in this_month_buyout.iterrows():
        total_buyouts += row['demand_purchase']
      #diversions are the additional water that was able to be diverted by the exporters
      for index, row in this_month_diversion.iterrows():
        total_diversions += row['demand'] * (-1.0)
      #find total buyout volume per af of purchased lease seller options (i.e., facilitated demand/leased demand)
      if total_purchases > 0.0:
        cost_list.append(total_buyouts/total_purchases)
        year_list.append(year)
        volume_list.append(total_diversions / 1000)
      
    cost_list = np.asarray(cost_list)
    year_list = np.asarray(year_list)
    #sort the years in which informal leases are purchased by 
    #the transaction cost (i.e., annual average facilitated demand per leased demand)
    sorted_cost_index = np.argsort(cost_list)
    
    #set up an array to track the cumulative volume leased, moving from lowest-transaction cost year to highest
    #set up an array to track the marginal transaction cost per volume of informal lease, moving from lowest-transaction cost yeaer to highest
    marginal_cost = np.zeros(len(cost_list))
    #total savings from informal transfers for all purchases based on a $200/AF (minimum) and $360/AF estimate of formal legal transaction costs
    cumulative_savings = np.zeros(4)
    
    unique_facilitator_list = np.zeros(len(marginal_cost))#cumulative number of unique facilitators used, moving from lowest-transaction cost year to highest
    list_of_facilitators = []#running list of unique facilitators
    for xxx in range(0, len(marginal_cost)):
      #cumulative volume of informal leases purchased at a transaction cost lower than marginal_cost[xxx]
      #transaction cost of informal leases purchased in year - indexed to sorted position XXX
      marginal_cost[xxx] = cost_list[sorted_cost_index[xxx]]
      #calculate the savings in a given year, add to cumulative total
      cumulative_savings[0] += (200. - marginal_cost[xxx] * 5.) * volume_list[sorted_cost_index[xxx-1]]#savings using $5/af facilitator fee, low bound of formal legal cost
      cumulative_savings[1] += (360. - marginal_cost[xxx] * 5.) * volume_list[sorted_cost_index[xxx-1]]#savings using $5/af facilitator fee, high bound of formal legal cost
      cumulative_savings[2] += (200. - marginal_cost[xxx] * 10.) * volume_list[sorted_cost_index[xxx-1]]#savings using $10/af facilitator fee, low bound of formal legal cost
      cumulative_savings[3] += (360. - marginal_cost[xxx] * 10.) * volume_list[sorted_cost_index[xxx-1]]#savings using $10/af facilitator fee, high bound of formal legal cost
      this_year = year_list[sorted_cost_index[xxx-1]]
      #add to list of unique facilitators from the facilitators needed to make this years informal leases
      this_month_buyout = all_buyouts[all_buyouts['year_use'] == this_year]
      this_month_unique = this_month_buyout['structure'].unique()
      for x in range(0, len(this_month_unique)):
        if this_month_unique[x] not in list_of_facilitators:
          list_of_facilitators.append(this_month_unique[x])
      unique_facilitator_list[xxx] = len(list_of_facilitators)#count of facilitators from years with the current marginal transaction costs or lower
      
    #set plot
    price_use = 5.#facilitator fees calculated in $5/af increments
    color_list = ['beige', 'goldenrod', 'indianred']
    legend_elements = []
    price_use = 5.0
    self.ax.fill_between([0.0, len(marginal_cost)], [200.0, 200.0], [360.0, 360.0], facecolor = 'maroon', edgecolor = 'black', linewidth = 0.5, alpha = 0.4)
    self.ax.plot([0.0, len(marginal_cost)], [200.0, 200.0], linewidth = 4.0, linestyle = '--', color = 'maroon')
    self.ax.plot([0.0, len(marginal_cost)], [360.0, 360.0], linewidth = 4.0, linestyle = '--', color = 'maroon')
    for inc_no in range(0, ani_plot):
      for xxx in range(0, len(marginal_cost)):
        price_use = inc_no * 5.0
        self.ax.fill_between([xxx, xxx + 0.85], np.ones(2) * marginal_cost[xxx] * price_use, np.ones(2) * marginal_cost[xxx] * (price_use+5), color = color_list[inc_no], edgecolor = 'black', linewidth = 1.0, alpha = 1.0)
      legend_elements.append(Patch(facecolor=color_list[inc_no], edgecolor='black', label='Facilitator Fee: $' + str(int(5*(inc_no+1))) + '/AF'))
    #plot range of informal costs
    #option to plot the cumulative number of unique facilitators, moving from lowest marginal transaction cost to highest
    if plot_facilitators:
      ax2 = self.ax.twinx()
      ax2.plot(cumulative_volume, unique_facilitator_list, color = 'black', linewidth = 4.5)
      ax2.set_ylabel('Number of Unique Facilitators', fontsize = 32, weight = 'bold', fontname = 'Gill Sans MT')
      ax2.set_ylim([0, np.max(unique_facilitator_list) * 9.0/5.0])
      legend_elements.append(Line2D([0], [0], color='black', lw = 4, label='Number of Facilitators'))
      for item in (ax2.get_yticklabels()):
        item.set_fontsize(22)
        item.set_fontname('Gill Sans MT')
    #format plot
    self.ax.set_ylim([0, 400.0])
    self.ax.set_xlim([0, len(marginal_cost)])
    self.ax.text(float(len(marginal_cost)) / 2.0, 280.0, 'Range of Transaction Costs,\nFormal Legal System', fontsize = 64, weight = 'bold', fontname = 'Gill Sans MT', ha = 'center', va = 'center', color = 'white')
    self.ax.set_ylabel('Annual Transaction Costs ($/AF)', fontsize = 32, weight = 'bold', fontname = 'Gill Sans MT')
    self.ax.set_xlabel('Years with Triggered Leases', fontsize = 32, weight = 'bold', fontname = 'Gill Sans MT')
    if len(legend_elements) > 0:
      self.ax.legend(handles=legend_elements, loc='center left', prop={'family':'Gill Sans MT','weight':'bold','size':32 }, framealpha = 1.0, ncol = 1)          
    for item in (self.ax.get_xticklabels()):
      item.set_fontsize(22)
      item.set_fontname('Gill Sans MT')
    for item in (self.ax.get_yticklabels()):
      item.set_fontsize(22)
      item.set_fontname('Gill Sans MT')
    
    plt.savefig(self.figure_name, dpi = 300, bbox_inches = 'tight', pad_inches = 0.0)
    plt.close()

  def plot_informal_price_alt(self, thresh_use_list, thresh_use_list_total, label_use_list, plot_facilitators = False):
    #this plots marginal transaction cost vs. total informal leasing, compared to the transaction costs of a similar volume of formal leases
    #load informal leasing output
    inc_no = 0
    legend_elements = []
    for thresh_use in thresh_use_list_total:
      all_diversions = pd.read_csv('results_' + thresh_use + '/diversions_5104055.csv')  
      year_list = []
      for year in range(1950, 2014):
        #find this years monthly informal lease and facilitator options exercised
        all_diversions['datetime_use'] = pd.to_datetime(all_diversions['date'])
        all_diversions['year_use'] = pd.DatetimeIndex(all_diversions['datetime_use']).year
        this_month_diversion = all_diversions[all_diversions['year_use'] == year]
        total_diversions = 0.0
        for index, row in this_month_diversion.iterrows():
          total_diversions += row['demand'] * (-1.0)
        #find total buyout volume per af of purchased lease seller options (i.e., facilitated demand/leased demand)
        if total_diversions > 0.0:
          year_list.append(year)
    total_years = len(year_list)      
    for thresh_use, label_use in zip(thresh_use_list, label_use_list):
      all_diversions = pd.read_csv('results_' + thresh_use + '/diversions_5104055.csv')  
      all_purchases = pd.read_csv('results_' + thresh_use + '/purchases_5104055.csv')  
      all_buyouts = pd.read_csv('results_' + thresh_use + '/buyouts_2_5104055.csv')
      all_buyouts = all_buyouts.drop_duplicates(subset = ['demand', 'demand_purchase', 'date'])#buyout data is duplicated because it the same structure can be recorded as 'bought out' by multiple water right leases but only needs to be bought out once
      for lease_df in [all_diversions, all_purchases, all_buyouts]:
        lease_df['datetime_use'] = pd.to_datetime(lease_df['date'])
        lease_df['year_use'] = pd.DatetimeIndex(lease_df['datetime_use']).year
        lease_df['month_use'] = pd.DatetimeIndex(lease_df['datetime_use']).month
    
      #find total volume of informal leases received by buyer and lease facilitator options exercised in each year
      cost_list = []
      year_list = []
      for year in range(1950, 2014):
        #find this years monthly informal lease and facilitator options exercised
        this_month_purchase = all_purchases[all_purchases['year_use'] == year]
        this_month_buyout = all_buyouts[all_buyouts['year_use'] == year]
        this_month_diversion = all_diversions[all_diversions['year_use'] == year]
        
        total_purchases = 0.0
        total_buyouts = 0.0
        total_diversions = 0.0
        #purchases from lease sellers are only for the consumptive portion of the use
        for index, row in this_month_purchase.iterrows():
          total_purchases += row['demand'] * row['consumptive']
        #facilitator purchases are for the entire facilitated demand amount at each structure        
        for index, row in this_month_buyout.iterrows():
          total_buyouts += row['demand_purchase']
        #diversions are the additional water that was able to be diverted by the exporters
        for index, row in this_month_diversion.iterrows():
          total_diversions += row['demand'] * (-1.0)
        #find total buyout volume per af of purchased lease seller options (i.e., facilitated demand/leased demand)
        if total_purchases > 0.0:
          cost_list.append(total_buyouts/total_purchases)
          year_list.append(year)
      
      cost_list = np.asarray(cost_list)
      year_list = np.asarray(year_list)
      #sort the years in which informal leases are purchased by 
      #the transaction cost (i.e., annual average facilitated demand per leased demand)
      sorted_cost_index = np.argsort(cost_list)
    
      #set up an array to track the cumulative volume leased, moving from lowest-transaction cost year to highest
      #set up an array to track the marginal transaction cost per volume of informal lease, moving from lowest-transaction cost yeaer to highest
      marginal_cost = np.zeros(len(cost_list))
      #total savings from informal transfers for all purchases based on a $200/AF (minimum) and $360/AF estimate of formal legal transaction costs
    
      unique_facilitator_list = np.zeros(len(marginal_cost))#cumulative number of unique facilitators used, moving from lowest-transaction cost year to highest
      list_of_facilitators = []#running list of unique facilitators
      for xxx in range(0, len(marginal_cost)):
        #cumulative volume of informal leases purchased at a transaction cost lower than marginal_cost[xxx]
        #transaction cost of informal leases purchased in year - indexed to sorted position XXX
        marginal_cost[xxx] = cost_list[sorted_cost_index[xxx]]
        #calculate the savings in a given year, add to cumulative total
        this_year = year_list[sorted_cost_index[xxx]]
        #add to list of unique facilitators from the facilitators needed to make this years informal leases
        this_month_buyout = all_buyouts[all_buyouts['year_use'] == this_year]
        this_month_unique = this_month_buyout['structure'].unique()
        for x in range(0, len(this_month_unique)):
          if this_month_unique[x] not in list_of_facilitators:
            list_of_facilitators.append(this_month_unique[x])
        unique_facilitator_list[xxx] = len(list_of_facilitators)#count of facilitators from years with the current marginal transaction costs or lower
      
      #set plot
      price_use = 5.#facilitator fees calculated in $5/af increments
      color_list = ['beige', 'darkkhaki', 'darkgoldenrod', 'indianred']
      price_use = 5.0
      if thresh_use == thresh_use_list[-1]:
        self.ax.fill_between([0.0, total_years], [200.0, 200.0], [360.0, 360.0], facecolor = 'maroon', edgecolor = 'black', linewidth = 0.5, alpha = 0.4)
        self.ax.plot([0.0, total_years], [200.0, 200.0], linewidth = 4.0, linestyle = '--', color = 'maroon')
        self.ax.plot([0.0, total_years], [360.0, 360.0], linewidth = 4.0, linestyle = '--', color = 'maroon')
      for xxx in range(0, len(marginal_cost)):
        price_use = 0.0
        self.ax.fill_between([xxx, xxx + 0.85], np.ones(2) * marginal_cost[xxx] * price_use, np.ones(2) * marginal_cost[xxx] * (price_use+5), color = color_list[inc_no], edgecolor = 'black', linewidth = 1.0, alpha = 1.0)
      legend_elements.append(Patch(facecolor=color_list[inc_no], edgecolor='black', label=label_use))
      inc_no += 1
      #plot range of informal costs
      #option to plot the cumulative number of unique facilitators, moving from lowest marginal transaction cost to highest
      if thresh_use == thresh_use_list[-1] and plot_facilitators:
        ax2 = self.ax.twinx()
        ax2.plot(np.arange(len(unique_facilitator_list)), unique_facilitator_list, color = 'black', linewidth = 4.5)
        ax2.set_ylabel('Number of Unique Facilitators', fontsize = 32, weight = 'bold', fontname = 'Gill Sans MT')
        ax2.set_ylim([0, np.max(unique_facilitator_list) * 9.0/5.0])
        legend_elements.append(Line2D([0], [0], color='black', lw = 4, label='Number of Facilitators'))
        for item in (ax2.get_yticklabels()):
          item.set_fontsize(22)
          item.set_fontname('Gill Sans MT')
    #format plot
    self.ax.set_ylim([0, 400.0])
    self.ax.set_xlim([0, total_years])
    self.ax.text(float(total_years) / 2.0, 280.0, 'Range of Transaction Costs,\nFormal Legal System', fontsize = 64, weight = 'bold', fontname = 'Gill Sans MT', ha = 'center', va = 'center', color = 'white')
    self.ax.set_ylabel('Annual Transaction Costs ($/AF)', fontsize = 32, weight = 'bold', fontname = 'Gill Sans MT')
    self.ax.set_xlabel('Years with Triggered Leases', fontsize = 32, weight = 'bold', fontname = 'Gill Sans MT')
    if len(legend_elements) > 0:
      self.ax.legend(handles=legend_elements, loc='center left', prop={'family':'Gill Sans MT','weight':'bold','size':32 }, framealpha = 1.0, ncol = 1)          
    for item in (self.ax.get_xticklabels()):
      item.set_fontsize(22)
      item.set_fontname('Gill Sans MT')
    for item in (self.ax.get_yticklabels()):
      item.set_fontsize(22)
      item.set_fontname('Gill Sans MT')
    
    plt.savefig(self.figure_name, dpi = 300, bbox_inches = 'tight', pad_inches = 0.0)
    plt.close()

    
  def plot_annual_option_payments(self, thresh_use_list):
    #this function plots the overall up-front option payments made under each leasing scenario
    bar_width = 0.85
    start_loc = 0
    #loop through each informal leasing scenario to plot all on same plot
    for thresh_use in thresh_use_list:
      #get list of lease facilitators
      all_facilitators = pd.read_csv('results_' + thresh_use + '/buyouts_2_5104055.csv')
      facilitators_unique = all_facilitators['structure'].unique()      
      #load all payments for crop price risk
      option_prices_seller = pd.read_csv('results_' + thresh_use + '/option_payments_sellers.csv', index_col = 0)
      #load all payments for return flow risk
      option_prices_facilitator = pd.read_csv('results_' + thresh_use + '/option_payments_facilitators.csv', index_col = 0)
      #total payments for crop price risk
      total_seller_fee = np.sum(option_prices_seller['annual payment'])
      #group return flow risk payments into payments to lease facilitators and payments to third parties
      total_facilitator_fee = 0.0
      total_third_party_fee = 0.0
      for index, row in option_prices_facilitator.iterrows():
        if index in facilitators_unique:
          total_facilitator_fee += row['annual payment']
        else:
          total_third_party_fee += row['annual payment']
      #plot bar chart
      self.ax.fill_between([start_loc, start_loc + bar_width], np.zeros(2), total_seller_fee/1000 * np.ones(2), facecolor = 'steelblue', edgecolor = 'black', linewidth = 0.5)
      start_loc += 1
      self.ax.fill_between([start_loc, start_loc + bar_width], np.zeros(2), total_facilitator_fee/1000 * np.ones(2), facecolor = 'indianred', edgecolor = 'black', linewidth = 0.5)
      start_loc += 1
      self.ax.fill_between([start_loc, start_loc + bar_width], np.zeros(2), total_third_party_fee/1000 * np.ones(2), facecolor = 'teal', edgecolor = 'black', linewidth = 0.5)
      start_loc += 1.5
    #format plot
    self.ax.set_ylabel('Total Up-front Option Payments ($1000/year)', fontsize = 18, weight = 'bold', fontname = 'Gill Sans MT')  
    self.ax.set_xticks([0.925, 4.425, 7.925, 11.425])
    self.ax.set_xticklabels(['Most Extreme\nDrought\nCBI = 550 tAF', 'Severe\nDrought\nCBI = 600 tAF', 'Moderate\nDrought\nCBI = 650 tAF', 'Mild\nDrought\nCBI = 700 tAF'])
    for item in (self.ax.get_xticklabels()):
      item.set_fontsize(18)
      item.set_fontname('Gill Sans MT')
    for item in (self.ax.get_yticklabels()):
      item.set_fontsize(18)
      item.set_fontname('Gill Sans MT')
    self.ax.set_xlim([0, 13.5])
    self.ax.set_ylim([0, 200])
    legend_elements = [Patch(facecolor='steelblue', edgecolor='black', label='Annual Seller Option Payments', alpha = 1.0),
                        Patch(facecolor='indianred', edgecolor='black', label='Annual Facilitator Option Payments', alpha = 1.0),
                        Patch(facecolor='teal', edgecolor='black', label='Annual Third-Party Option Payments', alpha = 1.0)]
    self.ax.legend(handles=legend_elements, loc='upper left', prop={'family':'Gill Sans MT','weight':'bold','size':22}, framealpha = 1.0, ncol = 1)  
    plt.savefig(self.figure_name, dpi = 300, bbox_inches = 'tight', pad_inches = 0.0)    

    plt.close()
    
  def plot_facilitator_option_payments(self, thresh_use_list):
    #this function plots the overall up-front option payments made under each leasing scenario
    bar_width = 0.85
    start_loc = 0
    #loop through each informal leasing scenario to plot all on same plot
    for thresh_use in thresh_use_list:
      #get list of lease facilitators
      all_facilitators = pd.read_csv('results_' + thresh_use + '/buyouts_2_5104055.csv')
      facilitators_unique = all_facilitators['structure'].unique()      
      #load all payments for crop price risk
      #load all payments for return flow risk
      option_prices_facilitator = pd.read_csv('results_' + thresh_use + '/option_payments_facilitators.csv', index_col = 0)
      #total payments for crop price risk
      #group return flow risk payments into payments to lease facilitators and payments to third parties
      total_facilitator_fee = 0.0
      total_third_party_fee = 0.0
      for index, row in option_prices_facilitator.iterrows():
        total_facilitator_fee += row['annual payment']
      #plot bar chart
      self.ax.fill_between([start_loc, start_loc + bar_width], np.zeros(2), 0.5*total_facilitator_fee/1000 * np.ones(2), facecolor = 'indianred', edgecolor = 'black', linewidth = 0.5, alpha = 0.3)
      self.ax.fill_between([start_loc, start_loc + bar_width], np.zeros(2), 0.5*total_facilitator_fee/1000 * np.ones(2), facecolor = 'none', edgecolor = 'black', linewidth = 0.5, alpha = 1.0)
      start_loc += 1
      self.ax.fill_between([start_loc, start_loc + bar_width], np.zeros(2), total_facilitator_fee/1000 * np.ones(2), facecolor = 'indianred', edgecolor = 'black', linewidth = 0.5, alpha = 0.6)
      self.ax.fill_between([start_loc, start_loc + bar_width], np.zeros(2), total_facilitator_fee/1000 * np.ones(2), facecolor = 'none', edgecolor = 'black', linewidth = 0.5, alpha = 1.0)
      start_loc += 1
      self.ax.fill_between([start_loc, start_loc + bar_width], np.zeros(2), 2.0*total_facilitator_fee/1000 * np.ones(2), facecolor = 'indianred', edgecolor = 'black', linewidth = 0.5, alpha = 0.9)
      self.ax.fill_between([start_loc, start_loc + bar_width], np.zeros(2), 2.0*total_facilitator_fee/1000 * np.ones(2), facecolor = 'none', edgecolor = 'black', linewidth = 0.5, alpha = 1.0)
      start_loc += 1.5
    #format plot
    self.ax.set_ylabel('Total Annual Up-front Option Payments ($1000/year)', fontsize = 18, weight = 'bold', fontname = 'Gill Sans MT')  
    self.ax.set_xticks([0.925, 4.425, 7.925, 11.425])
    self.ax.set_xticklabels(['Most Extreme\nDrought\nCBI = 550 tAF', 'Severe\nDrought\nCBI = 600 tAF', 'Moderate\nDrought\nCBI = 650 tAF', 'Mild\nDrought\nCBI = 700 tAF'])
    for item in (self.ax.get_xticklabels()):
      item.set_fontsize(18)
      item.set_fontname('Gill Sans MT')
    for item in (self.ax.get_yticklabels()):
      item.set_fontsize(18)
      item.set_fontname('Gill Sans MT')
    self.ax.set_xlim([-.15, 13.5])
    self.ax.set_ylim([0, 350])
    legend_elements = [Patch(facecolor='indianred', edgecolor='black', label='5% Return Flow Underestimation', alpha = 0.3),
                        Patch(facecolor='indianred', edgecolor='black', label='10% Return Flow Underestimation', alpha = 0.6),
                        Patch(facecolor='indianred', edgecolor='black', label='20% Return Flow Underestimation', alpha = 0.9)]
    self.ax.legend(handles=legend_elements, loc='upper left', prop={'family':'Gill Sans MT','weight':'bold','size':22}, framealpha = 1.0, ncol = 1)  
    plt.savefig(self.figure_name, dpi = 300, bbox_inches = 'tight', pad_inches = 0.0)    

    plt.close()
    
    
    
    





  
