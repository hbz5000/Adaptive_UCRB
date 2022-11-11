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
      
      #add in line break so peaches ($6000/AF) and grapes ($10000/AF) are on the same plot as annual crops
      if total_cost > 10000.0:
        total_cost = 700.0 + (total_cost - 10000.0)/5.0
      elif total_cost > 5500.0:
        total_cost = 400.0 + (total_cost - 5500.0)/5.0
      
      #acreage irrigated for this crop type      
      total_area = overall_crop_areas[x]
      #plot value per acre foot vs. cumulative acreage, with crops ordered from highest value crop to lowest value crop
      self.ax.fill_between([running_area, running_area + total_area], np.zeros(2), [total_cost, total_cost], facecolor = 'indianred', edgecolor = 'black', linewidth = 2.0)
      running_area += total_area

    #format plot
    self.ax.set_yticks([0.0, 100.0, 200.0, 400.0, 500.0, 700.0, 800.00])
    self.ax.set_yticklabels(['$0', '$100', '$200', '$5500', '$6000', '$10000', '$10500'])
    self.ax.set_ylabel('Cost of Fallowing per AF', fontsize = 42, weight = 'bold', fontname = 'Gill Sans MT')
    self.ax.set_xlabel('UCRB Irrigation (thousand acres)', fontsize = 42, weight = 'bold', fontname = 'Gill Sans MT')
    self.ax.set_ylim([0.0, 850.0])
    self.ax.set_xlim([-2.5, running_area * 1.05])
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

  def plot_available_water(self, tunnel_deliveries, storage_series, control_series, snowpack_data, snow_coefs, tunnel_station, reservoir_station, show_plot = False):
    #figure 3 in the manuscript - water supply index as a function of storage, tunnel exports, and snowpack flow equivalents
    #get index over historical simulation period
    month_index = ['OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP']
    colors_use, colors_use_vals = self.set_plotting_color_palette()
    already_diverted = 0.0
    #historical simulation period is the index of input dataframes
    diversion_vol = []
    storage_vol = []
    snowpack_vol = []
    plot_index = []
    #for each timestep, calculate the index componenets
    counter = 0
    for index, row in tunnel_deliveries.iterrows():
      if index.year >= 1950:
        if index.month == 10:
          already_diverted = 0.0
        already_diverted += tunnel_deliveries.loc[index, tunnel_station] / 1000.0#tunnel exports (taf)
        available_storage = storage_series.loc[index, reservoir_station] / 1000.0#storage (taf)
        control_location = control_series.loc[index, reservoir_station + '_location']#get river call location to determine snow/flow relationship
        #get month to determine snow/flow relationship
        if index.month > 9:
          month_val = month_index[index.month - 10]
        else:
          month_val = month_index[index.month + 2]
        #use regression to estimate flow equivalents from snowpack observation
        if control_location in snow_coefs[month_val]:#use call-specific regressions if the timestep has a 'common' call location
          snowpack_vol.append((snowpack_data.loc[index, 'basinwide_average'] * snow_coefs[month_val][control_location][0] + snow_coefs[month_val][control_location][1])/1000.0)
        else:#if not use a general regresssion
          snowpack_vol.append((snowpack_data.loc[index, 'basinwide_average'] * snow_coefs[month_val]['all'][0] + snow_coefs[month_val]['all'][1])/1000.0)
        diversion_vol.append(already_diverted)
        storage_vol.append(storage_series.loc[index, reservoir_station] / 1000.0)#snowpack flow equivalents (taf)
        plot_index.append(index)
        counter += 1
    #plot each water supply index component
    #storage component
    snowpack_vol = np.asarray(snowpack_vol)
    diversion_vol = np.asarray(diversion_vol)
    storage_vol = np.asarray(storage_vol)
    
    num_timesteps = len(storage_vol)
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
      for x_cnt, thresh in enumerate([550.0, 600.0, 650.0, 700.0]):
        self.ax.plot(plot_index, np.ones(num_timesteps) * thresh, color = colors_use[colors_use_vals[str(int(thresh))]], linewidth = 4.0, linestyle = '--')
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
    
    
  def plot_informal_leases(self, thresh_use_list):
    #figure 4 - annual frequency of informal lease purchases by informal leasing threshold
    
    #set color scheme    
    colors_use, color_use_value = self.set_plotting_color_palette()
    bar_width = 0.85#width of bars on plot (max = 1.0)
    freq_scale = 0.5#maximum plotting frequency (x-axis)
    av_scale = 40.0
    counter = 0#counter for informal leasing scenarios
    for thresh_use in thresh_use_list:
      all_leases = pd.read_csv('results_' + thresh_use + '/purchases_5104055.csv')#get record of additional exports (completed leases)
      all_leases['datetime'] = pd.to_datetime(all_leases['date'])
      lease_hist = []
      for x in range(1950, 2014):
        total_leases = 0.0
        #aggregate total leases to annual scale (data is monthly)
        for index, row in all_leases.iterrows():
          if row['datetime'].year == x:
            total_leases += row['demand'] / 1000.0
        #make list of years with more than 2.5 taf in completed leases
        if total_leases > 0.0:
          lease_hist.append(total_leases)
      #total frequency across the 64 year historical simulation
      freq_lease = float(len(lease_hist)) / 64.0
      ave_lease = np.sum(np.asarray(lease_hist)) / float(len(lease_hist))
      #plot bar of frequency for a given informal leasing scenario
      self.ax.fill_between([-1 * freq_lease / freq_scale, 0.0], [float(counter), float(counter)], [float(counter) + bar_width, float(counter) + bar_width], facecolor = 'indianred', edgecolor = 'black', linewidth = 0.5, alpha = 1.0)
      self.ax.fill_between([0.0, ave_lease/av_scale], [float(counter), float(counter)], [float(counter) + bar_width, float(counter) + bar_width], facecolor = 'steelblue', edgecolor = 'black', linewidth = 0.5, alpha = 1.0)
      counter += 1
    #format plot
    self.ax.set_ylabel('Lease Option Exercise Threshold', fontsize = 24, weight = 'bold', fontname = 'Gill Sans MT', labelpad = 25)
    self.ax.set_xlabel('Lease Option Exercise Rate (% of simulated years)    Average Annual Exercised Lease Options (tAF)  ', fontsize = 18, weight = 'bold', fontname = 'Gill Sans MT')
    self.ax.set_xlim([-1, 1])
    self.ax.set_xticks([-1, -0.5, 0.0, 0.5, 1.0])
    self.ax.set_xticklabels([str(int(freq_scale * 100)) + '%', str(int(freq_scale * 50)) + '%', '0', str(int(av_scale /2.0)) + ' tAF', str(int(av_scale)) + ' tAF'])
    self.ax.set_yticks([bar_width/2.0, 1.0 + bar_width/2.0, 2.0 + bar_width/2.0, 3.0 + bar_width/2.0])
    for y_cnt, thresh_use in enumerate(thresh_use_list):
      self.ax.get_yticklabels()[y_cnt].set_color(colors_use[color_use_value[thresh_use]])

    self.ax.set_yticklabels(['Mild Drought\n(700 tAF CBI)', 'Moderate Drought\n(650 tAF CBI)', 'Severe Drought\n(600 tAF CBI)', 'Extreme Drought\n(550 tAF CBI)'])
    
    for item in (self.ax.get_xticklabels()):
      item.set_fontsize(18)
      item.set_fontname('Gill Sans MT')
    for item in (self.ax.get_yticklabels()):
      item.set_fontsize(22)
      item.set_fontname('Gill Sans MT')
    plt.savefig(self.figure_name, dpi = 300, bbox_inches = 'tight', pad_inches = 0.0)
    plt.close()
    
  def plot_third_party_impacts(self, thresh_use_list):
    #figure 5 - total leases vs. third party impacts and potential compensatory releases for different informal leasing scenarios
    group_spacing = 2.5#total width of each informal lease scenario group
    bar_width = 0.85#width of individual bars
    for x_cnt, folder_name in enumerate(thresh_use_list):#informal lease scenario loop
      #read in aggregate leases, shortfalls, and compensatory releases for each leasing scenario
      third_party_impacts = pd.read_csv('results_' + folder_name + '/total_changes.csv', index_col = 0)
            
      start_val = 0
      total_exports = third_party_impacts.loc['Exports', 'change']#total leases
      e_losses = third_party_impacts.loc['Environment', 'change']#total instream flow shortfall
      con_losses = third_party_impacts.loc['Other', 'change']#total consumptive diversion shortfall
      self.ax.fill_between([x_cnt * group_spacing, x_cnt * group_spacing + bar_width], [0.0, 0.0], [total_exports, total_exports], facecolor = 'goldenrod', edgecolor = 'black', linewidth = 1.5, alpha = 0.8)
      self.ax.fill_between([x_cnt * group_spacing, x_cnt * group_spacing + bar_width], [e_losses, e_losses], [0.0, 0.0], facecolor = 'steelblue', edgecolor = 'black', linewidth = 1.5, alpha = 0.8)
      self.ax.fill_between([x_cnt * group_spacing, x_cnt * group_spacing + bar_width], [con_losses + e_losses,con_losses + e_losses], [e_losses, e_losses], facecolor = 'forestgreen', edgecolor = 'black', linewidth = 1.5, alpha = 0.8)
      comp_release = third_party_impacts.loc['Compensatory', 'change']#compensatory release
      self.ax.fill_between([x_cnt * group_spacing + 1, x_cnt * group_spacing + bar_width + 1], [0.0, 0.0], [total_exports - comp_release, total_exports - comp_release], facecolor = 'goldenrod', edgecolor = 'black', linewidth = 1.5, alpha = 0.8)
      self.ax.fill_between([x_cnt * group_spacing + 1, x_cnt * group_spacing + bar_width + 1], [total_exports - comp_release, total_exports - comp_release], [total_exports, total_exports], facecolor = 'indianred', edgecolor = 'black', linewidth = 1.5, alpha = 0.8)

    #format plot
    self.ax.plot([-0.15, x_cnt * group_spacing + bar_width + start_val + 0.15], [0.0, 0.0], linewidth = 1.5, color = 'black')
    self.ax.set_xlim([-0.15, 10])
    self.ax.set_ylim([-250, 475])
    legend_elements = [Patch(facecolor='goldenrod', edgecolor='black', label='C-BT Exports', alpha = 1.0),
                       Patch(facecolor='indianred', edgecolor='black', label='Compensatory \nReleases', alpha = 1.0),
                       Patch(facecolor='steelblue', edgecolor='black', label='Environmental\nShortfalls', alpha = 1.0),
                       Patch(facecolor='forestgreen', edgecolor='black', label='Consumptive\nShortfalls', alpha = 1.0)]
    
    self.ax.legend(handles=legend_elements, loc='upper left', prop={'family':'Gill Sans MT','weight':'bold','size':20}, framealpha = 1.0, ncol = 1)  
    self.ax.set_ylabel('Total Change in Diversions (tAF)', fontsize = 28, weight = 'bold', fontname = 'Gill Sans MT')
    self.ax.set_xticks([0.925, 3.425, 5.925, 8.425])
    self.ax.set_xticklabels(['Leases triggered by:\nExtreme Drought', 'Leases triggered by:\nSevere Drought', 'Leases triggered by:\nModerate Drought', 'Leases triggered by:\nMild Drought'])
            
    for item in (self.ax.get_xticklabels()):
      item.set_fontsize(18)
      item.set_fontname('Gill Sans MT')
    for item in (self.ax.get_yticklabels()):
      item.set_fontsize(18)
    plt.savefig(self.figure_name, dpi = 300, bbox_inches = 'tight', pad_inches = 0.0)
    plt.close()
    
  def plot_informal_price(self, thresh_use, plot_facilitators = False):
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
    volume_list = np.asarray(volume_list)
    year_list = np.asarray(year_list)
    #sort the years in which informal leases are purchased by 
    #the transaction cost (i.e., annual average facilitated demand per leased demand)
    sorted_cost_index = np.argsort(cost_list)
    
    #set up an array to track the cumulative volume leased, moving from lowest-transaction cost year to highest
    cumulative_volume = np.zeros(len(volume_list) + 1)
    #set up an array to track the marginal transaction cost per volume of informal lease, moving from lowest-transaction cost yeaer to highest
    marginal_cost = np.zeros(len(volume_list) + 1)
    #total savings from informal transfers for all purchases based on a $200/AF (minimum) and $360/AF estimate of formal legal transaction costs
    cumulative_savings = np.zeros(4)
    
    unique_facilitator_list = np.zeros(len(volume_list) + 1)#cumulative number of unique facilitators used, moving from lowest-transaction cost year to highest
    list_of_facilitators = []#running list of unique facilitators
    for xxx in range(1, len(cumulative_volume)):
      #cumulative volume of informal leases purchased at a transaction cost lower than marginal_cost[xxx]
      cumulative_volume[xxx] = cumulative_volume[xxx-1] + volume_list[sorted_cost_index[xxx-1]]
      #transaction cost of informal leases purchased in year - indexed to sorted position XXX
      marginal_cost[xxx] = cost_list[sorted_cost_index[xxx-1]]
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
    baseline = np.zeros(len(cumulative_volume))#for calculating the marginal transaction cost curve with different levels of 'facilitator fees'
    price_use = 5.#facilitator fees calculated in $5/af increments
    total_inc = 3
    color_list = ['beige', 'goldenrod', 'indianred']
    legend_elements = []
    self.ax.fill_between([0.0, np.max(cumulative_volume)], [200.0, 200.0], [360.0, 360.0], facecolor = 'maroon', edgecolor = 'black', linewidth = 0.5, alpha = 0.9)
    for inc_no in range(0, total_inc):
      self.ax.fill_between(cumulative_volume, baseline, baseline + marginal_cost * price_use, color = color_list[inc_no], edgecolor = 'black', linewidth = 1.0, alpha = 1.0)
      baseline += marginal_cost * price_use
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
    self.ax.set_ylim([0, 360.0])
    self.ax.set_xlim([0, np.max(cumulative_volume)])
    self.ax.text(np.max(cumulative_volume) / 2.0, 280.0, 'Range of Transaction Costs,\nFormal Legal System', fontsize = 64, weight = 'bold', fontname = 'Gill Sans MT', ha = 'center', va = 'center', color = 'white')
    self.ax.set_ylabel('Transaction Costs ($/AF)', fontsize = 32, weight = 'bold', fontname = 'Gill Sans MT')
    self.ax.set_xlabel('Cumulative Lease Volume (tAF)', fontsize = 32, weight = 'bold', fontname = 'Gill Sans MT')
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
    
    
    
    
    





  
