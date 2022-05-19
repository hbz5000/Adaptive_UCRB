from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
import collections as cl
import pandas as pd
import geopandas as gpd
from datetime import datetime
import json
import rights
from structure import Structure
from reservoir import Reservoir
from plotter import Plotter
from scipy.optimize import curve_fit
import scipy.stats as stats
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap
import matplotlib.dates as mdates
from matplotlib.patches import Patch


class Basin():
  # __slots__ = ["T", "key", "name",  "turnback_use", "contract_list", "turnout_list", "in_leiu_banking", "leiu_recovery",
  #              "in_district_direct_recharge", "in_district_storage", "recovery_fraction", "surface_water_sa",
  #              "must_fill", "seasonal_connection", "seepage", "inleiuhaircut", "recharge_decline", "project_contract",
  #              "rights", "service", "crop_list", "zone", "acreage", "MDD", "urban_profile", "participant_list",
  #              "inleiucap", "irrdemand", "deliveries", "contract_list_all", "non_contract_delivery_list",
  #              "current_balance", "paper_balance", "turnback_pool", "projected_supply", "carryover", "recharge_carryover",
  #              "delivery_carryover", "contract_carryover_list", "dynamic_recharge_cap",
  #              "annual_supplies", "daily_supplies_full", "annualdemand", "dailydemand", "annual_pumping", "use_recharge",
  #              "use_recovery", "extra_leiu_recovery", "max_recovery", "max_leiu_exchange", "direct_recovery_delivery",
  #              "pre_flood_demand", "recharge_rate", "thismonthuse", "monthusecounter", "monthemptycounter",
  #              "current_recharge_storage", "private_fraction", "has_private", "has_pesticide", "has_pmp", "recovery_use",
  #              "inleiubanked", "contract_exchange", "leiu_additional_supplies", "bank_deliveries", "tot_leiu_recovery_use",
  #              "direct_storage", "bank_timeseries", "recharge_rate_series", "use_recovery",
  #              "leiu_trade_cap", "loss_rate", "leiu_ownership", "private_acreage", "monthly_demand", 'reservoir_contract',
  #              'current_requested', 'monthlydemand', 'carryover_rights', 'initial_table_a', 'demand_days', 
  #              'total_banked_storage', 'min_direct_recovery', 'max_leiu_recharge', 'dailydemand_start', 'turnback_sales', 
  #              'turnback_purchases', 'annual_private_pumping', 'private_delivery', 'max_direct_recharge', 'irrseasondemand',
  #              'private_demand', 'regression_percent', 'pumping', 'demand_auto_errors', 'ytd_pumping', 
  #              'regression_errors_timeseries', 'hist_demand_dict', 'ytd_pumping_int', 'hist_pumping', 'regression_errors', 
  #              'delivery_percent_coefficient', 'regression_annual', 'last_days_demand_regression_error', 
  #              'k_close_wateryear', 'iter_count', 'contract_list_length', 'days_to_fill', 'recovery_capacity_remain',
  #              'acreage_by_year', 'delivery_location_list', 'number_years', 'table_a_request']

  def __iter__(self):
    self.iter_count = 0
    return self
  
  def __next__(self):
    if self.iter_count == 0:
      self.iter_count += 1
      return self
    else:
      raise StopIteration

  def __len__(self):
    return 1
                     
  is_Canal = 0
  is_District = 1
  is_Private = 0
  is_Waterbank = 0
  is_Reservoir = 0
  

  def __init__(self, input_data_dictionary):

    extended_table = gpd.read_file(input_data_dictionary['hydrography'], layer = 'WBDHU4')
    extended_table8 = gpd.read_file(input_data_dictionary['hydrography'], layer = 'WBDHU8')
    total_basin = pd.DataFrame()
    for huc4_watershed in input_data_dictionary['HUC4']:
      this_basin = extended_table[extended_table['HUC4'] == huc4_watershed]
      total_basin = pd.concat([total_basin, this_basin], ignore_index = True)        
    this_basin = gpd.GeoDataFrame(this_basin, crs = extended_table.crs, geometry = total_basin.geometry)
    self.huc_8_list = input_data_dictionary['HUC8']
    #self.basin_huc8 = self.clean_join(extended_table8, this_basin, 4326, 'inner', 'within')
    
    self.basin_snowpack = {}
    for huc8_basin in self.huc_8_list:
      self.basin_snowpack[huc8_basin] = pd.read_csv(input_data_dictionary['snow'] + huc8_basin + '.csv', index_col = 0)
      self.basin_snowpack[huc8_basin].index = pd.to_datetime(self.basin_snowpack[huc8_basin].index)
      new_index = []
      for x in self.basin_snowpack[huc8_basin].index:
        new_index.append(datetime(x.year, x.month, 1, 0, 0))
      
    self.basin_structures = gpd.read_file(input_data_dictionary['structures'])
    self.basin_structures = self.basin_structures.to_crs(epsg = 4326)
    
    self.structures_objects = {}
    self.structures_list = {}
    self.structures_list['unknown'] = []
    self.structures_list['total'] = []
    #for huc8_basin in self.huc_8_list:
      #this_watershed = self.basin_huc8[self.basin_huc8['HUC8'] == huc8_basin]
      #this_watershed_structures = self.clean_join(self.basin_structures, this_watershed, 4326, 'inner', 'within')
      #self.structures_list[huc8_basin] = []
      #for index_s, row_s in this_watershed_structures.iterrows():
        #self.structures_list[huc8_basin].append(str(row_s['WDID']))
        #self.structures_list['total'].append(str(row_s['WDID']))
        #self.structures_objects[str(row_s['WDID'])] = Structure(str(row_s['WDID']), huc8_basin)

  def clean_join(self, gdf1, gdf2, crs_int, howstring, opstring):
    gdf1 = gdf1.to_crs(epsg = crs_int)
    gdf2 = gdf2.to_crs(epsg = crs_int)
    for column in ['index_left', 'index_right']:
      try:
        gdf1.drop(column, axis = 1, inplace = True)
      except KeyError:
        pass
      try:
        gdf2.drop(column, axis = 1, inplace = True)
      except KeyError:
        pass

    joined_gdf = gpd.sjoin(gdf1, gdf2, how = howstring, op = opstring)
        
    return joined_gdf
    
    
    
  def combine_rights_data(self, structure_rights_name, structure_name, structure_rights_priority, structure_rights_decree, reservoir_rights_name, reservoir_name, reservoir_rights_priority, reservoir_rights_decree, instream_rights_name, instream_rights_structure_name, instream_rights_priority, instream_rights_decree):
    structure_rights_priority.extend(reservoir_rights_priority)
    structure_rights_name.extend(reservoir_rights_name)
    structure_name.extend(reservoir_name)
    structure_rights_decree.extend(reservoir_rights_decree)

    structure_rights_priority.extend(instream_rights_priority)
    structure_rights_name.extend(instream_rights_name)
    structure_name.extend(instream_rights_structure_name)
    structure_rights_decree.extend(instream_rights_decree)
  
    priority_order = np.argsort(np.asarray(structure_rights_priority))
    self.rights_stack_structure_ids = []
    self.rights_decree_stack = []
    self.rights_stack_ids = []
    self.rights_stack_priorities = []
    for stack_order in range(0, len(priority_order)):      
      self.rights_stack_ids.append(structure_rights_name[priority_order[stack_order]])
      self.rights_stack_structure_ids.append(structure_name[priority_order[stack_order]])
      self.rights_decree_stack.append(structure_rights_decree[priority_order[stack_order]])
      self.rights_stack_priorities.append(structure_rights_priority[priority_order[stack_order]])

  def create_reservoir(self, reservoir_name, idnum, capacity):
    self.structures_objects[idnum] = Reservoir(reservoir_name, idnum, capacity)
    self.reservoir_list.append(idnum)
    
  def update_structure_demand_delivery(self, new_demands, new_deliveries, monthly_max, date_use):

    for this_structure in new_demands:
      if this_structure in monthly_max:
        self.structures_objects[this_structure].adaptive_monthly_demand.loc[date_use, 'demand'] = min(new_demands[this_structure], monthly_max[this_structure][date_use.month - 1])
      else:
        self.structures_objects[this_structure].adaptive_monthly_demand.loc[date_use, 'demand'] = new_demands[this_structure] * 1.0
      self.structures_objects[this_structure].update_demand_rights(date_use)
    for this_structure in new_deliveries:
      self.structures_objects[this_structure].adaptive_monthly_deliveries.loc[date_use, 'deliveries'] = new_deliveries[this_structure] * 1.0
      self.structures_objects[this_structure].update_delivery_rights(date_use)
      
  def update_structure_plan_flows(self, new_plan_flows, date_use):
    station_used_list = []
    for station_start in self.plan_flows_list:
      for station_use in self.plan_flows_list[station_start]:
        if station_use in station_used_list:
          self.structures_objects[station_use].routed_plans.loc[date_use, 'plan'] += new_plan_flows[station_start] * 1.0
        else:
          self.structures_objects[station_use].routed_plans.loc[date_use, 'plan'] = new_plan_flows[station_start] * 1.0
          station_used_list.append(station_use)
      
  def update_structure_outflows(self, new_releases, date_use):
    for this_structure in self.structures_objects:
      if this_structure + '_outflow' in new_releases:
        self.structures_objects[this_structure].adaptive_monthly_outflows.loc[date_use, 'outflows'] = new_releases[this_structure + '_outflow'] * 1.0
      if this_structure + '_location' in new_releases:
        self.structures_objects[this_structure].adaptive_monthly_control.loc[date_use, 'location'] = str(new_releases[this_structure + '_location'])
  
  def update_structure_storage(self, structure_storage, datetime_val):
    for this_structure in self.structures_objects:
      if this_structure in structure_storage:
        self.structures_objects[this_structure].adaptive_reservoir_timeseries.loc[datetime_val, this_structure] = structure_storage[this_structure] * 1.0
      if this_structure + '_diversions' in structure_storage:
        self.structures_objects[this_structure].adaptive_reservoir_timeseries.loc[datetime_val, this_structure + '_diversions'] = structure_storage[this_structure + '_diversions'] * 1.0
  
  def adjust_structure_deliveries(self):
    monthly_deliveries = {}
    for structure_name in self.structures_objects:
      if np.max(self.structures_objects[structure_name].historical_monthly_demand['demand']) > 900000.0:
        monthly_deliveries[structure_name] = np.zeros(12)
        for index, row in self.structures_objects[structure_name].historical_monthly_deliveries.iterrows():
          monthly_deliveries[structure_name][index.month - 1] = max(row['deliveries'], monthly_deliveries[structure_name][index.month - 1])
        for datetime_val in self.structures_objects[structure_name].historical_monthly_demand.index:
          self.structures_objects[structure_name].historical_monthly_demand.loc[datetime_val, 'demand'] = monthly_deliveries[structure_name][datetime_val.month - 1] * 1.0
          self.structures_objects[structure_name].adaptive_monthly_demand.loc[datetime_val, 'demand'] = monthly_deliveries[structure_name][datetime_val.month - 1] * 1.0

    return monthly_deliveries
          
  def set_structure_demands(self, structure_demands, structure_demands_adaptive = 'none', use_rights = True):
    len_hist_demands = 0
    for structure_name in structure_demands:
      if structure_name in self.structures_objects:
        self.structures_objects[structure_name].historical_monthly_demand = pd.DataFrame(structure_demands[structure_name].values, index = structure_demands.index, columns = ['demand',])
        try:
          self.structures_objects[structure_name].adaptive_monthly_demand = pd.DataFrame(structure_demands_adaptive[structure_name].values, index = structure_demands_adaptive.index, columns = ['demand',])
        except:
          self.structures_objects[structure_name].adaptive_monthly_demand = self.structures_objects[structure_name].historical_monthly_demand.copy(deep = True)
        
        len_hist_demands = len(structure_demands[structure_name])
      else:        
        self.structures_objects[structure_name] = Structure(structure_name, 'unknown')
        self.structures_objects[structure_name].historical_monthly_demand = pd.DataFrame(structure_demands[structure_name].values, index = structure_demands.index, columns = ['demand',])
        try:
          self.structures_objects[structure_name].adaptive_monthly_demand = pd.DataFrame(structure_demands_adaptive[structure_name].values, index = structure_demands_adaptive.index, columns = ['demand',])
        except:
          self.structures_objects[structure_name].adaptive_monthly_demand = self.structures_objects[structure_name].historical_monthly_demand.copy(deep = True)
        
    for structure_name in self.structures_objects:
      try:
        len_hist_demands = len(self.structures_objects[structure_name].historical_monthly_demand)
      except:
        self.structures_objects[structure_name].historical_monthly_demand = pd.DataFrame(np.zeros(len(structure_demands.index)), index = structure_demands.index, columns = ['demand',])
        self.structures_objects[structure_name].adaptive_monthly_demand = self.structures_objects[structure_name].historical_monthly_demand.copy(deep = True)
      if use_rights:
        self.structures_objects[structure_name].make_sorted_rights_list()
        if isinstance(structure_demands_adaptive, pd.DataFrame):
          self.structures_objects[structure_name].use_adaptive = True
        self.structures_objects[structure_name].assign_demand_rights()

  def set_structure_deliveries(self, structure_deliveries, structure_deliveries_adaptive = 'none', use_rights = True):
    len_hist_deliveries = 0
    for structure_name in structure_deliveries:
      if structure_name in self.structures_objects:
        self.structures_objects[structure_name].historical_monthly_deliveries = pd.DataFrame(structure_deliveries[structure_name].values, index = structure_deliveries.index, columns = ['deliveries',])
        try:
          self.structures_objects[structure_name].adaptive_monthly_deliveries = pd.DataFrame(structure_deliveries_adaptive[structure_name].values, index = structure_deliveries_adaptive.index, columns = ['deliveries',])
        except:        
          self.structures_objects[structure_name].adaptive_monthly_deliveries = self.structures_objects[structure_name].historical_monthly_deliveries.copy(deep = True)
        
      else:
        self.structures_objects[structure_name] = Structure(structure_name, 'unknown')
        self.structures_objects[structure_name].historical_monthly_demand = pd.DataFrame(np.zeros(len(structure_deliveries.index)), index = structure_deliveries.index, columns = ['demand',])
        self.structures_objects[structure_name].adaptive_monthly_demand = pd.DataFrame(np.zeros(len(structure_deliveries.index)), index = structure_deliveries.index, columns = ['demand',])
        self.structures_objects[structure_name].make_sorted_rights_list()
        self.structures_objects[structure_name].historical_monthly_deliveries = pd.DataFrame(structure_deliveries[structure_name].values, index = structure_deliveries.index, columns = ['deliveries',])
        try:
          self.structures_objects[structure_name].adaptive_monthly_deliveries = pd.DataFrame(structure_deliveries_adaptive[structure_name].values, index = structure_deliveries_adaptive.index, columns = ['deliveries',])
        except:
          self.structures_objects[structure_name].adaptive_monthly_deliveries = self.structures_objects[structure_name].historical_monthly_deliveries.copy(deep = True)
    for structure_name in self.structures_objects:
      no_xdd = False
      try:
        len_hist_demands = len(self.structures_objects[structure_name].historical_monthly_deliveries)
      except:
        no_xdd = True
      if no_xdd:
        self.structures_objects[structure_name].historical_monthly_deliveries = pd.DataFrame(np.zeros(len(structure_deliveries.index)), index = structure_deliveries.index, columns = ['deliveries',])
        self.structures_objects[structure_name].adaptive_monthly_deliveries = self.structures_objects[structure_name].historical_monthly_deliveries.copy(deep = True)
      if isinstance(structure_deliveries_adaptive, pd.DataFrame):
        self.structures_objects[structure_name].use_adaptive = True
      if use_rights:
        self.structures_objects[structure_name].assign_delivery_rights()
      
  def find_annual_change_by_wyt(self, start_year, end_year, structure_print_list, iteration_no):
    
    column_name_df = []
    max_change_other = 0
    max_change_mf = 0
    max_change_exp = 0
    for x in structure_print_list:
      column_name_df.append(x + '_baseline')
      column_name_df.append(x + '_reoperation')
      
    delivery_scenarios = pd.DataFrame(columns = column_name_df, index = np.arange(start_year, end_year+1))
    snowpack_vals = np.zeros(end_year - start_year + 1)
    for yearnum in range(start_year, end_year + 1):
      datetime_val = datetime(yearnum, 9, 1, 0, 0)
      snowpack_vals[yearnum-start_year] = self.basin_snowpack['14010001'].loc[datetime_val, 'basinwide_average']

    index_sort = np.argsort(snowpack_vals)
    for structure_name in self.structures_objects:
      annual_deliveries = []
      total_annual_deliveries = 0.0
      for index, row in self.structures_objects[structure_name].historical_monthly_deliveries.iterrows():
        if index.year >= start_year and index.year <= end_year:
          if index.month == 4 and index.year > start_year:
            annual_deliveries.append(total_annual_deliveries)
            total_annual_deliveries = 0.0
          total_annual_deliveries += row['deliveries']/1000.0
      annual_deliveries.append(total_annual_deliveries) 
      annual_deliveries = np.asarray(annual_deliveries)
      
      annual_deliveries2 = []
      total_annual_deliveries = 0.0
      index_counter = 0
      for index, row in self.structures_objects[structure_name].adaptive_monthly_deliveries.iterrows():
        if index.year >= start_year and index.year <= end_year:
          if index.month == 4 and index.year > start_year:
            annual_deliveries2.append(total_annual_deliveries)
            total_annual_deliveries = 0.0
          total_annual_deliveries += row['deliveries']/1000.0
      annual_deliveries2.append(total_annual_deliveries)        
      annual_deliveries2 = np.asarray(annual_deliveries2)

      annual_demands = []
      self.structures_objects[structure_name].historical_monthly_demand.index = pd.to_datetime(self.structures_objects[structure_name].historical_monthly_demand.index)
      total_annual_deliveries = 0.0
      for index, row in self.structures_objects[structure_name].historical_monthly_demand.iterrows():
        if index.year >= start_year and index.year <= end_year:
          if index.month == 4 and index.year > start_year:
            annual_demands.append(total_annual_deliveries)
            total_annual_deliveries = 0.0
          total_annual_deliveries += row['demand']/1000.0        
      annual_demands.append(total_annual_deliveries)        
      annual_demands = np.asarray(annual_demands)
      
      delivery_change = annual_deliveries2 - annual_deliveries
      if 'Irrigation' in self.structures_objects[structure_name].structure_types:
        change_years = delivery_change < 0.0
        for x in range(0,  len(annual_deliveries2)-10):
          num_per = np.sum(change_years[x:(x+10)])
          max_change_other = max(max_change_other, num_per)
      elif 'Minimum Flow' in self.structures_objects[structure_name].structure_types:        
        change_years = delivery_change < 0.0
        for x in range(0,  len(annual_deliveries2)-10):
          num_per = np.sum(change_years[x:(x+10)])
          max_change_mf = max(max_change_mf, num_per)
      elif 'Export' in self.structures_objects[structure_name].structure_types:
        change_years = delivery_change > 0.0
        for x in range(0,  len(annual_deliveries2)-10):
          num_per = np.sum(change_years[x:(x+10)])
          max_change_exp = max(max_change_exp, num_per)
      elif 'Municipal' in self.structures_objects[structure_name].structure_types:
        change_years = delivery_change < 0.0
        for x in range(0,  len(annual_deliveries2)-10):
          num_per = np.sum(change_years[x:(x+10)])
          max_change_other = max(max_change_other, num_per)


      if np.sum(annual_demands) > 0.0:
        self.structures_objects[structure_name].baseline_filled = np.sum(annual_deliveries) / np.sum(annual_demands)
      elif np.sum(annual_deliveries) > 0.0:
        self.structures_objects[structure_name].baseline_filled = np.sum(annual_deliveries) / (float(len(annual_deliveries)) * np.max(annual_deliveries))
      else:
        self.structures_objects[structure_name].baseline_filled = 0.0

      if structure_name in structure_print_list:
        delivery_scenarios[structure_name + '_baseline'] = annual_deliveries
        delivery_scenarios[structure_name + '_reoperation'] = annual_deliveries2
      wet_years = []
      normal_years = []
      dry_years = []
      num_vals = len(index_sort) / 3
      counter = 0
      counter_type = 0
      for x in range(0, len(index_sort)):
        index_use = index_sort[x]
        if counter_type == 0:
          dry_years.append(annual_deliveries2[index_use] - annual_deliveries[index_use])
        elif counter_type == 1:
          normal_years.append(annual_deliveries2[index_use] - annual_deliveries[index_use])
        else:
          wet_years.append(annual_deliveries2[index_use] - annual_deliveries[index_use])
        counter += 1
        if counter > num_vals:
          counter = 0
          counter_type += 1
      self.structures_objects[structure_name].average_change = {}
      self.structures_objects[structure_name].average_change['wet'] = np.mean(wet_years)
      self.structures_objects[structure_name].average_change['normal'] = np.mean(normal_years)
      self.structures_objects[structure_name].average_change['dry'] = np.mean(dry_years)
      self.structures_objects[structure_name].average_change['2002'] = annual_deliveries2[2002-start_year] - annual_deliveries[2002-start_year]
      self.structures_objects[structure_name].average_change['1955'] = annual_deliveries2[1955-start_year] - annual_deliveries[1955-start_year]
      self.structures_objects[structure_name].average_change['1977'] = annual_deliveries2[1977-start_year] - annual_deliveries[1977-start_year]
      self.structures_objects[structure_name].average_change['all'] = np.sum(annual_deliveries2)- np.sum(annual_deliveries)
      
    delivery_scenarios.to_csv('total_export_deliveries.csv')
    frequency_use = pd.DataFrame(np.asarray([max_change_exp, max_change_mf, max_change_other]))
    frequency_use.to_csv('freq_changes_' + str(iteration_no) + '.csv')

  def find_station_revenues(self, station_id, et_requirements, marginal_net_benefits, irrigation_ucrb, aggregated_diversions, year_start, year_end):
    irrigation_structures = list(irrigation_ucrb['SW_WDID1'].astype(str))
    month_name_list = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    water_demand = []
    values_list = []
    if station_id in irrigation_structures:
      this_irrigated_area = irrigation_ucrb[irrigation_ucrb['SW_WDID1'] == station_id]
    elif station_id in aggregated_diversions:
      this_irrigated_area = gpd.GeoDataFrame()
      for ind_structure in aggregated_diversions[station_id]:
        if ind_structure in irrigation_structures:
          this_irrigated_area_int = irrigation_ucrb[irrigation_ucrb['SW_WDID1'] == ind_structure]
          this_irrigated_area = pd.concat([this_irrigated_area, this_irrigated_area_int])
    if station_id == '7200646':
      station_id = '7200646_I'
    for index_ia, row_ia in this_irrigated_area.iterrows():
      water_demand.append(row_ia['ACRES'] * et_requirements[row_ia['CROP_TYPE']] /12.0)
      values_list.append(marginal_net_benefits[row_ia['CROP_TYPE']] * 12.0 / et_requirements[row_ia['CROP_TYPE']])
    water_demand = np.asarray(water_demand)
    values_list = np.asarray(values_list)
    sorted_index = np.argsort(values_list)
    sorted_demand = water_demand[sorted_index]
    sorted_values = values_list[sorted_index]
    total_water_consumption = 0.0
    total_annual_revenue = np.zeros(year_end - year_start + 1)
    for index, row in self.structures_objects[station_id].adaptive_monthly_deliveries.iterrows():
      month_id = month_name_list[index.month - 1]
      total_demand = self.structures_objects[station_id].historical_monthly_demand.loc[index, 'demand']
      
      total_water_consumption += total_demand - row['deliveries'] #* (1.0 - self.structures_objects[station_id].return_fraction.loc[month_id, station_id])
      print(month_id, end = " ")
      print(total_water_consumption, end = " ")
      print(total_demand, end = " ")
      print(row['deliveries'])
      if month_id == 'SEP':
        if index.year >= year_start and index.year <= year_end:
          for mnb, etc in zip(sorted_values, sorted_demand):
            total_annual_revenue[index.year - year_start] += mnb * min(etc, total_water_consumption)
            total_water_consumption = max(total_water_consumption - etc, 0.0)
            print(index.year, end = " ")
            print(mnb, end = " ")
            print(etc, end = " ")
            print(total_annual_revenue[index.year - year_start], end = " ")
            print(total_water_consumption)
            if total_water_consumption == 0.0:
              break
          if total_water_consumption > 0.0:
            total_annual_revenue[index.year - year_start] += sorted_values[-1] * total_water_consumption
        total_water_consumption = 0.0  
    
    return total_annual_revenue
   
  def find_informal_water_price(self, structure_delivery, total_monthly_deliveries, structure_purchases, structure_buyouts, purchase_multiplier, buyout_charge):
    total_deliveries = pd.read_csv('total_export_deliveries.csv')
   
    structure_buyouts['datetime'] = pd.to_datetime(structure_buyouts['date'])
    buyout_years = np.zeros(len(structure_buyouts.index))
    buyout_months = np.zeros(len(structure_buyouts.index))
    counter = 0
    for index, row in structure_buyouts.iterrows():
      buyout_years[counter] = row['datetime'].year
      buyout_months[counter] = row['datetime'].month
      counter += 1
    structure_buyouts['yearnum'] = buyout_years  
    structure_buyouts['monthnum'] = buyout_months  
    structure_purchases = structure_purchases.drop_duplicates()
    structure_purchases['datetime'] = pd.to_datetime(structure_purchases['date'])
    purchase_years = np.zeros(len(structure_purchases.index))
    purchase_months = np.zeros(len(structure_purchases.index))
    counter = 0
    for index, row in structure_purchases.iterrows():
      purchase_years[counter] = row['datetime'].year
      purchase_months[counter] = row['datetime'].month
      counter += 1
    structure_purchases['yearnum'] = purchase_years
    structure_purchases['monthnum'] = purchase_months
    counter = 1950
    cost_per_af = []
    total_export = []
    #for index, row in total_deliveries.iterrows():
    for index, row in total_monthly_deliveries.iterrows():
      total_increase = row[structure_delivery + '_res_diversion']
      #total_increase = (row[structure_delivery + '_reoperation'] - row[structure_delivery + '_baseline']) * 1000.0
      total_cost = 0.0
      if total_increase > 0.0:
        this_year_purchases = structure_purchases[np.logical_and(structure_purchases['yearnum'] == index.year, structure_purchases['monthnum'] == index.month)]
        this_year_buyouts = structure_buyouts[np.logical_and(structure_buyouts['yearnum'] == index.year, structure_buyouts['monthnum'] == index.month)]
        this_year_buyouts = this_year_buyouts.drop_duplicates(subset = 'structure')
        for index, row in this_year_purchases.iterrows():          
          total_cost += row['demand'] * row['consumptive'] * self.structures_objects[row['structure']].purchase_price * purchase_multiplier
        for index, row in this_year_buyouts.iterrows():          
          total_cost += row['demand_purchase'] * buyout_charge
        cost_per_af.append(total_cost/total_increase)
        total_export.append(total_increase)
      counter += 1
    
    return np.asarray(cost_per_af), np.asarray(total_export)
    
  def set_structure_outflows(self, structure_outflows):
    for structure_name in self.structures_objects:
      if structure_name + '_outflow' in structure_outflows:
        self.structures_objects[structure_name].historical_monthly_outflows = pd.DataFrame(structure_outflows[structure_name + '_outflow'].values, index = structure_outflows.index, columns = ['outflows',])
        self.structures_objects[structure_name].adaptive_monthly_outflows = self.structures_objects[structure_name].historical_monthly_outflows.copy(deep = True)
      else:
        self.structures_objects[structure_name].historical_monthly_outflows = pd.DataFrame(np.ones(len(structure_outflows.index))*(-999.0), index = structure_outflows.index, columns = ['outflows',])
        self.structures_objects[structure_name].adaptive_monthly_outflows = pd.DataFrame(np.ones(len(structure_outflows.index))*(-999.0), index = structure_outflows.index, columns = ['outflows',])
      if structure_name + '_location' in structure_outflows:
        self.structures_objects[structure_name].historical_monthly_control = pd.DataFrame(list(structure_outflows[structure_name + '_location']), index = structure_outflows.index, columns = ['location',])
        self.structures_objects[structure_name].adaptive_monthly_control = self.structures_objects[structure_name].historical_monthly_control.copy(deep = True)
      else:
        self.structures_objects[structure_name].historical_monthly_control = pd.DataFrame(np.ones(len(structure_outflows.index))*(-999.0), index = structure_outflows.index, columns = ['location',])
        self.structures_objects[structure_name].adaptive_monthly_control = pd.DataFrame(np.ones(len(structure_outflows.index))*(-999.0), index = structure_outflows.index, columns = ['location',])
    
  def set_return_fractions(self, structure_returns):
    month_name_list = ['OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP']
    for structure_name in self.structures_objects:
      self.structures_objects[structure_name].return_fraction = pd.DataFrame(np.zeros(len(month_name_list)), index = month_name_list, columns = [structure_name,])
    for structure_name in structure_returns:
      if structure_name in self.structures_objects:
        self.structures_objects[structure_name].return_fraction = pd.DataFrame(structure_returns[structure_name], index = month_name_list, columns = [structure_name,])
  def set_plan_flows_list(self, downstream_data, start_point_list,  end_point):
    station_id_column_length = 12
    station_id_column_start = 0
    downstream_station_id_column_start = 36
    downstream_station_id_column_end = 48
    month_name_list = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    self.plan_flows_list = {}
    for j in range(0,len(downstream_data)):
      if downstream_data[j][0] != '#':
        first_line = int(j * 1)
        break
    for j in range(first_line, len(downstream_data)):
      station_id = str(downstream_data[j][station_id_column_start:station_id_column_length].strip())
      if station_id == 'end_point':
        break
      if station_id in start_point_list:
        self.plan_flows_list[station_id] = []
      for x in self.plan_flows_list:
        self.plan_flows_list[x].append(station_id)
  def set_plan_flows_2(self, plan_flows):
    for station_name in self.structures_objects:
      self.structures_objects[station_name].routed_plans = pd.DataFrame(index = plan_flows.index, columns = ['plan',])
      self.structures_objects[station_name].routed_plans['plan'] = np.zeros(len(plan_flows.index))
    for index, row in plan_flows.iterrows():
      for station_start in self.plan_flows_list:
        for station_use in self.plan_flows_list[station_start]:
          self.structures_objects[station_use].routed_plans.loc[index, 'plan'] += row[station_start] * 1.0
          
  def set_plan_flows(self, plan_flows, downstream_data):
    station_id_column_length = 12
    station_id_column_start = 0
    downstream_station_id_column_start = 36
    downstream_station_id_column_end = 48
    month_name_list = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    for j in range(0,len(downstream_data)):
      if downstream_data[j][0] != '#':
        first_line = int(j * 1)
        break
    for station_name in self.structures_objects:
      self.structures_objects[station_name].routed_plans = pd.DataFrame(index = plan_flows.index, columns = ['plan',])
      
    for index, row in plan_flows.iterrows():
      month_id = month_name_list[index.month - 1]
      running_plan_flows = 0.0
      for j in range(first_line, len(downstream_data)):
        station_id = str(downstream_data[j][station_id_column_start:station_id_column_length].strip())
        if station_id + '_diversion' in plan_flows.columns:
          running_plan_flows -= plan_flows.loc[index, station_id + '_diversion']
        if station_id + '_release' in plan_flows.columns:
          running_plan_flows += plan_flows.loc[index, station_id + '_release']
        self.structures_objects[station_id].routed_plans.loc[index, 'plan'] = running_plan_flows * 1.0
        
      
  def set_structure_types(self, aggregated_diversions, irrigation_shapefile, ditches_shapefile, structures_shapefile, marginal_net_benefits, et_requirements):
    irrigation_structures = list(irrigation_shapefile['SW_WDID1'].astype(str))
    ditch_structures = list(ditches_shapefile['wdid'].astype(str))
    other_structures = list(structures_shapefile['WDID'].astype(str))
    all_types_list = []
    for structure_name in self.structures_objects:
      self.structures_objects[structure_name].acreage = {}
      self.structures_objects[structure_name].structure_types = []
      if structure_name in aggregated_diversions:
        ind_structure_list = aggregated_diversions[structure_name]
      else:
        ind_structure_list = [structure_name,]
    
      for ind_structure in ind_structure_list:
        if ind_structure in irrigation_structures:
          this_irrigated_area = irrigation_shapefile[irrigation_shapefile['SW_WDID1'] == ind_structure]
          self.structures_objects[structure_name].structure_types.append('Irrigation')
          for index_ia, row_ia in this_irrigated_area.iterrows():
            if row_ia['CROP_TYPE'] in self.structures_objects[structure_name].acreage:
              self.structures_objects[structure_name].acreage[row_ia['CROP_TYPE']] += row_ia['ACRES']
            else:
              self.structures_objects[structure_name].acreage[row_ia['CROP_TYPE']] = row_ia['ACRES']
              
      for ind_structure in ind_structure_list:
        if ind_structure in ditch_structures:
          self.structures_objects[structure_name].structure_types.append('Irrigation')
          total_decree = 0.0
          for right_name in self.structures_objects[structure_name].rights_list:
            if self.structures_objects[structure_name].rights_objects[right_name].decree_af < 990.0 * 1.98 * 30.0:
              total_decree += self.structures_objects[structure_name].rights_objects[right_name].decree_af
          total_acreage = 0.0
          for crop_name in self.structures_objects[structure_name].acreage:
            total_acreage += self.structures_objects[structure_name].acreage[crop_name]
          implied_acreage = max(total_decree - total_acreage * 3.5 /6.0, 0.0)
          if 'ALFALFA' in self.structures_objects[structure_name].acreage:
            self.structures_objects[structure_name].acreage['ALFALFA'] += implied_acreage
          else:
            self.structures_objects[structure_name].acreage['ALFALFA'] = implied_acreage
        elif ind_structure in irrigation_structures:
          pass
        elif ind_structure in other_structures:
          this_structure = structures_shapefile[structures_shapefile['WDID'] == ind_structure]
          if this_structure.loc[this_structure.index[0], 'StructType'] not in all_types_list:
            all_types_list.append(this_structure.loc[this_structure.index[0], 'StructType'])
          if this_structure.loc[this_structure.index[0], 'StructType'] == 'Reservoir':
            self.structures_objects[structure_name].structure_types.append('Reservoir')
          elif this_structure.loc[this_structure.index[0], 'StructType'] == 'Minimum Flow':
            self.structures_objects[structure_name].structure_types.append('Minimum Flow')
          elif this_structure.loc[this_structure.index[0], 'StructType'] == 'Power Plant':
            self.structures_objects[structure_name].structure_types.append('Municipal')            
          else:
            if np.sum(self.structures_objects[structure_name].historical_monthly_demand['demand']) == 0.0:
              self.structures_objects[structure_name].structure_types.append('Carrier')
            elif structure_name == '36000662_D' or structure_name == '38000880_D' or structure_name == '5000734_D' or structure_name == '5300555_D':
              self.structures_objects[structure_name].structure_types.append('Irrigation')
            elif structure_name[:4] == '3600' or structure_name[:4] == '3601' or structure_name[:4] == '3700' or structure_name[:4] == '3800' or structure_name[:4] == '3801' or structure_name[:4] == '5300' or structure_name[:4] == '7200' or structure_name[:4] == '7201':
              self.structures_objects[structure_name].structure_types.append('Municipal')
            elif structure_name[:4] == '3604' or structure_name[:4] == '3704' or structure_name[:4] == '3804' or structure_name[:4] == '5104' or structure_name[:4] == '7204':
              self.structures_objects[structure_name].structure_types.append('Export')
            elif structure_name[:6] == '36_ADC' or  structure_name[:6] == '37_ADC' or structure_name[:6] == '39_ADC' or structure_name[:6] == '45_ADC' or structure_name[:6] == '50_ADC' or structure_name[:6] == '51_ADC' or structure_name[:6] == '52_ADC' or structure_name[:6] == '53_ADC' or structure_name[:6] == '70_ADC' or structure_name[:6] == '72_ADC':
              self.structures_objects[structure_name].structure_types.append('Irrigation')
            elif structure_name == '3900532':
              self.structures_objects[structure_name].structure_types.append('Irrigation')
            elif  structure_name == '3900967':
              self.structures_objects[structure_name].structure_types.append('Municipal')
            elif  structure_name == '5100941':
              self.structures_objects[structure_name].structure_types.append('Irrigation')
            elif  structure_name == '5100958' or structure_name == '5101070':
              self.structures_objects[structure_name].structure_types.append('Municipal')
            else:
              print('structure_type', end = " ")
              print(structure_name, end = " ")
              print(ind_structure)

        else:
          if ind_structure[3:6] == 'ARC':
            self.structures_objects[structure_name].structure_types.append('Reservoir')
          elif ind_structure[3:6] == 'ASC':
            self.structures_objects[structure_name].structure_types.append('Reservoir')
          elif ind_structure[3:6] == 'AMC':
            self.structures_objects[structure_name].structure_types.append('None')
          elif ind_structure[-2:] == 'HU':
            self.structures_objects[structure_name].structure_types.append('Reservoir')
          elif ind_structure[-2:] == '_I':
            self.structures_objects[structure_name].structure_types.append('Irrigation')
          elif ind_structure[-2:] == '_M':
            self.structures_objects[structure_name].structure_types.append('Minimum Flow')
          elif ind_structure[-2:] == 'PL':
            self.structures_objects[structure_name].structure_types.append('None')
          elif ind_structure == '7203904AG' or ind_structure == '7204033AG':
            self.structures_objects[structure_name].structure_types.append('Reservoir')
          elif ind_structure == '3604683SU':
            self.structures_objects[structure_name].structure_types.append('Export')
          elif ind_structure == '36GMCON':
            self.structures_objects[structure_name].structure_types.append('Reservoir')
          elif ind_structure == '36_KeyMun':
            self.structures_objects[structure_name].structure_types.append('Municipal')
          elif ind_structure[:6] == '37VAIL':
            self.structures_objects[structure_name].structure_types.append('Municipal')
          elif ind_structure[:7] == '3803713':
            self.structures_objects[structure_name].structure_types.append('Municipal')
          elif ind_structure == '3804625SU':
            self.structures_objects[structure_name].structure_types.append('Export')
          elif ind_structure == '4200520':
            self.structures_objects[structure_name].structure_types.append('Municipal')
          elif ind_structure == '4200541':
            self.structures_objects[structure_name].structure_types.append('Municipal')
          elif ind_structure[:7] == '5003668':
            self.structures_objects[structure_name].structure_types.append('None')
          elif ind_structure == '70FD1' or ind_structure == '70FD2':
            self.structures_objects[structure_name].structure_types.append('None')
          elif ind_structure[:7] == '7200813':
            if ind_structure == '7200813':
              self.structures_objects[structure_name].structure_types.append('Irrigation')
            else:
              self.structures_objects[structure_name].structure_types.append('Municipal')
          elif ind_structure == '7202003_M':
            self.structures_objects[structure_name].structure_types.append('Minimum Flow')
          elif ind_structure == '72_GJMun' or ind_structure == '72_UWCD' or ind_structure == 'ChevDem':
            self.structures_objects[structure_name].structure_types.append('Municipal')
          elif ind_structure == 'MoffatBF':
            self.structures_objects[structure_name].structure_types.append('Minimum Flow')
          elif ind_structure == 'Baseflow':
            self.structures_objects[structure_name].structure_types.append('Minimum Flow')
          elif ind_structure == 'CSULimitPLN' or ind_structure == 'HUPLimitPLN' or ind_structure == 'ColRivPln':
            self.structures_objects[structure_name].structure_types.append('None')
          elif ind_structure == '3702059_2':
            self.structures_objects[structure_name].structure_types.append('Minimum Flow')
          elif ind_structure == '5300584P':
            self.structures_objects[structure_name].structure_types.append('Minimum Flow')
          elif ind_structure == '3804625M2':
            self.structures_objects[structure_name].structure_types.append('Minimum Flow')
          elif ind_structure == '3903508_Ex':
            self.structures_objects[structure_name].structure_types.append('None')
          elif ind_structure == '7202001_2':
            self.structures_objects[structure_name].structure_types.append('Minimum Flow')
          elif ind_structure[:2] == '09' or ind_structure[-3:] == 'Dwn':
            self.structures_objects[structure_name].structure_types.append('Minimum Flow')
          else:
            print('none type')
            print(structure_name, end = " ")
            print(ind_structure)
    for structure_name in self.structures_objects:
      if structure_name in irrigation_structures:
        this_irrigated_area = irrigation_shapefile[irrigation_shapefile['SW_WDID1'] == structure_name]
        self.structures_objects[structure_name].purchase_price = 9999
        for index_ia, row_ia in this_irrigated_area.iterrows():
          self.structures_objects[structure_name].purchase_price = min(self.structures_objects[structure_name].purchase_price, marginal_net_benefits[row_ia['CROP_TYPE']] / (et_requirements[row_ia['CROP_TYPE']] /12.0))
        if self.structures_objects[structure_name].purchase_price == 9999:
          self.structures_objects[structure_name].purchase_price = 100.0
    
      elif structure_name in aggregated_diversions:
        for ind_structure in aggregated_diversions[structure_name]:
          if ind_structure in irrigation_structures:
            this_irrigated_area = irrigation_shapefile[irrigation_shapefile['SW_WDID1'] == ind_structure]
            self.structures_objects[structure_name].purchase_price = 9999
            for index_ia, row_ia in this_irrigated_area.iterrows():
              self.structures_objects[structure_name].purchase_price = min(self.structures_objects[structure_name].purchase_price, marginal_net_benefits[row_ia['CROP_TYPE']] / (et_requirements[row_ia['CROP_TYPE']] /12.0))
            if self.structures_objects[structure_name].purchase_price == 9999:
              self.structures_objects[structure_name].purchase_price = 100.0

      elif 'Irrigation' in self.structures_objects[structure_name].structure_types:
        self.structures_objects[structure_name].purchase_price = 100.0
      elif 'Minimum Flow' in self.structures_objects[structure_name].structure_types:
        self.structures_objects[structure_name].purchase_price = 100.0
      else:
        self.structures_objects[structure_name].purchase_price = 1000.0
        
  def find_change_by_wyt(self, start_year, end_year):
    cum_change_delivery = np.zeros(end_year - start_year + 1)
    station_id_column_start = 0
    station_id_column_length = 12
    for j in range(first_line, len(downstream_data)):
      station_id = str(downstream_data[j][station_id_column_start:station_id_column_length].strip())
      initial_year_toggle = 0
      for index, row in self.structures_objects[station_id].historical_monthly_deliveries.iterrows():
        if initial_year_toggle == 0:
          initial_year = index.year
          initial_year_toggle = 1
        if index.month > 9:
          year_num = index.year - initial_year
        else:
          year_num = index.year - initial_year - 1
        cum_change_delivery[year_num] += self.structures_objects[station_id].adaptive_monthly_deliveries.loc[index, 'deliveries'] - row['deliveries']
      self.structures_objects[station_id].cumulative_change = np.sort(cum_change_delivery)
  
    
  def find_percent_delivered(self, marginal_net_benefits, et_requirements, aggregated_diversions, irrigation_ucrb, ditches_ucrb, structure_buyouts, structure_purchases, downstream_data, purchase_multiplier = 1.25, buyout_cost = 50):
    irrigation_structures = list(irrigation_ucrb['SW_WDID1'].astype(str))
    ditch_structures = list(ditches_ucrb['wdid'].astype(str))
    percent_filled_columns = ['pct_filled_baseline', 'total_deliveries', 'structure_name', 'structure_types', 'lost_revenue',]
    for x in range(0, 10):
      percent_filled_columns.append('change_' + str(x))
      percent_filled_columns.append('revene_' + str(x))
      percent_filled_columns.append('cum_change_neg' + str(x))
      percent_filled_columns.append('cum_change_pos' + str(x))
    percent_filled = pd.DataFrame(index = self.structures_objects, columns = percent_filled_columns)
    baseline_filled = np.zeros(len(self.structures_objects))
    total_deliveries = np.zeros(len(self.structures_objects))
    change_dict = {}
    for x in range(0, 10):
      change_dict['change_' + str(x)] = np.zeros(len(self.structures_objects))
      change_dict['revenue_' + str(x)] = np.zeros(len(self.structures_objects))
      change_dict['cum_change_pos' + str(x)] = np.zeros(len(self.structures_objects))
      change_dict['cum_change_neg' + str(x)] = np.zeros(len(self.structures_objects))
    min_cost = np.zeros(len(self.structures_objects))
    counter = 0
    structure_list = []
    structure_type_list = []
    for j in range(0,len(downstream_data)):
      if downstream_data[j][0] != '#':
        first_line = int(j * 1)
        break
    cum_change_delivery = np.zeros(200)
    station_id_column_start = 0
    station_id_column_length = 12
    for j in range(first_line, len(downstream_data)):
      station_id = str(downstream_data[j][station_id_column_start:station_id_column_length].strip())
      initial_year_toggle = 0
      for index, row in self.structures_objects[station_id].historical_monthly_deliveries.iterrows():
        if initial_year_toggle == 0:
          initial_year = index.year
          initial_year_toggle = 1
        if index.month > 9:
          year_num = index.year - initial_year
        else:
          year_num = index.year - initial_year - 1
        cum_change_delivery[year_num] += self.structures_objects[station_id].adaptive_monthly_deliveries.loc[index, 'deliveries'] - row['deliveries']
      self.structures_objects[station_id].cumulative_change = np.sort(cum_change_delivery)

    for structure_name in self.structures_objects:
      if structure_name in irrigation_structures:
        this_irrigated_area = irrigation_ucrb[irrigation_ucrb['SW_WDID1'] == structure_name]
        min_cost[counter] = 9999.9
        for index_ia, row_ia in this_irrigated_area.iterrows():
          min_cost[counter] = min(min_cost[counter], marginal_net_benefits[row_ia['CROP_TYPE']] / (et_requirements[row_ia['CROP_TYPE']] /12.0))
        if min_cost[counter] == 9999.9:
          min_cost[counter] = 200.0
    
      elif structure_name in aggregated_diversions:
        for ind_structure in aggregated_diversions[structure_name]:
          if ind_structure in irrigation_structures:
            this_irrigated_area = irrigation_ucrb[irrigation_ucrb['SW_WDID1'] == ind_structure]
            min_cost[counter] = 9999.9
            for index_ia, row_ia in this_irrigated_area.iterrows():
              min_cost[counter] = min(min_cost[counter], marginal_net_benefits[row_ia['CROP_TYPE']] / (et_requirements[row_ia['CROP_TYPE']] /12.0))
            if min_cost[counter] == 9999.9:
              min_cost[counter] = 200.0

      elif structure_name in ditch_structures:
        min_cost[counter] = 200.0
      else:
        min_cost[counter] = 1000.0
      if structure_name == '5100848':
        counter_for_puchase = counter * 1 + 0
      initial_year_toggle = 0
      hist_demand = np.zeros(200)
      for index, row in self.structures_objects[structure_name].historical_monthly_demand.iterrows():
        if initial_year_toggle == 0:
          initial_year = index.year
          initial_year_toggle = 1
        if index.month > 9:
          year_num = index.year - initial_year
        else:
          year_num = index.year - initial_year - 1
        hist_demand[year_num] += row['demand']

      adp_demand = np.zeros(200)
      for index, row in self.structures_objects[structure_name].adaptive_monthly_demand.iterrows():
        if initial_year_toggle == 0:
          initial_year = index.year
          initial_year_toggle = 1
        if index.month > 9:
          year_num = index.year - initial_year
        else:
          year_num = index.year - initial_year - 1
        adp_demand[year_num] += row['demand']

      hist_delivery = np.zeros(200)
      for index, row in self.structures_objects[structure_name].historical_monthly_deliveries.iterrows():
        if initial_year_toggle == 0:
          initial_year = index.year
          initial_year_toggle = 1
        if index.month > 9:
          year_num = index.year - initial_year
        else:
          year_num = index.year - initial_year - 1
        hist_delivery[year_num] += row['deliveries']

      adp_delivery = np.zeros(200)
      for index, row in self.structures_objects[structure_name].adaptive_monthly_deliveries.iterrows():
        if initial_year_toggle == 0:
          initial_year = index.year
          initial_year_toggle = 1
        if index.month > 9:
          year_num = index.year - initial_year
        else:
          year_num = index.year - initial_year - 1
        adp_delivery[year_num] += row['deliveries']
      delivery_change = np.zeros(200)
      for x in range(0, len(adp_delivery)):
        delivery_change[x] = adp_delivery[x] - hist_delivery[x]

      total_revenue = np.zeros(200)
      for x in range(0, len(total_revenue)):
        total_revenue[x] = delivery_change[x] * min_cost[counter]
        
      structure_purchases['datetime_vals'] = pd.to_datetime(structure_purchases['date'])
      these_buyouts = structure_purchases[structure_purchases['structure'].astype(str) == structure_name]
      for index, row in these_buyouts.iterrows():
        if row['datetime_vals'].month > 9:
          year_num = row['datetime_vals'].year - initial_year
        else:
          year_num = row['datetime_vals'].year - initial_year - 1
        total_revenue[year_num] += row['demand'] * min_cost[counter] * purchase_multiplier 

      structure_buyouts['datetime_vals'] = pd.to_datetime(structure_buyouts['date'])
      these_buyouts = structure_buyouts[structure_buyouts['structure'].astype(str) == structure_name]
      for index, row in these_buyouts.iterrows():
        if row['datetime_vals'].month > 9:
          year_num = row['datetime_vals'].year - initial_year
        else:
          year_num = row['datetime_vals'].year - initial_year - 1
        total_revenue[year_num] += row['demand'] * buyout_cost


      sorted_change = np.sort(delivery_change)
      sorted_index = np.argsort(delivery_change)
      sorted_revenue = total_revenue[sorted_index]
      sorted_cumulative_change = np.sort(self.structures_objects[structure_name].cumulative_change)
      if np.sum(hist_demand) > 0.0:
        baseline_filled[counter] = np.sum(hist_delivery) / np.sum(hist_demand)
      elif np.sum(hist_delivery) > 0.0:
        total_years = self.structures_objects[structure_name].historical_monthly_deliveries.index[-1].year - self.structures_objects[structure_name].historical_monthly_deliveries.index[0].year
        total_decree = 0.0
        for sri in self.structures_objects[structure_name].rights_list:
          total_decree += self.structures_objects[structure_name].rights_objects[sri].decree_af
        baseline_filled[counter] = np.sum(hist_delivery) / (total_years* 365.0 * total_decree / 30.0)
      counter_change = 0
      counter_change_2 = 0
      for x in range(0, len(sorted_change)):
        if sorted_change[x] != 0.0:
          if 'change_' + str(counter_change) in change_dict:
            change_dict['change_' + str(counter_change)][counter] = sorted_change[x]
            change_dict['revenue_' + str(counter_change)][counter] = sorted_revenue[x]
          else:
            change_dict['change_' + str(counter_change)] = np.zeros(len(self.structures_objects))
            change_dict['revenue_' + str(counter_change)] = np.zeros(len(self.structures_objects))
            change_dict['change_' + str(counter_change)][counter] = sorted_change[x]
            change_dict['revenue_' + str(counter_change)][counter] = sorted_revenue[x]
          
          counter_change += 1
          
      for x in range(0, 10):
        change_dict['cum_change_neg' + str(x)][counter] = self.structures_objects[structure_name].cumulative_change[x]
      for x in range(1, 11):
        change_dict['cum_change_pos' + str(x-1)][counter] = self.structures_objects[structure_name].cumulative_change[0-x]
        
      total_deliveries[counter] = np.sum(hist_delivery) / 63.0
      if 'Irrigation' in self.structures_objects[structure_name].structure_types:
        structure_type_list.append('Irrigation')
      elif 'Minimum Flow' in self.structures_objects[structure_name].structure_types:
        structure_type_list.append('Minimum Flow')
      elif 'Export' in self.structures_objects[structure_name].structure_types:
        structure_type_list.append('Export')
      elif 'Municipal' in self.structures_objects[structure_name].structure_types:
        structure_type_list.append('Municipal')
      else:
        structure_type_list.append('None')

      structure_list.append(structure_name)
      counter += 1
    snowpack_list = []
    rev_orig = []
    rev_adapt = []
    structure_use = '5100848'
    this_structure_cost = min_cost[counter_for_puchase]
    ind_buyouts = structure_buyouts[structure_buyouts['structure'].astype(str) == structure_use]
    ind_purchases = structure_purchases[structure_purchases['structure'].astype(str) == structure_use]
    ind_buyouts['date'] = pd.to_datetime(ind_buyouts['date'])
    ind_purchases['date'] = pd.to_datetime(ind_purchases['date'])
    for year_num in range(1950, 2014): 
      datetime_val = datetime(year_num, 5, 1, 0, 0)
      snowpack_list.append(self.basin_snowpack['14010001'].loc[datetime_val, 'basinwide_average'])

      total_init_deliv = 0.0
      total_new_deliv = 0.0
      annual_buyouts = 0.0
      annual_purchases = 0.0
      for month_num in range(10, 13):
        datetime_val_month = datetime(year_num-1, month_num, 1, 0, 0)
        total_init_deliv += self.structures_objects[structure_use].historical_monthly_deliveries.loc[datetime_val_month, 'deliveries']
        total_new_deliv += self.structures_objects[structure_use].adaptive_monthly_deliveries.loc[datetime_val_month, 'deliveries']
        this_month_buyouts = ind_buyouts[ind_buyouts['date'].astype(str) == datetime_val_month]
        this_month_purchases = ind_purchases[ind_purchases['date'].astype(str) == datetime_val_month]
        annual_buyouts += np.sum(this_month_buyouts['demand'])
        annual_purchases += np.sum(this_month_purchases['demand'])
        
      for month_num in range(1, 10):
        datetime_val_month = datetime(year_num, month_num, 1, 0, 0)
        total_init_deliv += self.structures_objects[structure_use].historical_monthly_deliveries.loc[datetime_val_month, 'deliveries']
        total_new_deliv += self.structures_objects[structure_use].adaptive_monthly_deliveries.loc[datetime_val_month, 'deliveries']
        this_month_buyouts = ind_buyouts[ind_buyouts['date'] == datetime_val_month]
        this_month_purchases = ind_purchases[ind_purchases['date'] == datetime_val_month]
        annual_buyouts += np.sum(this_month_buyouts['demand'])
        annual_purchases += np.sum(this_month_purchases['demand'])

      rev_orig.append(total_init_deliv * this_structure_cost)  
      rev_adapt.append(total_new_deliv * this_structure_cost + annual_purchases * purchase_multiplier * this_structure_cost + annual_buyouts * buyout_cost)  
    fig, ax = plt.subplots()
    ax.scatter(snowpack_list, rev_orig, c = 'black', s = 25)    
    ax.scatter(snowpack_list, rev_adapt, c = 'indianred', s = 25)    
    ax.set_xlabel('May 1 Snowpack % of Average', fontsize = 14, weight = 'bold', fontname = 'Gill Sans MT')    
    ax.set_ylabel('Total Revenue, Red Top Valley ($MM)', fontsize = 14, weight = 'bold', fontname = 'Gill Sans MT')    
    legend_location = 'upper right'
    #legend_location_alt = 'upper right'
    legend_element = [Patch(facecolor='black', edgecolor='black', label='Historical Baseline'),
                     Patch(facecolor='indianred', edgecolor='black', label='w/ Informal Transfers')]
    #legend_element2 = [Patch(facecolor='indianred', edgecolor='black', label='Purchase Partners')]
    #legend_element3 = [Patch(facecolor='steelblue', edgecolor='black', label='Buyout Partners')]
    #legend_element4 = [Patch(facecolor='black', edgecolor='black', label='Uninvolved Parties')]
    legend_properties = {'family':'Gill Sans MT','weight':'bold','size':14}
    ax.legend(handles=legend_element, loc=legend_location, prop=legend_properties)

    plt.show()
    percent_filled['pct_filled_baseline'] = baseline_filled
    percent_filled['total_deliveries'] = total_deliveries
    percent_filled['structure_name'] = structure_list
    percent_filled['structure_types'] = structure_type_list
    for x in range(0, 10):
      if 'change_' + str(x) in change_dict:
        percent_filled['change_' + str(x)] = change_dict['change_' + str(x)]
        percent_filled['revenue_' + str(x)] = change_dict['revenue_' + str(x)]
        percent_filled['cum_change_pos' + str(x)] = change_dict['cum_change_pos' + str(x)]
        percent_filled['cum_change_neg' + str(x)] = change_dict['cum_change_neg' + str(x)]
      else:
        break
    
    return percent_filled
               
  def set_rights_to_reservoirs(self, rights_name_list, structure_name_list, rights_priority_list, rights_decree_list, fill_type_list):
    counter = 0
    for rights_name, reservoir_name, rights_priority, rights_decree, fill_type in zip(rights_name_list, structure_name_list, rights_priority_list, rights_decree_list, fill_type_list):
      if reservoir_name not in self.structures_objects:
        self.structures_objects[reservoir_name] = Reservoir('small_count_' + str(counter), reservoir_name, rights_decree)
        counter += 1
      self.structures_objects[reservoir_name].initialize_right(rights_name, rights_priority, 0.0, fill_type)
      
  def set_rights_to_structures(self, rights_name_list, structure_name_list, rights_priority_list, rights_decree_list):
    for rights_name, structure_name, rights_priority, rights_decree in zip(rights_name_list, structure_name_list, rights_priority_list, rights_decree_list):
      if structure_name not in self.structures_objects:
        self.structures_list['unknown'].append(structure_name)
        self.structures_list['total'].append(structure_name)
        self.structures_objects[structure_name] = Structure(structure_name, 'unknown')
      self.structures_objects[structure_name].initialize_right(rights_name, rights_priority, rights_decree)

  def set_rights_to_instream(self, rights_name_list, structure_name_list, rights_priority_list, rights_decree_list):
    for rights_name, structure_name, rights_priority, rights_decree in zip(rights_name_list, structure_name_list, rights_priority_list, rights_decree_list):
      if structure_name not in self.structures_objects:
        self.structures_list['unknown'].append(structure_name)
        self.structures_list['total'].append(structure_name)
        self.structures_objects[structure_name] = Structure(structure_name, 'unknown')
      self.structures_objects[structure_name].initialize_right(rights_name, rights_priority, rights_decree)
      
  def make_snow_regressions(self, release_data, snowpack_data, res_station, year_start, year_end):
    month_index = ['OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP']
    monthly_control = {}
    monthly_control_int = {}
    for x in range(1, 13):
      monthly_control[x] = []
      monthly_control_int[x] = []
    for index, row in release_data.iterrows():
      if index > datetime(1950, 10, 1, 0, 0):
        this_row_month = index.month
        monthly_control_int[this_row_month].append(row[res_station + '_location'])
    
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

    coef = {}
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
      flow2['all'] = []
      snowpack['all'] = []
      
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
          
        if remaining_usable_flow < 0.0 or pd.isna(current_snowpack):
          skip_this = 1
        else:
          if control_location in flow2:
            snowpack[control_location].append(current_snowpack)
            flow2[control_location].append(remaining_usable_flow)
          snowpack['all'].append(current_snowpack)
          flow2['all'].append(remaining_usable_flow)

      coef[month_index[month_num]] = {}
      for control_loc in control_location_list:        
        coef[month_index[month_num]][control_loc] = np.polyfit(np.asarray(snowpack[control_loc]), np.asarray(flow2[control_loc]), 1)
      use_av = False
      try:
        coef[month_index[month_num]]['all'] =  np.polyfit(np.asarray(snowpack['all']), np.asarray(flow2['all']), 1)
      except:
        use_av = True      
      if use_av:
        coef[month_index[month_num]]['all'] = np.zeros(2)
        coef[month_index[month_num]]['all'][0] = 0.0
        coef[month_index[month_num]]['all'][1] = np.mean(np.asarray(flow2['all']))        
    return coef

  def find_available_water(self, snow_coefs, ytd_diversions, res_station, snow_station, datetime_val):
    total_water = []
    month_index = ['OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP']

    already_diverted = 0.0
    available_storage = self.structures_objects[res_station].adaptive_reservoir_timeseries.loc[datetime_val, res_station]/1000.0
    already_diverted += self.structures_objects[res_station].adaptive_reservoir_timeseries.loc[datetime_val, res_station + '_diversions']/1000.0
    if datetime_val.month > 9:
      month_val = month_index[datetime_val.month - 10]
    else:
      month_val = month_index[datetime_val.month + 2]
    control_location = self.structures_objects[res_station].adaptive_monthly_control.loc[datetime_val, 'location']
    if control_location in snow_coefs[month_val]:
      available_snowmelt = (self.basin_snowpack[snow_station].loc[datetime_val, 'basinwide_average'] * snow_coefs[month_val][control_location][0] + snow_coefs[month_val][control_location][1])/1000.0
    else:
      available_snowmelt = (self.basin_snowpack[snow_station].loc[datetime_val, 'basinwide_average'] * snow_coefs[month_val]['all'][0] + snow_coefs[month_val]['all'][1])/1000.0
    total_water = available_storage + available_snowmelt + ytd_diversions

    return total_water

  def find_adaptive_purchases(self, downstream_data, res_station, date_use):

    station_id_column_length = 12
    station_id_column_start = 0
    downstream_station_id_column_start = 36
    downstream_station_id_column_end = 48
    
    diversion_structures = ['5104601', '5104603', '5104625', '5104634', '5104655', '5104700', '3900574', '3704614', '3804613', '3804617', '3804625', '5103710', '5100958', '5103695']
    available_purchases = {}
    purchase_types = ['right',  'structure', 'demand', 'priority', 'consumptive_fraction']
    curr_position = 0
    month_name_list = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    month_id = month_name_list[date_use.month - 1]
    current_control_location = self.structures_objects[res_station].adaptive_monthly_control.loc[date_use, 'location']
    #total_physical_supply = self.structures_objects[station_id].adaptive_monthly_outflows.loc[datetime_val, 'outflows'] * 1.0
    total_physical_supply = 999999.9
    for j in range(0,len(downstream_data)):
      if downstream_data[j][0] != '#':
        first_line = int(j * 1)
        break
    start_station = str(res_station)
    reservoir_right = self.structures_objects[res_station].sorted_rights[0]
    for rt in purchase_types:
      available_purchases[rt] = []
    change_points_structure = []
    change_points_right_id = []
    change_points_demand = []
    change_points_buyout_demand = []
    change_points_buyout_purchase = []
    change_points_date = []
    change_points_right = []
    change_points_consumptive = []
    for j in range(first_line, len(downstream_data)):
      station_id = str(downstream_data[j][station_id_column_start:station_id_column_length].strip())
      downstream_id = str(downstream_data[j][downstream_station_id_column_start:downstream_station_id_column_end].strip())
      if 'Irrigation' in self.structures_objects[station_id].structure_types:
        try:
          for r_id in self.structures_objects[station_id].rights_list:
            available_purchases['demand'].append(self.structures_objects[station_id].rights_objects[r_id].adaptive_monthly_deliveries.loc[date_use, 'deliveries'])
            available_purchases['priority'].append(self.structures_objects[station_id].rights_objects[r_id].priority)
            available_purchases['right'].append(r_id)
            available_purchases['structure'].append(station_id)
            available_purchases['consumptive_fraction'].append(1.0 - self.structures_objects[station_id].return_fraction.loc[month_id, station_id])
        except:
          pass
      
      if station_id == start_station:
        total_available_supply = max(self.structures_objects[station_id].adaptive_monthly_outflows.loc[date_use, 'outflows'] - max(self.structures_objects[station_id].routed_plans.loc[date_use, 'plan'], 0.0), 0.0)
        if total_available_supply > -100.0:
          for ind_right in self.structures_objects[station_id].sorted_rights:
            if self.structures_objects[station_id].rights_objects[ind_right].priority > self.structures_objects[res_station].rights_objects[reservoir_right].priority:
              total_available_supply += self.structures_objects[station_id].rights_objects[ind_right].adaptive_monthly_deliveries.loc[date_use, 'deliveries']
          water_exchanges_choke = total_physical_supply - total_available_supply
          print(j, end = " ")
          print(station_id, end = " ")
          print(start_station, end = " ")
          print(total_available_supply, end = " ")
          print(total_physical_supply, end = " ")
          print(water_exchanges_choke, end = " ")
          print(len(available_purchases['priority']), end = " ")
          print(current_control_location, end = " ")
          print(available_purchases['structure'])
          if water_exchanges_choke > 0.0:
            for rt in purchase_types:
              available_purchases[rt] = np.asarray(available_purchases[rt])
            sorted_order = np.argsort(available_purchases['priority'] * -1.0)
            sorted_demand = available_purchases['demand'][sorted_order]
            sorted_rights = available_purchases['right'][sorted_order]
            sorted_structs = available_purchases['structure'][sorted_order]
            sorted_priorities = available_purchases['priority'][sorted_order]
            sorted_returns = available_purchases['consumptive_fraction'][sorted_order]
            total_demand = 0.0
            toggle_exchange = 0
            x = 0
            for x in range(0, len(sorted_demand)):
              total_demand += sorted_demand[x] * sorted_returns[x]
              total_right_delivery = self.structures_objects[sorted_structs[x]].rights_objects[sorted_rights[x]].adaptive_monthly_deliveries.loc[date_use, 'deliveries']
              more_senior_deliveries = 0.0
              for ind_right in self.structures_objects[sorted_structs[x]].sorted_rights:
                more_senior_deliveries += self.structures_objects[sorted_structs[x]].rights_objects[ind_right].adaptive_monthly_deliveries.loc[date_use, 'deliveries']
                if ind_right == sorted_rights[x]:
                  break
              print(x, end = " ")
              print(total_demand, end = " ")
              print(total_right_delivery, end = " ")
              print(sorted_structs[x], end = " ")
              print(sorted_rights[x])
              if total_right_delivery > 0.0:
                total_structure_deliveries = self.structures_objects[sorted_structs[x]].adaptive_monthly_deliveries.loc[date_use, 'deliveries']
                total_structure_demand = self.structures_objects[sorted_structs[x]].adaptive_monthly_demand.loc[date_use, 'demand']
                change_points_structure.append(sorted_structs[x])
                change_points_right_id.append(sorted_rights[x])
                change_points_demand.append(total_right_delivery)
                change_points_buyout_demand.append(more_senior_deliveries)
                change_points_buyout_purchase.append(0.0)
                change_points_date.append(date_use)
                change_points_right.append(sorted_priorities[x])
                change_points_consumptive.append(sorted_returns[x])
              if total_demand > water_exchanges_choke:
                break
            for rt in purchase_types:
              available_purchases[rt] = []
            for y in range(x+1, len(sorted_demand)):
              available_purchases['priority'].append(sorted_priorities[y])
              available_purchases['demand'].append(sorted_demand[y])
              available_purchases['right'].append(sorted_rights[y])
              available_purchases['structure'].append(sorted_structs[y])
              available_purchases['consumptive_fraction'].append(sorted_returns[y])
            total_physical_supply -= float(total_physical_supply - total_available_supply)
            print(change_points_right_id, end = " ")
            print(available_purchases['right'])
        start_station = str(downstream_id)
        
      if station_id == current_control_location or total_physical_supply <= 0.0:
        for rt in purchase_types:
          available_purchases[rt] = np.asarray(available_purchases[rt])
        if total_physical_supply > 0.0:
          sorted_order = np.argsort(available_purchases['priority'] * -1.0)
          sorted_demand = available_purchases['demand'][sorted_order]
          sorted_rights = available_purchases['right'][sorted_order]
          sorted_structs = available_purchases['structure'][sorted_order]
          sorted_priorities = available_purchases['priority'][sorted_order]
          sorted_returns = available_purchases['consumptive_fraction'][sorted_order]
          total_demand = 0.0
          toggle_exchange = 0
          for x in range(0, len(sorted_demand)):
            total_demand += sorted_demand[x] * sorted_returns[x]
            total_right_delivery = self.structures_objects[sorted_structs[x]].rights_objects[sorted_rights[x]].adaptive_monthly_deliveries.loc[date_use, 'deliveries']
            if total_right_delivery > 0.0:
              total_structure_deliveries = self.structures_objects[sorted_structs[x]].adaptive_monthly_deliveries.loc[date_use, 'deliveries']
              total_structure_demand = self.structures_objects[sorted_structs[x]].adaptive_monthly_demand.loc[date_use, 'demand']
              change_points_structure.append(sorted_structs[x])
              change_points_right_id.append(sorted_rights[x])
              change_points_demand.append(total_right_delivery)
              change_points_buyout_demand.append(total_structure_deliveries)
              change_points_buyout_purchase.append(0.0)
              change_points_date.append(date_use)
              change_points_right.append(sorted_priorities[x])
              change_points_consumptive.append(sorted_returns[x])
            if total_demand > total_physical_supply:
              break
        break

    change_points_purchase_df = pd.DataFrame(columns = ['structure', 'demand', 'right', 'consumptive', 'date'])
    change_points_buyout_df = pd.DataFrame(columns = ['structure', 'demand', 'demand_purchase', 'date'])
    if len(change_points_right) > 0:  
      all_sorted_order = np.argsort(np.asarray(change_points_right) * -1.0)
      sorted_right_ids = np.asarray(change_points_right_id)[all_sorted_order]
      sorted_structure_ids = np.asarray(change_points_structure)[all_sorted_order]
      last_right = sorted_right_ids[-1]
      last_structure = sorted_structure_ids[-1]
    
      change_points_purchase_df['structure'] = change_points_structure
      change_points_purchase_df['demand'] = change_points_demand
      change_points_purchase_df['right'] = change_points_right
      change_points_purchase_df['consumptive'] = change_points_consumptive
      change_points_purchase_df['date'] = change_points_date

      change_points_buyout_df['structure'] = change_points_structure
      change_points_buyout_df['demand'] = change_points_buyout_demand
      change_points_buyout_df['demand_purchase'] = change_points_buyout_purchase
      change_points_buyout_df['date'] = change_points_date
    else:
      last_right = ''
      last_structure = ''
    
    return change_points_purchase_df, change_points_buyout_df, last_right, last_structure
          
  def check_purchases(self, change_points_purchase_1, structure_outflows, structure_outflows_old, downstream_data):
    station_id_column_length = 12
    station_id_column_start = 0

    for j in range(0,len(downstream_data)):
      if downstream_data[j][0] != '#':
        first_line = int(j * 1)
        break
    running_purchases = 0.0
    for j in range(first_line, len(downstream_data)):
      station_id = str(downstream_data[j][station_id_column_start:station_id_column_length].strip())
      if station_id in list(change_points_purchase_1['structure'].astype(str)):
        this_station_purchase = change_points_purchase_1[change_points_purchase_1['structure'].astype(str)==station_id]
        running_purchases += np.sum(this_station_purchase['demand'])
    
    
  def find_buyout_partners(self, last_right, last_structure, res_struct_id, date_use):
    
    diversion_structures = ['5104601', '5104603', '5104625', '5104634', '5104655', '5104700', '3900574', '3704614', '3804613', '3804617', '3804625', '5103710', '5100958', '5103695']
    change_points_structure = []
    change_points_buyout_demand = []
    change_points_buyout_purchase = []
    change_points_date = []
    
    sri = self.structures_objects[res_struct_id].sorted_rights[0]
    use_buyouts = True
    try:
      initial_priority = self.structures_objects[last_structure].rights_objects[last_right].priority
    except:
      use_buyouts = False
    end_priority = self.structures_objects[res_struct_id].rights_objects[sri].priority
    if use_buyouts:
      for right_priority, right_id, struct_id in zip(self.rights_stack_priorities, self.rights_stack_ids, self.rights_stack_structure_ids):
        if right_priority >= initial_priority and right_priority < end_priority:
          total_decree = 0.0
          for ind_right in self.structures_objects[struct_id].rights_list:
            if self.structures_objects[struct_id].rights_objects[ind_right].priority < end_priority:
              total_decree += self.structures_objects[struct_id].rights_objects[ind_right].decree_af
          total_demand_level = self.structures_objects[struct_id].adaptive_monthly_deliveries.loc[date_use, 'deliveries'] * 1.0
          total_extra_demand = max(min(self.structures_objects[struct_id].adaptive_monthly_demand.loc[date_use, 'demand'], total_decree) - self.structures_objects[struct_id].adaptive_monthly_deliveries.loc[date_use, 'deliveries'], 0.0)

          if total_extra_demand > 0.0:
            change_points_structure.append(struct_id)
            change_points_buyout_demand.append(total_demand_level)
            change_points_buyout_purchase.append(total_extra_demand)
            change_points_date.append(date_use)              
                
    change_points_buyout_df = pd.DataFrame(columns = ['structure', 'demand', 'demand_purchase', 'date'])
    if len(change_points_structure) > 0:
      change_points_buyout_df['structure'] = change_points_structure
      change_points_buyout_df['demand'] = change_points_buyout_demand
      change_points_buyout_df['demand_purchase'] = change_points_buyout_purchase
      change_points_buyout_df['date'] = change_points_date

    return change_points_buyout_df, end_priority
                    
                          
  def read_rights_parameter_file(self):
    right_parameters = pd.read_csv('UCRB_analysis-master\index_figs\ind_right_fill_curve_snow.csv')
    right_parameters2 = pd.read_csv('UCRB_analysis-master\index_figs\ind_right_fill_curve_call.csv')
    for structure_name in self.structures_objects:
      for rights_name in self.structures_objects[structure_name].rights_objects:
        self.structures_objects[structure_name].rights_objects[rights_name].param_vals = {}
        self.structures_objects[structure_name].rights_objects[rights_name].snowpack_range = {}
        self.structures_objects[structure_name].rights_objects[rights_name].param_vals_2 = {}
        self.structures_objects[structure_name].rights_objects[rights_name].call_range = {}

    for index, row in right_parameters.iterrows():
      structure_name = row['structure']
      rights_name = row['rights']
      mn = row['month']
          
      self.structures_objects[structure_name].rights_objects[rights_name].param_vals[str(mn)] = np.array([row['param1'], row['param2']])
      self.structures_objects[structure_name].rights_objects[rights_name].snowpack_range[str(mn)] = np.array([row['snowmin'], row['snowmax']])
    for index, row in right_parameters2.iterrows():
      structure_name = row['structure']
      rights_name = row['rights']
      mn = row['month']
          
      self.structures_objects[structure_name].rights_objects[rights_name].param_vals_2[str(mn)] = np.array([row['param1'], row['param2']])
      self.structures_objects[structure_name].rights_objects[rights_name].snowpack_range[str(mn)] = np.array([row['callmin'], row['callmax']])

  
  def plot_forecast_index(self, numyears):
    rights_counter = 0
    for structure_name in self.structures_objects:            
      for rights_name in self.structures_objects[structure_name].rights_objects:
        rights_counter += 1
        index_figure = Plotter()
        total_forecast = {}
        for mn in range(0, 6):
          total_forecast = {}
          total_forecast[str(mn)] = np.zeros(numyears)
          for yearcnt in range(0, numyears):
            if self.structures_objects[structure_name].rights_objects[rights_name].monthly_snowpack[yearcnt*12 + mn] > -100.0:
              current_snowpack = self.structures_objects[structure_name].rights_objects[rights_name].monthly_snowpack[yearcnt*12 + mn]
              current_cb = self.structures_objects[structure_name].rights_objects[rights_name].distance_from_call[yearcnt*12 + mn]
              
              if str(mn) in self.structures_objects[structure_name].rights_objects[rights_name].param_vals:
                total_forecast[str(mn)][yearcnt] = self.estimate_logit_syn(current_snowpack, self.structures_objects[structure_name].rights_objects[rights_name].param_vals[str(mn)][0], self.structures_objects[structure_name].rights_objects[rights_name].param_vals[str(mn)][1])
              else:
                annual_boolean = np.logical_and(monthly_boolean == mn, self.structures_objects[structure_name].rights_objects[rights_name].monthly_snowpack > -100.0)
                total_forecast[str(mn)][yearcnt] = np.mean(self.structures_objects[structure_name].rights_objects[rights_name].percent_filled[annual_boolean])
              
          index_figure.add_forecast_pdf(total_forecast[str(mn)], 'gnuplot', mn)         
        index_figure.save_fig('UCRB_analysis-master\index_figs\index_forecast_S' + structure_name + 'R' + rights_name + '.png')

  def plot_exposure(self, numyears, pct_customers, restrict_purchase, read_from_file):
    if read_from_file:
      self.read_rights_parameter_file()

    transbasin = np.genfromtxt('UCRB_analysis-master/Structures_files/TBD_new.txt',dtype='str').tolist()
    mun_ind = np.genfromtxt('UCRB_analysis-master/Structures_files/M_I.txt',dtype='str').tolist()
    env_flows = ['7202003',]
    irrigation = np.genfromtxt('UCRB_analysis-master/Structures_files/irrigation.txt',dtype='str').tolist()
    right_type_list = ['transbasin', 'mi', 'env', 'irrigation']
    cost_list = [1500.0, 1000.0, 500.0, 400.0]
    cost_variability = [0.1, 0.1, 0.0, 0.5]
    right_structure_list = [transbasin, mun_ind, env_flows, irrigation]
    total_exposure = {}
    total_premiums = {}
    total_demand = {}
    random_length = 1000
    random_index = np.random.rand(random_length)
    for right_type in right_type_list:
      total_exposure[right_type] = {}
      total_premiums[right_type] = {}
      total_demand[right_type] = {}
      for mn in range(0, 6):
        total_exposure[right_type][str(mn)] = np.zeros(numyears)
        total_premiums[right_type][str(mn)] = np.zeros(numyears)
        total_demand[right_type][str(mn)] = np.zeros(numyears)
      
    for mn in range(0, 6):
      random_right_counter = 0
      for structure_name in self.structures_objects:
        use_structure = False
        for right_type_string, ave_cost, cost_var, structure_list in zip(right_type_list, cost_list, cost_variability, right_structure_list):
          if structure_name in structure_list:
            cost_per_af = (random_index[random_right_counter] * 2.0 - 1.0) * cost_var + ave_cost
            right_type = right_type_string
            use_structure = True
            
        if use_structure:
          for rights_name in self.structures_objects[structure_name].rights_objects:
            total_pct_fill = np.sum(self.structures_objects[structure_name].rights_objects[rights_name].monthly_deliveries) / np.sum(self.structures_objects[structure_name].rights_objects[rights_name].monthly_demand)
            if pct_customers == 'all':
              upper_bound = 2.0
              lower_bound = -1.0
            else:
              upper_bound = pct_customers * (mn + 1)
              lower_bound = pct_customers * mn
            if total_pct_fill > restrict_purchase and random_index[random_right_counter] <= upper_bound and random_index[random_right_counter] > lower_bound:
              for yearcnt in range(0, numyears):
                if self.structures_objects[structure_name].rights_objects[rights_name].monthly_snowpack[yearcnt*12 + mn] > -100.0:
                  current_snowpack = self.structures_objects[structure_name].rights_objects[rights_name].monthly_snowpack[yearcnt*12 + mn]
                  current_cb = self.structures_objects[structure_name].rights_objects[rights_name].distance_from_call[yearcnt*12 + mn]
                  index_forecast = self.estimate_logit_syn(current_snowpack, self.structures_objects[structure_name].rights_objects[rights_name].param_vals[str(mn)][0], self.structures_objects[structure_name].rights_objects[rights_name].param_vals[str(mn)][1])                        
                  for monthcnt in range(mn, 12):
                    volumetric_shortfall = self.structures_objects[structure_name].rights_objects[rights_name].monthly_demand[yearcnt*12 + monthcnt] - self.structures_objects[structure_name].rights_objects[rights_name].monthly_deliveries[yearcnt*12 + monthcnt]
                 
                    total_exposure[right_type][str(mn)][yearcnt] += volumetric_shortfall * cost_per_af
                    total_demand[right_type][str(mn)][yearcnt] += self.structures_objects[structure_name].rights_objects[rights_name].monthly_demand[yearcnt*12 + monthcnt]
                    total_premiums[right_type][str(mn)][yearcnt] += (1.0 - index_forecast) * self.structures_objects[structure_name].rights_objects[rights_name].monthly_demand[yearcnt*12 + monthcnt] * cost_per_af
            random_right_counter += 1
            if random_right_counter == random_length:
              random_right_counter = 0
              
    return total_exposure, total_premiums, total_demand
    
#        for rights_name in self.structures_objects[structure_name].rights_objects:
           
  def plot_all_rights(self, read_from_file):
    use_colors = {}
    reg_colors = sns.color_palette('gnuplot_r', 5)
    if read_from_file:
      self.read_rights_parameter_file()
    basin_places = {}
    basin_places['14010001'] = 0
    basin_places['14010002'] = 1
    basin_places['14010003'] = 2
    basin_places['14010004'] = 3
    basin_places['14010005'] = 4
    month_label = ['October', 'November', 'December', 'January', 'February', 'March']
    for mn in range(0, 6): 
      param1 = []
      param2 = []
      callbuffer = []
      basinloc = []
      for structure_name in self.structures_objects:
        if self.structures_objects[structure_name].basin in basin_places:
          current_basin = basin_places[self.structures_objects[structure_name].basin]
        else:
          current_basin = 5
          
        for rights_name in self.structures_objects[structure_name].rights_objects:
          if str(mn) in self.structures_objects[structure_name].rights_objects[rights_name].left_bins:
            for dbl, dbr in zip(self.structures_objects[structure_name].rights_objects[rights_name].left_bins[str(mn)], self.structures_objects[structure_name].rights_objects[rights_name].right_bins[str(mn)]):
              param1.append(self.structures_objects[structure_name].rights_objects[rights_name].param_vals[str(int(dbl)) + str(int(dbr)) + str(mn)][0])
              param2.append(max(min(self.structures_objects[structure_name].rights_objects[rights_name].param_vals[str(int(dbl)) + str(int(dbr)) + str(mn)][1], 5), -5))
              callbuffer.append(self.structures_objects[structure_name].rights_objects[rights_name].median_vals[str(int(dbl)) + str(int(dbr)) + str(mn)])
              basinloc.append(current_basin)
      param1 = np.asarray(param1)
      param2 = np.asarray(param2)
      callbuffer = np.asarray(callbuffer)
      basinloc = np.asarray(basinloc)
      confidence_bins_left = np.asarray(pd.qcut(pd.Series(param1), q = 2).cat.categories.left)
      confidence_bins_right = np.asarray(pd.qcut(pd.Series(param1), q = 2).cat.categories.right)
      buffer_bins_left = np.asarray(pd.qcut(pd.Series(callbuffer), q = 4).cat.categories.left)
      buffer_bins_right = np.asarray(pd.qcut(pd.Series(callbuffer), q = 4).cat.categories.right)
      confidence_bin_count = 0
      fig, ax = plt.subplots(3,2, figsize = (16,6))
      for cbl, cbr in zip(confidence_bins_left, confidence_bins_right):
        current_params = np.logical_and(param1 >= cbl, param1 < cbr)
        for basinnum in range(0, 6):
          if basinnum < 3:
            counter1 = basinnum
            counter2 = 0
          else:
            counter1 = basinnum - 3
            counter2 = 1
          multiplier_fill = 99999999999999
          for bbl, bbr in zip(buffer_bins_left, buffer_bins_right):
            hist_params = np.logical_and(np.logical_and(np.logical_and(param1 >= cbl, param1 < cbr), np.logical_and(callbuffer >= bbl, callbuffer < bbr)), basinloc == basinnum)
            if len(np.unique(param2[hist_params])) > 1:
              pos = np.linspace(np.min(param2[hist_params]), np.max(param2[hist_params]), 101)            
              kde_fill = stats.gaussian_kde(param2[hist_params])
              multiplier_fill = min(0.95 / np.max(kde_fill(pos)), multiplier_fill)
          color_count = 0
          ax[counter1][counter2].plot([-100.0, 100.0], [1.0,1.0], color = 'black', linewidth = 1.5) 
          for bbl, bbr in zip(buffer_bins_left, buffer_bins_right):  
            hist_params = np.logical_and(np.logical_and(np.logical_and(param1 >= cbl, param1 < cbr), np.logical_and(callbuffer >= bbl, callbuffer < bbr)), basinloc == basinnum)
            if len(np.unique(param2[hist_params])) > 1:
              pos = np.linspace(np.min(param2[hist_params]), np.max(param2[hist_params]), 101)            
              kde_fill = stats.gaussian_kde(param2[hist_params])
              ax[counter1][counter2].fill_between(pos, np.ones(len(pos))*confidence_bin_count, multiplier_fill * kde_fill(pos) + np.ones(len(pos))*confidence_bin_count, edgecolor = 'black', alpha = 0.6, facecolor = reg_colors[color_count])
            color_count += 1
        confidence_bin_count += 1
      fig.text(0.5, 0.0, 'Snowpack Accumulation Required for 50% Allocation (standard error from average)', ha='center', fontsize = 14, weight = 'bold', fontname = 'Gill Sans MT')
      basin_name = ['14010001', '14010002', '14010003', '14010004', '14010005', 'unknown']
      basincounter = 0
      for counter1 in range(0, 3):
        for counter2 in range(0, 2):
          ax[counter1][counter2].set_ylabel(basin_name[basincounter], fontsize = 14, weight = 'bold', fontname = 'Gill Sans MT')
          ax[counter1][counter2].set_xlim([-5, 5])
          ax[counter1][counter2].set_ylim([0, 2])
          basincounter += 1 
          if counter2 == 0:
            ax[counter1][counter2].set_yticks([0.5, 1.5])
            ax[counter1][counter2].set_yticklabels(['Weak\nSignal', 'Strong\nSignal'], fontsize = 14, weight = 'bold', fontname = 'Gill Sans MT')
          else:
            ax[counter1][counter2].set_yticks([])
            ax[counter1][counter2].set_yticklabels('')
          if counter1 == 0 or counter1 == 1:
            ax[counter1][counter2].set_xticks([])
            ax[counter1][counter2].set_xticklabels('')
          else:
            ax[counter1][counter2].set_xticks([-4, -2, 0, 2, 4])
            ax[counter1][counter2].set_xticklabels(['-4', 'Mostly delivered', '0', 'Mostly shortfalls', '4'], fontsize = 14, weight = 'bold', fontname = 'Gill Sans MT')
          
      legend_elements = []
      legend_location = 'upper right'
      lc = 0
      for bbl, bbr in zip(buffer_bins_left, buffer_bins_right):  
        legend_elements.append(Line2D([0], [0], color=reg_colors[lc], lw = 2, label=str('{:.2f}'.format(int(bbl/100.0)*0.1)) + ' to ' + str('{:.2f}'.format(int(bbr/100.0)*0.1)) + ' tAF'))
        lc+=1        
      leg = ax[2][0].legend(handles=legend_elements, loc=legend_location, prop={'family':'Gill Sans MT','weight':'bold','size':8}, framealpha = 1.0, ncol = 2)
      leg.set_title('Call Buffer', prop={'family':'Gill Sans MT','weight':'bold','size':8})
      fig.text(0.5, 0.99, month_label[mn], ha='center', fontsize = 16, weight = 'bold', fontname = 'Gill Sans MT')
      plt.tight_layout()
      plt.savefig('UCRB_analysis-master/index_figs/rights_performance_' + month_label[mn] + '.png', dpi = 150, bbox_inches = 'tight', pad_inches = 0.0)
      plt.close()
          


  def plot_demand_index(self, year_start, year_end):
    monthly_boolean = np.zeros(12 * (year_end - year_start))
    rights_counter = 0
    
    right_dict_list_snow = ['structure', 'rights', 'month', 'param1', 'param2', 'snowmin', 'snowmax']
    right_dict_list_call = ['structure', 'rights', 'month',  'param1', 'param2', 'callmin', 'callmax']
    transbasin = np.genfromtxt('UCRB_analysis-master/Structures_files/TBD.txt',dtype='str').tolist()
    self.right_parameters_snow = {}
    self.right_parameters_call = {}
    for x in right_dict_list_snow:
      self.right_parameters_snow[x] = []
    for x in right_dict_list_call:
      self.right_parameters_call[x] = []

    for structure_name in self.structures_objects:
      for rights_name in self.structures_objects[structure_name].rights_objects:
        rights_counter += 1
        for curr_year in range(0, year_end - year_start):
          for x in range(0, 12):
            monthly_boolean[curr_year * 12 + x] = x
            next_mo_demand = 0.0
            remaining_demand = 0.0
            if x < 11:
              remaining_demand = np.sum(self.structures_objects[structure_name].rights_objects[rights_name].monthly_demand[(curr_year * 12 + x + 1):(curr_year * 12 + 12)])
              remaining_deliveries = np.sum(self.structures_objects[structure_name].rights_objects[rights_name].monthly_deliveries[(curr_year * 12 + x + 1):(curr_year * 12 + 12)])
            if curr_year * 12 + x + 1 < len(self.structures_objects[structure_name].rights_objects[rights_name].monthly_demand):
              next_mo_demand = self.structures_objects[structure_name].rights_objects[rights_name].monthly_demand[curr_year * 12 + x + 1]
              next_mo_deliveries = self.structures_objects[structure_name].rights_objects[rights_name].monthly_deliveries[curr_year * 12 + x + 1]
            if remaining_demand > 0.0:
              self.structures_objects[structure_name].rights_objects[rights_name].percent_filled[curr_year * 12 + x] = float(remaining_deliveries) / float(remaining_demand)
            if next_mo_demand > 0.0:
              self.structures_objects[structure_name].rights_objects[rights_name].percent_filled_single[curr_year * 12 + x] = float(next_mo_deliveries) / float(next_mo_demand)

        month_label_list = ['October', 'November', 'December', 'January', 'February', 'March']
        month_label_list2 = ['April', 'May', 'June', 'July', 'August', 'September']
        all_years_used = np.logical_and(self.structures_objects[structure_name].rights_objects[rights_name].monthly_snowpack > -100.0, monthly_boolean < 6)
        use_vals = self.structures_objects[structure_name].rights_objects[rights_name].monthly_snowpack[all_years_used]
        use_calls = self.structures_objects[structure_name].rights_objects[rights_name].distance_from_call[monthly_boolean > 5]        
        x_range = [np.min(use_vals), np.max(use_vals)]
        x_range_2 = [np.min(use_calls), np.max(use_calls)]
        self.structures_objects[structure_name].rights_objects[rights_name].param_vals = {}
        self.structures_objects[structure_name].rights_objects[rights_name].param_vals_2 = {}
        self.structures_objects[structure_name].rights_objects[rights_name].call_range = {}
        self.structures_objects[structure_name].rights_objects[rights_name].snowpack_range = {}
        
        index_plots = Plotter(3, 2)
        counter1 = 0
        counter2 = 0
        for mn in range(0, 6):

          annual_boolean = np.logical_and(monthly_boolean == mn, self.structures_objects[structure_name].rights_objects[rights_name].monthly_snowpack > -100.0)
          total_fill = np.logical_and(annual_boolean, self.structures_objects[structure_name].rights_objects[rights_name].percent_filled >= 0.95)
          shortage_filled = np.logical_and(annual_boolean, self.structures_objects[structure_name].rights_objects[rights_name].percent_filled < 0.95)

          snow_vals = self.structures_objects[structure_name].rights_objects[rights_name].monthly_snowpack[annual_boolean]
          fill_vals = self.structures_objects[structure_name].rights_objects[rights_name].percent_filled[annual_boolean]
          
          snow_vals_fill = self.structures_objects[structure_name].rights_objects[rights_name].monthly_snowpack[total_fill]
          snow_vals_shortage = self.structures_objects[structure_name].rights_objects[rights_name].monthly_snowpack[shortage_filled]

          index_plots.plot_index_observation(x_range, snow_vals, fill_vals, counter1, counter2, month_label_list[mn])                              
          index_plots.plot_snow_pdf(x_range, snow_vals_fill, snow_vals_shortage, 0.33, counter1, counter2)

          dict_name = str(mn)
          use_params = self.calculate_index_regression_snowpack(annual_boolean, structure_name, rights_name, dict_name)
          if use_params:
            self.set_rights_parameters(mn, structure_name, rights_name, dict_name)
            numsteps = 100
            reg_counter = 0
            estimated_fill = np.zeros(numsteps)
            for x in np.nditer(np.linspace(x_range[0], x_range[1], num = numsteps)):
              estimated_fill[reg_counter] = self.estimate_logit_syn(x, self.structures_objects[structure_name].rights_objects[rights_name].param_vals[dict_name][0], self.structures_objects[structure_name].rights_objects[rights_name].param_vals[dict_name][1])
              reg_counter += 1
            index_plots.plot_index_projection(estimated_fill, numsteps, x_range, counter1, counter2)

          counter1 += 1
          if counter1 == 3:
            counter1 = 0
            counter2 +=1 
        index_plots.save_fig('UCRB_analysis-master\index_figs\S' + structure_name + 'R' + rights_name + '_snowpack.png')
        plt.close()

        index_plots = Plotter(3, 2)
        counter1 = 0
        counter2 = 0
        for mn in range(0, 6):
          annual_boolean_shortage = monthly_boolean == mn + 6
          total_fill_single = np.logical_and(annual_boolean_shortage, self.structures_objects[structure_name].rights_objects[rights_name].percent_filled_single >= 0.95)
          shortage_filled_single = np.logical_and(annual_boolean_shortage, self.structures_objects[structure_name].rights_objects[rights_name].percent_filled_single < 0.95)

          distance_vals = self.structures_objects[structure_name].rights_objects[rights_name].distance_from_call[annual_boolean_shortage]
          fill_single_vals = self.structures_objects[structure_name].rights_objects[rights_name].percent_filled_single[annual_boolean_shortage]
          
          distance_vals_fill = self.structures_objects[structure_name].rights_objects[rights_name].distance_from_call[total_fill_single]
          distance_vals_shortage = self.structures_objects[structure_name].rights_objects[rights_name].distance_from_call[shortage_filled_single]

          index_plots.plot_index_observation(x_range_2, distance_vals, fill_single_vals, counter1, counter2, month_label_list[mn])                              
          index_plots.plot_snow_pdf(x_range_2, distance_vals_fill, distance_vals_shortage, 0.33, counter1, counter2)
          
          dict_name = str(mn)
          use_params = self.calculate_index_regression_calldistance(annual_boolean_shortage, structure_name, rights_name, dict_name)
          if use_params:
            self.set_rights_parameters2(mn, structure_name, rights_name, dict_name)
            numsteps = 100
            reg_counter = 0
            estimated_fill = np.zeros(numsteps)
            for x in np.nditer(np.linspace(x_range_2[0], x_range_2[1], num = numsteps)):
              estimated_fill[reg_counter] = self.estimate_logit_syn(x, self.structures_objects[structure_name].rights_objects[rights_name].param_vals_2[dict_name][0], self.structures_objects[structure_name].rights_objects[rights_name].param_vals_2[dict_name][1])
              reg_counter += 1

            index_plots.plot_index_projection(estimated_fill, numsteps, x_range_2, counter1, counter2)
          counter1 += 1
          if counter1 == 3:
            counter1 = 0
            counter2 +=1 

        index_plots.save_fig('UCRB_analysis-master\index_figs\S' + structure_name + 'R' + rights_name + '_calldistance.png')          
        plt.close()

                
    rights_parameter_snow_df = pd.DataFrame(self.right_parameters_snow)
    rights_parameter_snow_df.to_csv('UCRB_analysis-master\index_figs\ind_right_fill_curve_snow.csv')

    rights_parameter_call_df = pd.DataFrame(self.right_parameters_call)
    rights_parameter_call_df.to_csv('UCRB_analysis-master\index_figs\ind_right_fill_curve_call.csv')
    
  def set_rights_parameters(self, ma, structure_name, rights_name, cat_name): 
    self.right_parameters_snow['structure'].append(structure_name)
    self.right_parameters_snow['rights'].append(rights_name)
    self.right_parameters_snow['month'].append(ma)
    self.right_parameters_snow['param1'].append(self.structures_objects[structure_name].rights_objects[rights_name].param_vals[cat_name][0])
    self.right_parameters_snow['param2'].append(self.structures_objects[structure_name].rights_objects[rights_name].param_vals[cat_name][1])
    self.right_parameters_snow['snowmin'].append(self.structures_objects[structure_name].rights_objects[rights_name].snowpack_range[cat_name][0])
    self.right_parameters_snow['snowmax'].append(self.structures_objects[structure_name].rights_objects[rights_name].snowpack_range[cat_name][1])
  def set_rights_parameters2(self, ma, structure_name, rights_name, cat_name): 
    self.right_parameters_call['structure'].append(structure_name)
    self.right_parameters_call['rights'].append(rights_name)
    self.right_parameters_call['month'].append(ma)
    self.right_parameters_call['param1'].append(self.structures_objects[structure_name].rights_objects[rights_name].param_vals_2[cat_name][0])
    self.right_parameters_call['param2'].append(self.structures_objects[structure_name].rights_objects[rights_name].param_vals_2[cat_name][1])
    self.right_parameters_call['callmin'].append(self.structures_objects[structure_name].rights_objects[rights_name].call_range[cat_name][0])
    self.right_parameters_call['callmax'].append(self.structures_objects[structure_name].rights_objects[rights_name].call_range[cat_name][1])
                
  def estimate_logit_syn(self, current_date, min_x, min_z):
    start_val = 0.0
    end_val = 1.0
    logit_val = 1.0/(1.0 + np.exp((-1*min_x)*(current_date - min_z )))

    return logit_val
    
  def calculate_index_regression_snowpack(self, annual_boolean, structure_name, rights_name, dict_name):
  
    snow_data = np.zeros(len(self.structures_objects[structure_name].rights_objects[rights_name].monthly_snowpack[annual_boolean]) + 2)
    fill_data = np.zeros(len(self.structures_objects[structure_name].rights_objects[rights_name].monthly_snowpack[annual_boolean]) + 2)
          
    snow_data[:-2] = self.structures_objects[structure_name].rights_objects[rights_name].monthly_snowpack[annual_boolean]
    fill_data[:-2] = self.structures_objects[structure_name].rights_objects[rights_name].percent_filled[annual_boolean]
          
    snow_data[-2] = -1000
    snow_data[-1] = 1000
          
    fill_data[-2] = 0
    fill_data[-1] = 1
    use_params = True
    
    try:
      popt, pcov = curve_fit(f = self.estimate_logit_syn, xdata = snow_data, ydata = fill_data, bounds = ((0.0, -np.inf),(10.0, np.inf)))
    except:
      use_params = False
    if use_params:
      self.structures_objects[structure_name].rights_objects[rights_name].param_vals[dict_name] = popt    
      self.structures_objects[structure_name].rights_objects[rights_name].snowpack_range[dict_name] = [np.min(self.structures_objects[structure_name].rights_objects[rights_name].monthly_snowpack[annual_boolean]), np.max(self.structures_objects[structure_name].rights_objects[rights_name].monthly_snowpack[annual_boolean])]
    
    return use_params

  def calculate_index_regression_calldistance(self, annual_boolean, structure_name, rights_name, dict_name):
  
    call_data = np.zeros(len(self.structures_objects[structure_name].rights_objects[rights_name].distance_from_call[annual_boolean]) + 2)
    fill_data = np.zeros(len(self.structures_objects[structure_name].rights_objects[rights_name].distance_from_call[annual_boolean]) + 2)
          
    call_data[:-2] = self.structures_objects[structure_name].rights_objects[rights_name].distance_from_call[annual_boolean]
    fill_data[:-2] = self.structures_objects[structure_name].rights_objects[rights_name].percent_filled_single[annual_boolean]
          
    call_data[-2] = -9999
    call_data[-1] = 9999
          
    fill_data[-2] = 0
    fill_data[-1] = 1
    use_params = True
    
    try:
      popt, pcov = curve_fit(f = self.estimate_logit_syn, xdata = call_data, ydata = fill_data)
    except:
      use_params = False
    if use_params:
      self.structures_objects[structure_name].rights_objects[rights_name].param_vals_2[dict_name] = popt    
      self.structures_objects[structure_name].rights_objects[rights_name].call_range[dict_name] = [np.min(self.structures_objects[structure_name].rights_objects[rights_name].distance_from_call[annual_boolean]), np.max(self.structures_objects[structure_name].rights_objects[rights_name].distance_from_call[annual_boolean])]
    
    return use_params


            
  def object_equals(self, other):
    ##This function compares two instances of an object, returns True if all attributes are identical.
    equality = {}
    if (self.__dict__.keys() != other.__dict__.keys()):
      return ('Different Attributes')
    else:
      differences = 0
      for i in self.__dict__.keys():
        if type(self.__getattribute__(i)) is dict:
          equality[i] = True
          for j in self.__getattribute__(i).keys():
            if (type(self.__getattribute__(i)[j] == other.__getattribute__(i)[j]) is bool):
              if ((self.__getattribute__(i)[j] == other.__getattribute__(i)[j]) == False):
                equality[i] = False
                differences += 1
            else:
              if ((self.__getattribute__(i)[j] == other.__getattribute__(i)[j]).all() == False):
                equality[i] = False
                differences += 1
        else:
          if (type(self.__getattribute__(i) == other.__getattribute__(i)) is bool):
            equality[i] = (self.__getattribute__(i) == other.__getattribute__(i))
            if equality[i] == False:
              differences += 1
          else:
            equality[i] = (self.__getattribute__(i) == other.__getattribute__(i)).all()
            if equality[i] == False:
              differences += 1
    return (differences == 0)



