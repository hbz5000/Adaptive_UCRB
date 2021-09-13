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
from plotter import Plotter
from scipy.optimize import curve_fit
import scipy.stats as stats
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap


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

    self.huc_8_list = input_data_dictionary['HUC8']
    self.basin_huc8 = self.clean_join(extended_table8, this_basin, 4326, 'inner', 'within')
    
    self.basin_snowpack = {}
    for huc8_basin in self.huc_8_list:
      self.basin_snowpack[huc8_basin] = pd.read_csv(input_data_dictionary['snow'] + huc8_basin + '.csv', index_col = 0)
      self.basin_snowpack[huc8_basin].index = pd.to_datetime(self.basin_snowpack[huc8_basin].index)
      
    self.basin_structures = gpd.read_file(input_data_dictionary['structures'])
    self.basin_structures = self.basin_structures.to_crs(epsg = 4326)
    print(np.unique(self.basin_structures['StructType']))
    for index, row in self.basin_structures.iterrows():
      if row['WDID'] == '5101310':
        print(row)
    self.structures_objects = {}
    self.structures_list = {}
    self.structures_list['unknown'] = []
    self.structures_list['total'] = []
    for huc8_basin in self.huc_8_list:
      this_watershed = self.basin_huc8[self.basin_huc8['HUC8'] == huc8_basin]
      this_watershed_structures = self.clean_join(self.basin_structures, this_watershed, 4326, 'inner', 'within')
      self.structures_list[huc8_basin] = []
      for index_s, row_s in this_watershed_structures.iterrows():
        self.structures_list[huc8_basin].append(str(row_s['WDID']))
        self.structures_list['total'].append(str(row_s['WDID']))
        self.structures_objects[str(row_s['WDID'])] = Structure(str(row_s['WDID']), huc8_basin)

  def clean_join(self, gdf1, gdf2, crs_int, howstring, opstring):
    #gdf1 = gdf1.to_crs(epsg = crs_int)
    #gdf2 = gdf2.to_crs(epsg = crs_int)
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
    
  def read_ind_structure_deliveries(self, input_data_dictionary, start_year, end_year):
    delivery_data = self.read_text_file(input_data_dictionary['deliveries'])
    structure_name = '5104634'
    call_structure = '3604512'
    monthly_deliveries = np.zeros((105, 12))
    total_delivery = np.zeros(105)
    month_num = {}
    counter = 0
    control_call = {}
    for month_name in ['OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP']:
      month_num[month_name] = counter
      counter += 1
      control_call[month_name] = {}
    for line in delivery_data:
      data = line.split()
      if data:
        if len(data) > 1 and data[0] !='#':
          struct_id = str(data[1].strip())
          if struct_id == structure_name:
            print(data)
            month_id = data[3].strip()
            if month_id in month_num:
              month_number = month_num[month_id]
              if month_number > 2:
                year_id = int(data[2].strip()) - start_year - 1
              else:
                year_id = int(data[2].strip()) - start_year             
              monthly_deliveries[year_id, month_number] = float(data[19].strip())
              total_delivery[year_id] += float(data[19].strip())
          if struct_id == call_structure:
            month_id = data[3].strip()          
            if month_id in month_num:
              call_struct_id = str(data[33].strip())
              if call_struct_id in control_call[month_id]:
                control_call[month_id][call_struct_id] += 1.0/105.0
              else:
                control_call[month_id][call_struct_id] = 1.0/105.0
    print(control_call)
    total_ranking = np.zeros(105, dtype = np.int64)
    argsorted_delivery = np.argsort(total_delivery)
    qalycolors = sns.color_palette('RdYlBu', 105)
    for x in range(0, len(argsorted_delivery)):
      total_ranking[argsorted_delivery[x]] = int(x)
    print(argsorted_delivery)
    print(total_ranking)
    fig, ax = plt.subplots()
    for x in range(0, 105):
      ax.plot(monthly_deliveries[x,:], linewidth = 2.0, color = qalycolors[total_ranking[x]])
    ax.set_xlim([0, 11])
    ax.set_ylim([0, 40000])
    ax.set_ylabel('Monthly TBD', fontsize = 14, weight = 'bold', fontname = 'Gill Sans MT')
    ax.set_xticks([0, 5.5, 11])
    ax.set_xticklabels(['October', 'March', 'September'], fontsize = 14, weight = 'bold', fontname = 'Gill Sans MT')
    cax = fig.add_axes([0.15, 0.785, 0.04, 0.08])
    cmap = pl.cm.gnuplot_r
    my_cmap = ListedColormap(sns.color_palette('RdYlBu').as_hex())
    sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=plt.Normalize(vmin=0, vmax=8))
    # fake up the array of the scalar mappable. Urgh...
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_ticks([0, 8])
    cbar.ax.set_yticklabels(['Fewest Annual TBD (' + str(int(total_delivery[argsorted_delivery[1]]/10000)*10) + ' tAF)', 'Most Annual TBD (' + str(int(total_delivery[argsorted_delivery[104]]/10000)*10) + ' tAF)'], fontsize = 12, weight = 'bold', fontname = 'Gill Sans MT')
    plt.show()
    return control_call
  def create_new_simulation(self, input_data_dictionary, start_year, end_year):
    print('initalize structure timeseries')
    for structure_name in self.structures_objects:
      self.structures_objects[structure_name].monthly_demand = np.zeros((end_year - start_year) * 12)
      self.structures_objects[structure_name].monthly_deliveries = np.zeros((end_year - start_year) * 12)

    print('read files')
    demand_data = self.read_text_file(input_data_dictionary['structure_demand'])
    delivery_data = self.read_text_file(input_data_dictionary['deliveries'])

    print('record demand')
    self.read_demand_data(demand_data)
    print('record deliveries')
    self.read_structure_deliveries(delivery_data, start_year, True)
    print('assign deliveries to rights')
    for structure_name in self.structures_objects:
      self.structures_objects[structure_name].make_sorted_rights_list()
      self.structures_objects[structure_name].assign_demand_rights()
      
    print('locate distance to call')
    call_structure = self.read_call_files(input_data_dictionary['calls'], start_year)
    self.find_senior_downstream_call(call_structure, start_year, end_year, True)    
    self.find_snowpack(start_year, end_year)

        
  def read_demand_data(self, all_split_data_DDM):
    toggle_on = 0
    for j in range(0, len(all_split_data_DDM)):
      if all_split_data_DDM[j][0] == '#':
        toggle_on = 1
      elif toggle_on == 1:
        first_line = int(j * 1)
        toggle_on = 0    
      else:
        this_row = all_split_data_DDM[j].split('.')
        row_data = []
        row_data.extend(this_row[0].split())
        start_year_index = (int(row_data[0].strip()) - 1909) * 12
        structure_name = str(row_data[1].strip())
        self.structures_objects[structure_name].monthly_demand[start_year_index] = float(row_data[2].strip())
        for x in range(1, 12):
          self.structures_objects[structure_name].monthly_demand[start_year_index + x] = float(this_row[x].strip())

  def read_rights_data(self, all_data_DDR):
    column_lengths=[12,24,12,16,8,8]
    split_line = ['']*len(column_lengths)
    character_breaks=np.zeros(len(column_lengths),dtype=int)
    character_breaks[0]=column_lengths[0]
    for i in range(1,len(column_lengths)):
      character_breaks[i]=character_breaks[i-1]+column_lengths[i]
  
    all_rights_name = []
    all_rights_priority = []
    all_rights_decree = []
    all_rights_structure_name = []
    for j in range(0,len(all_data_DDR)):
      if all_data_DDR[j][0] == '#':
        first_line = int(j * 1)
      else:
        split_line[0]=all_data_DDR[j][0:character_breaks[0]]
        for i in range(1,len(split_line)):
          split_line[i]=all_data_DDR[j][character_breaks[i-1]:character_breaks[i]]
        structure_name = str(split_line[2].strip())
        right_name = str(split_line[0].strip())
        right_priority = float(split_line[3].strip())
        if int(split_line[5].strip()) == 1:
          right_decree = float(split_line[4].strip())
        else:
          right_decree = 0.0
        
        all_rights_name.append(right_name)
        all_rights_priority.append(right_priority) 
        all_rights_decree.append(right_decree)
        all_rights_structure_name.append(structure_name)
        if structure_name not in self.structures_objects:
          self.structures_list['unknown'].append(structure_name)
          self.structures_list['total'].append(structure_name)
          self.structures_objects[structure_name] = Structure(structure_name, 'unknown')
        self.structures_objects[structure_name].initialize_right(right_name, right_priority, right_decree)

    priority_order = np.argsort(np.asarray(all_rights_priority))
    self.rights_priority_stack = []
    self.rights_structure_stack = []
    for stack_order in range(0, len(priority_order)):      
      self.rights_priority_stack.append(all_rights_name[priority_order[stack_order]])
      self.rights_structure_stack.append(all_rights_structure_name[priority_order[stack_order]])

  def read_downstream_structure(self, downstream_data):
    column_lengths=[12,24,13,17,4]
    split_line = ['']*len(column_lengths)
    character_breaks=np.zeros(len(column_lengths),dtype=int)
    character_breaks[0]=column_lengths[0]
    for i in range(1,len(column_lengths)):
      character_breaks[i]=character_breaks[i-1]+column_lengths[i]

    id_list = []
    upstream_list = []
    downstream_pairs = {}
    for j in range(0,len(downstream_data)):
      if downstream_data[j][0] == '#':
        first_line = int(j * 1)
      else:
        split_line[0]=downstream_data[j][0:character_breaks[0]]
        for i in range(1,len(split_line)):
          split_line[i]=downstream_data[j][character_breaks[i-1]:character_breaks[i]]
        downstream_pairs[split_line[0].strip()] = split_line[2].strip()

    for structure_id in downstream_pairs:
      if structure_id == 'coloup_end':
        break      
      if structure_id not in self.structures_objects:
        self.structures_list['unknown'].append(structure_id)
        self.structures_list['total'].append(structure_id)
        self.structures_objects[structure_id] = Structure(structure_id, 'unknown')
      
      current_id = downstream_pairs[structure_id]
      self.structures_objects[structure_id].downstream_structures.append(current_id)
      
      while current_id != 'coloup_end' and downstream_pairs[current_id] != 'coloup_end':
        if current_id not in self.structures_objects:
          self.structures_list['unknown'].append(current_id)
          self.structures_list['total'].append(current_id)
          self.structures_objects[current_id] = Structure(current_id, 'unknown')

        self.structures_objects[structure_id].downstream_structures.append(downstream_pairs[current_id])
        current_id = downstream_pairs[current_id]

  def create_rights_stack(self):
    unsorted_rights_stack_priority = []
    unsorted_rights_stack_ids = []
    unsorted_rights_stack_structure_ids = []
    for structure_ids in self.structures_objects:
      for rights_too in self.structures_objects[structure_ids].rights_list:
        unsorted_rights_stack_priority.append(self.structures_objects[structure_ids].rights_objects[rights_too].priority)
        unsorted_rights_stack_ids.append(rights_too)
        unsorted_rights_stack_structure_ids.append(structure_ids)

    sorted_order = np.argsort(np.asarray(unsorted_rights_stack_priority))
    self.rights_stack_priorities = []
    self.rights_stack_ids = []
    self.rights_stack_structure_ids = []
    
    for sorted_order_int in range(0, len(sorted_order)):
      self.rights_stack_priorities.append(unsorted_rights_stack_priority[sorted_order[sorted_order_int]])
      self.rights_stack_ids.append(unsorted_rights_stack_ids[sorted_order[sorted_order_int]])      
      self.rights_stack_structure_ids.append(unsorted_rights_stack_structure_ids[sorted_order[sorted_order_int]])      
              
  def read_structure_deliveries(self, delivery_data, start_year, read_from_file):
    if read_from_file:
      all_deliveries = pd.read_csv('UCRB_analysis-master/Sobol_sample/Experiment_files/monthly_delivery_by_struct.csv', index_col = 0)
      for x in all_deliveries:
        self.structures_objects[x].monthly_deliveries = all_deliveries[x]
    else:
      deliveries_dict = {}
      month_num = {}
      counter = 0
      for month_name in ['OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP']:
        month_num[month_name] = counter
        counter += 1
      for line in delivery_data:
        data = line.split()
        if data:
          if len(data) > 1 and data[0] !='#':
            struct_id = str(data[1].strip())
            if '(' in struct_id or ')' in struct_id or struct_id[0] == '*':
              struct_id = 'donotusethis'
            if struct_id in self.structures_list['total']:
              month_id = data[3].strip()
              if month_id in month_num:
                month_number = month_num[month_id]
                if month_number > 2:
                  year_id = int(data[2].strip()) - start_year - 1
                else:
                  year_id = int(data[2].strip()) - start_year 
            
                total_delivery = float(data[4].strip()) - float(data[17].strip())
                self.structures_objects[struct_id].monthly_deliveries[year_id * 12 + month_number]  = total_delivery * 1.0
      for x in self.structures_objects:
        deliveries_dict[x] = self.structures_objects[x].monthly_deliveries
      month_count = 10
      year_count = start_year * 1
      datetime_index = []
      for date_count in range(0, len(self.structures_objects[x].monthly_deliveries)):
        datetime_index.append(datetime(year_count, month_count, 1, 0, 0))
        month_count += 1
        if month_count == 13:
          month_count = 1
          year_count += 1
      deliveries_df = pd.DataFrame(deliveries_dict, index = datetime_index)
      deliveries_df.to_csv('UCRB_analysis-master/Sobol_sample/Experiment_files/monthly_delivery_by_struct.csv')
          
  def read_call_files(self, call_file, start_year):
    call_structs_df = pd.DataFrame(columns = ['year', 'month', 'structure', 'right', 'priority'])
    counter = 0
    month_num = {}
    for month_name in ['OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP']:
      month_num[month_name] = counter
      counter += 1
      
    data_col = False
    call_struct = []
    year_call = []
    month_call = []
    call_right = []
    call_priority = []
    with open (call_file, 'rt') as xca_file:
      for line in xca_file:
        data = line.split()
        if data:
          if data[0] != '#' and data_col:
            if len(self.structures_objects[data[5].strip()].sorted_rights) > 0:
              for ind_right in self.structures_objects[data[5].strip()].sorted_rights:
                if month_num[data[2].strip()] > 2:
                  month_step = int(data[1].strip()) - start_year - 1 + month_num[data[2].strip()]
                else:
                  month_step = int(data[1].strip()) - start_year + month_num[data[2].strip()]
                if self.structures_objects[data[5].strip()].rights_objects[ind_right].monthly_demand[month_step] > self.structures_objects[data[5].strip()].rights_objects[ind_right].monthly_deliveries[month_step]:
                  break
              call_struct.append(data[5].strip())
              call_priority.append(self.structures_objects[data[5].strip()].rights_objects[ind_right].priority)
              call_right.append(ind_right)
              year_call.append(int(data[1].strip()))
              month_call.append(data[2].strip())
          else:
            data_col = True
            try:
              int(data[1])
            except:
              data_col = False
    
    call_structs_df['year'] = year_call
    call_structs_df['month'] = month_call
    call_structs_df['structure'] = call_struct
    call_structs_df['right'] = call_right
    call_structs_df['priority'] = call_priority

    return call_structs_df
    
  def find_senior_downstream_call(self, call_structure, start_year, end_year, read_from_file):

    if read_from_file:
      for structure_name in self.structures_objects:
        structure_call_distance = pd.read_csv('UCRB_analysis-master/Sobol_sample/Experiment_files/call_distances/' + structure_name + '.csv', index_col = 0)
        for rights_name in structure_call_distance:
          self.structures_objects[structure_name].rights_objects[rights_name].distance_from_call = structure_call_distance[rights_name] / 10000.0
    else:     
      month_list = ['OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP']
      total_counter = 0

      for year_num in range(start_year, end_year):
        year_add = 0
        for month_name in month_list:
          if month_name == 'JAN':
            year_add += 1
          this_month_calls = call_structure.loc[np.logical_and(call_structure['year'] == (year_num + year_add), call_structure['month'] == month_name)]  
          calling_structures = []
          calling_rights = []

          for index_cm, row_cm in this_month_calls.iterrows():
            structure_name = row_cm['structure']
            calling_structures.append(structure_name)
            for ind_right in self.structures_objects[structure_name].sorted_rights:
              if self.structures_objects[structure_name].rights_objects[ind_right].monthly_demand[total_counter] > self.structures_objects[structure_name].rights_objects[ind_right].monthly_deliveries[total_counter] * 1.0001:
                break          
            calling_rights.append(self.structures_objects[structure_name].rights_objects[ind_right].priority)
          for structure_name in self.structures_objects:
            self.structures_objects[structure_name].msdr = self.structures_objects[structure_name].find_senior_downstream_call(calling_structures, calling_rights)
            for rights_name in self.structures_objects[structure_name].rights_list:
              if self.structures_objects[structure_name].msdr < this_right_priority:
                self.structures_objects[structure_name].rights_objects[rights_name].constraining_call = self.structures_objects[structure_name].msdr * 1.0
              else:
                self.structures_objects[structure_name].rights_objects[rights_name].constraining_call = -1
                
          for structure_name in self.structures_objects:
            for rights_name in self.structures_objects[structure_name].rights_list:
              this_right_priority = self.structures_objects[structure_name].rights_objects[rights_name].priority * 1.0
              if self.structures_objects[structure_name].msdr > this_right_priority:
                stack_distance_boolean = np.logical_and(np.asarray(self.rights_stack_priorities) < self.structures_objects[structure_name].msdr, np.asarray(self.rights_stack_priorities) > this_right_priority)
                for stack_counter, ind_right, ind_right_struct in zip(stack_distance_boolean, self.rights_stack_ids, self.rights_stack_structure_ids):
                  if stack_counter:
                    if ind_right_struct in self.structures_objects[structure_name].downstream_structures or self.structures_objects[ind_right_struct].msdr == self.structures_objects[structure_name].msdr:
                      self.structures_objects[structure_name].rights_objects[rights_name].distance_from_call[total_counter] += self.structures_objects[ind_right_struct].rights_objects[ind_right].monthly_demand[total_counter]                
              else:
                stack_distance_boolean = np.logical_and(np.asarray(self.rights_stack_priorities) > self.structures_objects[structure_name].msdr, np.asarray(self.rights_stack_priorities) < this_right_priority)
                for stack_counter, ind_right, ind_right_struct in zip(stack_distance_boolean, self.rights_stack_ids, self.rights_stack_structure_ids):
                  if stack_counter:
                    if ind_right_struct in self.structures_objects[structure_name].downstream_structures or self.structures_objects[ind_right_struct].msdr == self.structures_objects[structure_name].msdr:
                      self.structures_objects[structure_name].rights_objects[rights_name].distance_from_call[total_counter] -= self.structures_objects[ind_right_struct].rights_objects[ind_right].monthly_demand[total_counter]

          total_counter += 1
      for structure_name in self.structures_objects:
        distance_dict = {}
        for rights_name in self.structures_objects[structure_name].rights_objects:
          distance_dict[rights_name] = self.structures_objects[structure_name].rights_objects[rights_name].distance_from_call
        distance_df = pd.DataFrame(distance_dict)
        distance_df.to_csv('UCRB_analysis-master/Sobol_sample/Experiment_files/call_distances/' + structure_name + '.csv')
        
  def find_snowpack(self, start_year, end_year):

    for struct_name in self.structures_objects:
      if self.structures_objects[struct_name].basin == 'unknown' or self.structures_objects[struct_name].basin == '14010005':
        watershed_snow = self.basin_snowpack['14010001']
      else:
        watershed_snow = self.basin_snowpack[self.structures_objects[struct_name].basin]
      year_counter = 0
      month_counter = 10
      no_keys = True
      try:
        key_list = next(iter(self.structures_objects[struct_name].rights_objects))
      except:
        no_keys = False
      if no_keys:
        for total_counter in range(0, len(self.structures_objects[struct_name].rights_objects[key_list].monthly_snowpack)):
          current_date = str(start_year + year_counter) + '-' + str(month_counter).zfill(2) + '-01'
          for ind_right in self.structures_objects[struct_name].rights_list:
            self.structures_objects[struct_name].rights_objects[ind_right].monthly_snowpack[total_counter] = watershed_snow.loc[current_date, 'basinwide_average']
            if self.structures_objects[struct_name].basin == '14010004' and self.structures_objects[struct_name].rights_objects[ind_right].monthly_snowpack[total_counter] < -100:
              self.structures_objects[struct_name].rights_objects[ind_right].monthly_snowpack[total_counter] = self.basin_snowpack['14010003'].loc[current_date, 'basinwide_average']
            elif self.structures_objects[struct_name].basin == '14010002' and self.structures_objects[struct_name].rights_objects[ind_right].monthly_snowpack[total_counter] < -100:
              self.structures_objects[struct_name].rights_objects[ind_right].monthly_snowpack[total_counter] = self.basin_snowpack['14010001'].loc[current_date, 'basinwide_average']
              
          month_counter += 1
          if month_counter == 13:
            year_counter += 1
            month_counter = 1
            
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
        print(rights_counter)
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
        print(rights_counter)
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



