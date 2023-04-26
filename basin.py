import numpy as np 
import pandas as pd
import geopandas as gpd
from datetime import datetime
from structure import Structure
from reservoir import Reservoir
from scipy.stats import norm


class Basin():

  def __init__(self):
    #initialize basin dictionaries
    self.basin_snowpack = {}
    self.structures_objects = {}
    self.structures_list = {}
    self.structures_list['unknown'] = []
    self.structures_list['total'] = []

  def load_basin_snowpack(self, input_data_dictionary):
    #group snowpack into HUC8 hydrologic units
    self.huc_8_list = input_data_dictionary['HUC8']
    #read in monthly snowpack data from each basin, set index to datetime
    for huc8_basin in self.huc_8_list:
      self.basin_snowpack[huc8_basin] = pd.read_csv(input_data_dictionary['snow'] + huc8_basin + '.csv', index_col = 0)
      self.basin_snowpack[huc8_basin].index = pd.to_datetime(self.basin_snowpack[huc8_basin].index)

  def create_reservoir(self, reservoir_name, idnum, capacity):
    self.structures_objects[idnum] = Reservoir(reservoir_name, idnum, capacity)
    self.reservoir_list.append(idnum)

  def set_rights_to_reservoirs(self, rights_name_list, structure_name_list, rights_priority_list, rights_decree_list, fill_type_list):
    counter = 0
    #create reservoir objects and link their water rights to the object
    for rights_name, reservoir_name, rights_priority, rights_decree, fill_type in zip(rights_name_list, structure_name_list, rights_priority_list, rights_decree_list, fill_type_list):
      if reservoir_name not in self.structures_objects:
        self.structures_objects[reservoir_name] = Reservoir('small_count_' + str(counter), reservoir_name, rights_decree)
        counter += 1
      self.structures_objects[reservoir_name].initialize_right(rights_name, rights_priority, rights_decree, fill_type = fill_type)
      
  def set_rights_to_structures(self, rights_name_list, structure_name_list, rights_priority_list, rights_decree_list):
    #create structure object and link water rights to the object
    for rights_name, structure_name, rights_priority, rights_decree in zip(rights_name_list, structure_name_list, rights_priority_list, rights_decree_list):
      if structure_name not in self.structures_objects:
        self.structures_list['unknown'].append(structure_name)
        self.structures_list['total'].append(structure_name)
        self.structures_objects[structure_name] = Structure(structure_name, 'unknown')
      self.structures_objects[structure_name].initialize_right(rights_name, rights_priority, rights_decree)

  def combine_rights_data(self, structure_rights_name, structure_name, structure_rights_priority, structure_rights_decree, reservoir_rights_name, reservoir_name, reservoir_rights_priority, reservoir_rights_decree, instream_rights_name, instream_rights_structure_name, instream_rights_priority, instream_rights_decree):
    #take lists of water rights ids, structure ids, right priority, and right decree and sort them by right priority
    #to create sorted lists of rights, their diversion structure, and the decreed rate of diversion for the entire basin
    #start w/ individual, unordered lists of reservoirs, instream, and structures - need a single ordered list

    #add reservoir rights list to list of structures
    structure_rights_priority.extend(reservoir_rights_priority)
    structure_rights_name.extend(reservoir_rights_name)
    structure_name.extend(reservoir_name)
    structure_rights_decree.extend(reservoir_rights_decree)

    #add structure rights to list of structures/reservoirs
    structure_rights_priority.extend(instream_rights_priority)
    structure_rights_name.extend(instream_rights_name)
    structure_name.extend(instream_rights_structure_name)
    structure_rights_decree.extend(instream_rights_decree)
  
    #order by priority
    priority_order = np.argsort(np.asarray(structure_rights_priority))
    #lists are saved as elements of the basin class
    self.rights_stack_structure_ids = []
    self.rights_decree_stack = []
    self.rights_stack_ids = []
    self.rights_stack_priorities = []
    #rearrange all lists
    for stack_order in range(0, len(priority_order)):      
      self.rights_stack_ids.append(structure_rights_name[priority_order[stack_order]])
      self.rights_stack_structure_ids.append(structure_name[priority_order[stack_order]])
      self.rights_decree_stack.append(structure_rights_decree[priority_order[stack_order]])
      self.rights_stack_priorities.append(structure_rights_priority[priority_order[stack_order]])

  def set_structure_demands(self, structure_demands, structure_demands_adaptive = 'none', use_rights = True):
    len_hist_demands = 0
    #set demands at specific structure objects using timeseries data read from StateMod inputs
    for structure_name in structure_demands:
      #either create a new structure or set historical monthly demands
      #adaptve monthly demands (for informal transfers) begin as copies of the baseline demands
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
        
    #set demands to individual water rights (i.e., first X AF/month of demand to most senior Y CFS of water rights, etc.)
    for structure_name in self.structures_objects:
      #check if struture has demands - if not set to zero
      try:
        len_hist_demands = len(self.structures_objects[structure_name].historical_monthly_demand)
      except:
        self.structures_objects[structure_name].historical_monthly_demand = pd.DataFrame(np.zeros(len(structure_demands.index)), index = structure_demands.index, columns = ['demand',])
        self.structures_objects[structure_name].adaptive_monthly_demand = self.structures_objects[structure_name].historical_monthly_demand.copy(deep = True)
      #sort rights by priority, then assign demand
      if use_rights:
        #sort all rights associated with the structure from most senior to lease senior
        self.structures_objects[structure_name].make_sorted_rights_list()
        if isinstance(structure_demands_adaptive, pd.DataFrame):
          self.structures_objects[structure_name].use_adaptive = True
        self.structures_objects[structure_name].assign_demand_rights()

  def set_structure_deliveries(self, structure_deliveries, structure_deliveries_adaptive = 'none', use_rights = True):
    len_hist_deliveries = 0
    #set deliveries at specific structure objects using timeseries data read from StateMod inputs
    for structure_name in structure_deliveries:
      #deliveries dataframe columns: 'structure id' = total deliveries; 'structure id'_priority = deliveries from priority water rights; 'structure_id'_return = return flows from that structure's diversions; 'structure id'_flow = flow past structure
      #for each structure, make dataframe with all 4 timeseries
      #if no adaptive delivery inputs - use a copy of the baseline deliveries
      if structure_name[-8:] != 'priority' and structure_name[-6:] != 'return' and structure_name[-4:] != 'flow':
        if structure_name in self.structures_objects:
          self.structures_objects[structure_name].historical_monthly_deliveries = pd.DataFrame(structure_deliveries[[structure_name, structure_name + '_priority', structure_name + '_return', structure_name + '_flow']].values, index = structure_deliveries.index, columns = ['deliveries','priority', 'return', 'flow'])
          try:
            self.structures_objects[structure_name].adaptive_monthly_deliveries = pd.DataFrame(structure_deliveries_adaptive[[structure_name, structure_name + '_priority', structure_name + '_return', structure_name + '_flow']].values, index = structure_deliveries_adaptive.index, columns = ['deliveries','priority', 'return', 'flow'])
          except:        
            self.structures_objects[structure_name].adaptive_monthly_deliveries = self.structures_objects[structure_name].historical_monthly_deliveries.copy(deep = True)
        
        else:
          self.structures_objects[structure_name] = Structure(structure_name, 'unknown')
          self.structures_objects[structure_name].historical_monthly_demand = pd.DataFrame(np.zeros(len(structure_deliveries.index)), index = structure_deliveries.index, columns = ['demand',])
          self.structures_objects[structure_name].adaptive_monthly_demand = pd.DataFrame(np.zeros(len(structure_deliveries.index)), index = structure_deliveries.index, columns = ['demand',])
          self.structures_objects[structure_name].make_sorted_rights_list()
          self.structures_objects[structure_name].historical_monthly_deliveries = pd.DataFrame(structure_deliveries[[structure_name, structure_name + '_priority', structure_name + '_return', structure_name + '_flow']].values, index = structure_deliveries.index, columns = ['deliveries','priority', 'return', 'flow'])
          try:
            self.structures_objects[structure_name].adaptive_monthly_deliveries = pd.DataFrame(structure_deliveries_adaptive[[structure_name, structure_name + '_priority', structure_name + '_return', structure_name + '_flow']].values, index = structure_deliveries_adaptive.index, columns = ['deliveries', 'priority', 'return', 'flow'])
          except:
            self.structures_objects[structure_name].adaptive_monthly_deliveries = self.structures_objects[structure_name].historical_monthly_deliveries.copy(deep = True)

    #assign deliveries at each structure to the individual water rights (i.e., first X AF/month of deliveries to most senior Y CFS of water rights, etc.)
    for structure_name in self.structures_objects:
      no_xdd = False
      try:
        len_hist_demands = len(self.structures_objects[structure_name].historical_monthly_deliveries)
      except:
        no_xdd = True
      if no_xdd:
        #check if struture has deliveries - if not set to zero
        self.structures_objects[structure_name].historical_monthly_deliveries = pd.DataFrame(np.zeros((len(structure_deliveries.index), 4)), index = structure_deliveries.index, columns = ['deliveries','priority', 'return', 'flow'])
        self.structures_objects[structure_name].adaptive_monthly_deliveries = self.structures_objects[structure_name].historical_monthly_deliveries.copy(deep = True)
      if isinstance(structure_deliveries_adaptive, pd.DataFrame):
        self.structures_objects[structure_name].use_adaptive = True
      if use_rights:
        self.structures_objects[structure_name].assign_delivery_rights()

  def set_return_fractions(self, structure_returns):
    #set average monthly return flows (as a fraction of diversions) at each structure
    #take values from dataframe input and make new dataframes for individual structures in the structure objects
    for structure_name in self.structures_objects:
      self.structures_objects[structure_name].return_fraction = pd.DataFrame(np.ones(len(structure_returns.index)), index = structure_returns.index, columns = [structure_name,])
    for structure_name in structure_returns:
      if structure_name in self.structures_objects:
        self.structures_objects[structure_name].return_fraction = pd.DataFrame(structure_returns[structure_name].values, index = structure_returns.index, columns = [structure_name,])

  def set_structure_types(self, input_data_dictionary):
    #get list of structures, irrigation nodes, and ditch nodes from input files
    #note: these lists contain smaller structures that are aggregated into aggregate nodes in StateMod
    structures_ucrb = gpd.read_file(input_data_dictionary['structures'])
    irrigation_ucrb = gpd.read_file(input_data_dictionary['irrigation'])
    ditches_ucrb = gpd.read_file(input_data_dictionary['ditches'])
    other_structures = list(structures_ucrb['WDID'].astype(str))
    irrigation_structures = list(irrigation_ucrb['SW_WDID1'].astype(str))
    ditch_structures = list(ditches_ucrb['wdid'].astype(str))
    
    #get list of individual structure ids that make up each 'aggregated' structure in StateMod
    #the dictionary keys are the aggregated StateMod structure ids, the lists are the individual structures listed in the structure lists
    agg_diversions = pd.read_csv(input_data_dictionary['aggregated_diversions'])
    aggregated_diversions = {}
    for index, row in agg_diversions.iterrows():
      if row['statemod_diversion'] in aggregated_diversions:
        aggregated_diversions[row['statemod_diversion']].append(str(row['individual_diversion']))
      else:
        aggregated_diversions[row['statemod_diversion']] = [str(row['individual_diversion']), ]

    #assign a 'type' to each structure object
    for structure_name in self.structures_objects:
      self.structures_objects[structure_name].acreage = {}
      self.structures_objects[structure_name].structure_types = []      
      #get list of individual structures (equal to the structure name for normal StateMod structure objects, list of many small structures for aggregated StateMod structure object)
      if structure_name in aggregated_diversions:
        ind_structure_list = aggregated_diversions[structure_name]
      else:
        ind_structure_list = [structure_name,]    
      #for each individual structure, assign type based on the 'list' its in, within specific values assigned to those not in particular structure files
      #if structure is irrigation, find total crop acreage
      for ind_structure in ind_structure_list:
        #if structure is in irrigation list, its an irrigation type
        if ind_structure in irrigation_structures:
          self.structures_objects[structure_name].structure_types.append('Irrigation')
          this_irrigated_area = irrigation_ucrb[irrigation_ucrb['SW_WDID1'] == ind_structure]
          for index_ia, row_ia in this_irrigated_area.iterrows():
            if row_ia['CROP_TYPE'] in self.structures_objects[structure_name].acreage:
              self.structures_objects[structure_name].acreage[row_ia['CROP_TYPE']] += row_ia['ACRES']
            else:
              self.structures_objects[structure_name].acreage[row_ia['CROP_TYPE']] = row_ia['ACRES']              
        #if structure is in ditch list, its also an irrigation type
        elif ind_structure in ditch_structures:
          self.structures_objects[structure_name].structure_types.append('Irrigation')
          #for ditch structures, estimate acreage from the total decree value (excepting decrees for the 'max' diversion (decree = 999.9 cfs)
          total_decree = 0.0
          for right_name in self.structures_objects[structure_name].rights_list:
            if np.mean(self.structures_objects[structure_name].rights_objects[right_name].decree_af) < 990.0 * 1.98 * 30.0:
              total_decree += np.mean(self.structures_objects[structure_name].rights_objects[right_name].decree_af)
          total_acreage = 0.0
          for crop_name in self.structures_objects[structure_name].acreage:
            total_acreage += self.structures_objects[structure_name].acreage[crop_name]
          implied_acreage = max(total_decree - total_acreage * 3.5 /6.0, 0.0)
          #assume its for alfalfa
          if 'ALFALFA' in self.structures_objects[structure_name].acreage:
            self.structures_objects[structure_name].acreage['ALFALFA'] += implied_acreage
          else:
            self.structures_objects[structure_name].acreage['ALFALFA'] = implied_acreage
        #if structure is in other structure types, assign type based on data about structure
        elif ind_structure in other_structures:
          this_structure = structures_ucrb[structures_ucrb['WDID'] == ind_structure]
          if this_structure.loc[this_structure.index[0], 'StructType'] == 'Reservoir':
            self.structures_objects[structure_name].structure_types.append('Reservoir')
          elif this_structure.loc[this_structure.index[0], 'StructType'] == 'Minimum Flow':
            self.structures_objects[structure_name].structure_types.append('Minimum Flow')
          elif this_structure.loc[this_structure.index[0], 'StructType'] == 'Power Plant':
            self.structures_objects[structure_name].structure_types.append('Municipal')            
          else:
            #assign remainder individually
            if np.sum(self.structures_objects[structure_name].historical_monthly_demand['demand']) == 0.0:
              self.structures_objects[structure_name].structure_types.append('Carrier')
            elif structure_name == '36000662_D' or structure_name == '38000880_D' or structure_name == '5000734_D' or structure_name == '5300555_D' or structure_name == '3900532' or structure_name == '5100941':
              self.structures_objects[structure_name].structure_types.append('Irrigation')
            elif structure_name[:4] == '3600' or structure_name[:4] == '3601' or structure_name[:4] == '3700' or structure_name[:4] == '3800' or structure_name[:4] == '3801' or structure_name[:4] == '5300' or structure_name[:4] == '7200' or structure_name[:4] == '7201' or structure_name == '3900967' or structure_name == '5100958' or structure_name == '5101070':
              self.structures_objects[structure_name].structure_types.append('Municipal')
            elif structure_name[:4] == '3604' or structure_name[:4] == '3704' or structure_name[:4] == '3804' or structure_name[:4] == '5104' or structure_name[:4] == '7204':
              self.structures_objects[structure_name].structure_types.append('Export')
            elif structure_name[:6] == '36_ADC' or  structure_name[:6] == '37_ADC' or structure_name[:6] == '39_ADC' or structure_name[:6] == '45_ADC' or structure_name[:6] == '50_ADC' or structure_name[:6] == '51_ADC' or structure_name[:6] == '52_ADC' or structure_name[:6] == '53_ADC' or structure_name[:6] == '70_ADC' or structure_name[:6] == '72_ADC':
              self.structures_objects[structure_name].structure_types.append('Irrigation')
        #for others, assign types individually
        else:
          if ind_structure[3:6] == 'ARC' or ind_structure[3:6] == 'ASC' or ind_structure[-2:] == 'HU' or ind_structure == '7203904AG' or ind_structure == '7204033AG' or ind_structure == '36GMCON':
            self.structures_objects[structure_name].structure_types.append('Reservoir')
          elif ind_structure[3:6] == 'AMC' or ind_structure[-2:] == 'PL' or  ind_structure[:7] == '5003668' or ind_structure == '70FD1' or ind_structure == '70FD2' or ind_structure == 'CSULimitPLN' or ind_structure == 'HUPLimitPLN' or ind_structure == 'ColRivPln' or ind_structure == '3903508_Ex':
            self.structures_objects[structure_name].structure_types.append('None')
          elif ind_structure[-2:] == '_I':
            self.structures_objects[structure_name].structure_types.append('Irrigation')
          elif ind_structure[-2:] == '_M' or ind_structure == 'MoffatBF' or ind_structure == 'Baseflow' or ind_structure == '3702059_2' or ind_structure == '5300584P' or ind_structure == '3804625M2' or ind_structure == '7202001_2' or ind_structure[:2] == '09' or ind_structure[-3:] == 'Dwn':
            self.structures_objects[structure_name].structure_types.append('Minimum Flow')
          elif ind_structure == '3604683SU' or ind_structure == '3804625SU':
            self.structures_objects[structure_name].structure_types.append('Export')
          elif ind_structure == '36_KeyMun' or ind_structure[:6] == '37VAIL' or ind_structure[:7] == '3803713' or ind_structure == '4200520' or ind_structure == '4200541' or ind_structure == '72_GJMun' or ind_structure == '72_UWCD' or ind_structure == 'ChevDem':
            self.structures_objects[structure_name].structure_types.append('Municipal')
          elif ind_structure[:7] == '7200813':
            if ind_structure == '7200813':
              self.structures_objects[structure_name].structure_types.append('Irrigation')
            else:
              self.structures_objects[structure_name].structure_types.append('Municipal')              

  def set_structure_inflows(self, structure_inflows, structure_inflows_adaptive = 'none'):
    #this sets the baseline simulation output timeseries for structure inflows and controlling call locations to the structure objects
    #the 'adaptive' timeseries are initialized with the output from the baseline simulation
    for structure_name in self.structures_objects:
      #set inflows
      if structure_name + '_inflow' in structure_inflows:
        self.structures_objects[structure_name].historical_monthly_inflows = pd.DataFrame(structure_inflows[structure_name + '_inflow'].values, index = structure_inflows.index, columns = ['inflows',])
        try:
          self.structures_objects[structure_name].adaptive_monthly_inflows = pd.DataFrame(structure_inflows_adaptive[structure_name + '_inflow'].values, index = structure_inflows_adaptive.index, columns = ['inflows',])
        except:
          self.structures_objects[structure_name].adaptive_monthly_inflows = self.structures_objects[structure_name].historical_monthly_inflows.copy(deep = True)
      #set to -999 if no data
      else:
        self.structures_objects[structure_name].historical_monthly_inflows = pd.DataFrame(np.ones(len(structure_inflows.index))*(-999.0), index = structure_inflows.index, columns = ['inflows',])
        self.structures_objects[structure_name].adaptive_monthly_inflows = pd.DataFrame(np.ones(len(structure_inflows.index))*(-999.0), index = structure_inflows.index, columns = ['inflows',])
      #set call structure locations
      if structure_name + '_location' in structure_inflows:
        self.structures_objects[structure_name].historical_monthly_control = pd.DataFrame(list(structure_inflows[structure_name + '_location']), index = structure_inflows.index, columns = ['location',])
        self.structures_objects[structure_name].adaptive_monthly_control = self.structures_objects[structure_name].historical_monthly_control.copy(deep = True)
      #set to -999 if no data
      else:
        self.structures_objects[structure_name].historical_monthly_control = pd.DataFrame(np.ones(len(structure_inflows.index))*(-999.0), index = structure_inflows.index, columns = ['location',])
        self.structures_objects[structure_name].adaptive_monthly_control = pd.DataFrame(np.ones(len(structure_inflows.index))*(-999.0), index = structure_inflows.index, columns = ['location',])
      if structure_name + '_available' in structure_inflows:
        self.structures_objects[structure_name].historical_monthly_available = pd.DataFrame(list(structure_inflows[structure_name + '_available']), index = structure_inflows.index, columns = ['available',])
        self.structures_objects[structure_name].adaptive_monthly_available = self.structures_objects[structure_name].historical_monthly_control.copy(deep = True)
      #set to -999 if no data
      else:
        self.structures_objects[structure_name].historical_monthly_available = pd.DataFrame(np.ones(len(structure_inflows.index))*(-999.0), index = structure_inflows.index, columns = ['available',])
        self.structures_objects[structure_name].adaptive_monthly_available = pd.DataFrame(np.ones(len(structure_inflows.index))*(-999.0), index = structure_inflows.index, columns = ['available',])

  def set_plan_flows(self, plan_flows, downstream_data, start_point_list, end_point):
    #make a dictionary where each reservoir (start_point_list) is a key to a list
    #that contains all of the stations downstream of that point
    station_id_column_length = 12
    station_id_column_start = 0
    #the dictionary is saved as an element of the basin
    self.plan_flows_list = {}
    for j in range(0,len(downstream_data)):
      if downstream_data[j][0] != '#':
        first_line = int(j * 1)
        break
    for j in range(first_line, len(downstream_data)):
      station_id = str(downstream_data[j][station_id_column_start:station_id_column_length].strip())
      if station_id == end_point:
        break
      if station_id in start_point_list:
        self.plan_flows_list[station_id] = []
      for x in self.plan_flows_list:
        self.plan_flows_list[x].append(station_id)
        
    #initialize a dataframe for each structure with a timeseries of zero values
    for station_name in self.structures_objects:
      self.structures_objects[station_name].routed_plans = pd.DataFrame(index = plan_flows.index, columns = ['plan',])
      self.structures_objects[station_name].routed_plans['plan'] = np.zeros(len(plan_flows.index))
    
    #using the list of downstream stations from every reservoir release,
    #sum the total reservoir releases happening upstream of the station
    for index, row in plan_flows.iterrows():
      for station_start in self.plan_flows_list:
        for station_use in self.plan_flows_list[station_start]:
          self.structures_objects[station_use].routed_plans.loc[index, 'plan'] += row[station_start] * 1.0

  def adjust_structure_deliveries(self):
    #for structures where demand is set to 9999999.9, set monthly demand to maximum monthly delivery in the baseline simulation
    monthly_deliveries = {}
    for structure_name in self.structures_objects:
      #find 'max' demand structures
      if np.max(self.structures_objects[structure_name].historical_monthly_demand['demand']) > 900000.0:
        monthly_deliveries[structure_name] = np.zeros(12)
        #find maximum delivery in that month
        for index, row in self.structures_objects[structure_name].historical_monthly_deliveries.iterrows():
          monthly_deliveries[structure_name][index.month - 1] = max(row['deliveries'], monthly_deliveries[structure_name][index.month - 1])
        #reset maximum historical and adaptive monthly demands
        for datetime_val in self.structures_objects[structure_name].historical_monthly_demand.index:
          self.structures_objects[structure_name].historical_monthly_demand.loc[datetime_val, 'demand'] = monthly_deliveries[structure_name][datetime_val.month - 1] * 1.0
          self.structures_objects[structure_name].adaptive_monthly_demand.loc[datetime_val, 'demand'] = monthly_deliveries[structure_name][datetime_val.month - 1] * 1.0

    return monthly_deliveries
          


  def make_snow_regressions(self, snowpack_basin, historical_monthly_control, simulated_reservoir_timeseries, historical_monthly_available, res_station, year_start, year_end):
    #this function makes regression equations to calculate the relationship between snowpack
    #and future flow available for storage at a reservoir based on the month & if there are any 'calls' on the river
    
    snowpack_data = self.basin_snowpack[snowpack_basin]
    month_index = ['OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP']
    monthly_control = {}
    monthly_control_int = {}
    for x in range(1, 13):
      monthly_control[x] = []
      monthly_control_int[x] = []
    #make a list of all the 'control' locations making calls at the reservoir
    for index, row in historical_monthly_control.iterrows():
      if index > datetime(1950, 10, 1, 0, 0):
        this_row_month = index.month
        monthly_control_int[this_row_month].append(row['location'])
    
    #if those locations show up in more than 10 years, develop
    #a snowpack/flow regression for years with that specific control location
    for x in range(1,13):
      total_list = list(set(monthly_control_int[x]))
      monthly_control[x] = []
      for cont_loc in total_list:
        num_obs = 0
        #find how many times a given control location
        #was observed in the historical record
        for all_loc in monthly_control_int[x]:
          if all_loc == cont_loc:
            num_obs += 1
        #if more than 10, save as part of list for regressions
        if num_obs > 10:
          monthly_control[x].append(cont_loc)

    coef = {}
    #individual regressions in each month
    for month_num in range(0, 12):
      year_add = 0
      month_start = 10
      if month_start + month_num > 12:
        month_start = -2
        year_add = 1
      #find list of control locations to develop regressions for
      control_location_list = monthly_control[month_start + month_num]
      flow2 = {}
      snowpack = {}
      #relationship is between snowpack and flow
      for x in control_location_list:
        flow2[x] = []
        snowpack[x] = []
      #years where the call location has showed up in <10 other years use a regression with observations from the entire record
      flow2['all'] = []
      snowpack['all'] = []
      
      #in each year of the simulation, get observations of snowpack and 'usable flow' for reservoir storage
      for year_num in range(year_start, year_end):
        datetime_val = datetime(year_num + year_add, month_start + month_num, 1, 0, 0)
        remaining_usable_flow = 0.0
        current_snowpack = snowpack_data.loc[datetime_val, 'basinwide_average']#snowpack observation
        control_location = historical_monthly_control.loc[datetime_val, 'location']#call location classification
        
        #find remaining flow that can be impounded at the reservoir
        for lookahead_month in range(month_num, 12):
          if lookahead_month > 2:
            lookahead_datetime = datetime(year_num + 1, lookahead_month - 2, 1, 0, 0)
          else:
            lookahead_datetime = datetime(year_num, lookahead_month + 10, 1, 0, 0)
          remaining_usable_flow += float(simulated_reservoir_timeseries.loc[lookahead_datetime, res_station + '_diversions']) + float(historical_monthly_available.loc[lookahead_datetime, 'available'])
          
        #add the snow/flow observations to the list corresponding with the call location
        if remaining_usable_flow < 0.0 or pd.isna(current_snowpack):
          skip_this = 1
        else:
          if control_location in flow2:
            snowpack[control_location].append(current_snowpack)
            flow2[control_location].append(remaining_usable_flow)
          snowpack['all'].append(current_snowpack)
          flow2['all'].append(remaining_usable_flow)

      #for each list, make a linear regression between snow (obs) and flow (predicted)
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

  def update_structure_demand_delivery(self, new_demands, new_deliveries, monthly_max, date_use):
    #this function takes a dictionary of delivery data and a dictionary of demand data, with single values from a single simulation timestep and updates struture object timeseries data
    #update demands
    for this_structure in new_demands:
      if this_structure in monthly_max:
        self.structures_objects[this_structure].adaptive_monthly_demand.loc[date_use, 'demand'] = min(new_demands[this_structure], monthly_max[this_structure][date_use.month - 1])
      else:
        self.structures_objects[this_structure].adaptive_monthly_demand.loc[date_use, 'demand'] = new_demands[this_structure] * 1.0
      #take updated structure demands and assign the updates to individual rights
      self.structures_objects[this_structure].update_demand_rights(date_use)
    #update delivery
    for this_structure in new_deliveries:
      if this_structure[-8:] != 'priority' and this_structure[-6:] != 'return' and this_structure[-4:] != 'flow':
        self.structures_objects[this_structure].adaptive_monthly_deliveries.loc[date_use, 'deliveries'] = new_deliveries[this_structure] * 1.0
        self.structures_objects[this_structure].adaptive_monthly_deliveries.loc[date_use, 'priority'] = new_deliveries[this_structure + '_priority'] * 1.0
        self.structures_objects[this_structure].adaptive_monthly_deliveries.loc[date_use, 'return'] = new_deliveries[this_structure + '_return'] * 1.0
        self.structures_objects[this_structure].adaptive_monthly_deliveries.loc[date_use, 'flow'] = new_deliveries[this_structure + '_flow'] * 1.0
        #take updated structure deliveries and assign the updates to individual rights
        self.structures_objects[this_structure].update_delivery_rights(date_use)

  def update_structure_plan_flows(self, new_plan_flows, date_use):
    #this function takes a dictionary of reservoir data, with single values from a single simulation timestep and updates struture object timeseries data on upstream reservoir releases  
    station_used_list = []
    #self.plan_flows_list is a dictionary where each reservoir is a key, and those keys contain a list of structures downstream of that reservoir
    for station_start in self.plan_flows_list:
      for station_use in self.plan_flows_list[station_start]:
        if station_use in station_used_list:
          self.structures_objects[station_use].routed_plans.loc[date_use, 'plan'] += new_plan_flows[station_start] * 1.0
        else:
          self.structures_objects[station_use].routed_plans.loc[date_use, 'plan'] = new_plan_flows[station_start] * 1.0
          station_used_list.append(station_use)

  def update_structure_inflows(self, new_releases, date_use):
    for this_structure in self.structures_objects:
      if this_structure + '_inflow' in new_releases:
        self.structures_objects[this_structure].adaptive_monthly_inflows.loc[date_use, 'inflows'] = new_releases[this_structure + '_inflow'] * 1.0
      if this_structure + '_location' in new_releases:
        self.structures_objects[this_structure].adaptive_monthly_control.loc[date_use, 'location'] = str(new_releases[this_structure + '_location'])

  def update_structure_storage(self, new_storage, date_use):
    for this_structure in self.structures_objects:
      if this_structure + '_end_storage' in new_storage:
        self.structures_objects[this_structure].adaptive_reservoir_timeseries.loc[date_use, this_structure + '_end_storage'] = new_storage[this_structure + '_end_storage'] * 1.0
      if this_structure in new_storage:      
        self.structures_objects[this_structure].adaptive_reservoir_timeseries.loc[date_use, this_structure] = new_storage[this_structure] * 1.0


  def find_available_water(self, snow_coefs, tunnel_transfer_to, res_station, snow_station, datetime_val, month_start):
    #this function calculates the value of the water supply index based on storage, snowpack, and year-to-date diversions
    month_index = ['OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP']
    
    #first, calculate year-to-date tunnel diversions from storage
    this_month = datetime_val.month
    if this_month < 10:
      year_start = datetime_val.year - 1
    else:
      year_start = datetime_val.year
    already_diverted = 0.0
    year_add = 0
    for month_num in range(0, 12):
      if month_num + month_start > 12:
        month_start = -2
        year_add = 1
      current_timestep = datetime(year_start + year_add, month_num + month_start, 1, 0, 0)
      if current_timestep == datetime_val:
        break
      already_diverted += self.structures_objects[tunnel_transfer_to].adaptive_monthly_deliveries.loc[current_timestep, 'deliveries']/1000.0
    
    #then, calculate current storage
    available_storage = self.structures_objects[res_station].adaptive_reservoir_timeseries.loc[datetime_val, res_station]/1000.0
    
    #then, calculate snowpack regression for future flow
    if datetime_val.month > 9:
      month_val = month_index[datetime_val.month - 10]
    else:
      month_val = month_index[datetime_val.month + 2]
    control_location = self.structures_objects[res_station].adaptive_monthly_control.loc[datetime_val, 'location']
    if control_location in snow_coefs[month_val]:
      available_snowmelt = (self.basin_snowpack[snow_station].loc[datetime_val, 'basinwide_average'] * snow_coefs[month_val][control_location][0] + snow_coefs[month_val][control_location][1])/1000.0
    else:
      available_snowmelt = (self.basin_snowpack[snow_station].loc[datetime_val, 'basinwide_average'] * snow_coefs[month_val]['all'][0] + snow_coefs[month_val]['all'][1])/1000.0
    total_water = available_storage + available_snowmelt + already_diverted

    return total_water

  def find_adaptive_purchases(self, downstream_data, res_station, date_use):
    #this function identifies water rights that can be informally leased in a given timestep and determines how much demand must be reduced
    
    available_purchases = {}#dictionary of rights to informally lease
    purchase_types = ['right',  'structure', 'demand', 'priority', 'consumptive_fraction']#info about purchased rights
    for rt in purchase_types:
      available_purchases[rt] = []
    #final lists of attributes of water rights to informally lease
    change_points_structure = []
    change_points_right_id = []
    change_points_demand = []
    change_points_buyout_demand = []
    change_points_buyout_purchase = []
    change_points_date = []
    change_points_right = []
    change_points_consumptive = []

    total_physical_supply = 999999.9#initialize value for total supply remaining that can be purchased (as you move 'downstream' in the flow network)
    current_control_location = self.structures_objects[res_station].adaptive_monthly_control.loc[date_use, 'location']
    start_station = str(res_station)#everything upstream of the diversion point can be purchased, once downstream of the diversion point purchases are limited
    reservoir_right = self.structures_objects[res_station].sorted_rights[0]#right associated with diversion location
    
    #get current month key for return flows
    month_name_list = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    month_id = month_name_list[date_use.month - 1]
    #set column index points to read river node network
    station_id_column_length = 12
    station_id_column_start = 0
    downstream_station_id_column_start = 36
    downstream_station_id_column_end = 48
    #find start of river network file
    for j in range(0,len(downstream_data)):
      if downstream_data[j][0] != '#':
        first_line = int(j * 1)
        break
        
    #loop through river node network to find water rights to purchase based on where they are located
    for j in range(first_line, len(downstream_data)):
      #find current structure and location immediately downstream
      station_id = str(downstream_data[j][station_id_column_start:station_id_column_length].strip())
      downstream_id = str(downstream_data[j][downstream_station_id_column_start:downstream_station_id_column_end].strip())

      #only purchase irrigation rights
      #this will make a list of potential rights to purchase (upstream of the structure of interest)
      
      #initially, start station is set to the diversion location, then it is each successive downstream node
      if station_id == start_station:
        #max that can be leased at this location or downstream is equal to the flow minus any releases made for reservoir owners
        total_available_supply = max(self.structures_objects[station_id].adaptive_monthly_inflows.loc[date_use, 'inflows'] - max(self.structures_objects[station_id].routed_plans.loc[date_use, 'plan'], 0.0), 0.0)
        #the 'choke' volume is how much water needs to be leased upstream of this point
        water_exchanges_choke = total_physical_supply - total_available_supply
        if water_exchanges_choke > 0.0:
          #set each attribute list for potential leased rights as a numpy array
          for rt in purchase_types:
            available_purchases[rt] = np.asarray(available_purchases[rt])
          #sort the lists by priority of right, most junior first
          sorted_order = np.argsort(available_purchases['priority'] * -1.0)
          sorted_demand = available_purchases['demand'][sorted_order]
          sorted_rights = available_purchases['right'][sorted_order]
          sorted_structs = available_purchases['structure'][sorted_order]
          sorted_priorities = available_purchases['priority'][sorted_order]
          sorted_returns = available_purchases['consumptive_fraction'][sorted_order]
          
          total_demand = 0.0
          x = 0
          for x in range(0, len(sorted_demand)):
            #find consumptive portion of the water right
            total_demand += sorted_demand[x] * sorted_returns[x]
            #find baseline diversion (total leased volume)
            total_right_delivery = self.structures_objects[sorted_structs[x]].rights_objects[sorted_rights[x]].adaptive_monthly_deliveries.loc[date_use, 'deliveries']
            #only lease rights where deliveries are being made
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
            #once the total consumptive portion of the leased water reaches the 'choke' volume, stop purchasing leases
            if total_demand > water_exchanges_choke:
              break
          #reset potential leased rights
          for rt in purchase_types:
            available_purchases[rt] = []
          #all of the potential leased rights that were not leased at this step can still be leased at later steps
          for y in range(x+1, len(sorted_demand)):
            available_purchases['priority'].append(sorted_priorities[y])
            available_purchases['demand'].append(sorted_demand[y])
            available_purchases['right'].append(sorted_rights[y])
            available_purchases['structure'].append(sorted_structs[y])
            available_purchases['consumptive_fraction'].append(sorted_returns[y])
          #update 'max' choke point value
          total_physical_supply -= float(total_physical_supply - total_available_supply)
        #next check for leased rights happens at the next downstream node
        start_station = str(downstream_id)
        
      #stop searching for new informal leases if we hit the current location of the river supply or the max choke point volume goes to zero
      if total_physical_supply <= 0.0:
        #if you hit the call location then lease water from remaining potential rights
        for rt in purchase_types:
          available_purchases[rt] = np.asarray(available_purchases[rt])
        #order potential leases by seniority
        if total_physical_supply > 0.0:
          sorted_order = np.argsort(available_purchases['priority'] * -1.0)
          sorted_demand = available_purchases['demand'][sorted_order]
          sorted_rights = available_purchases['right'][sorted_order]
          sorted_structs = available_purchases['structure'][sorted_order]
          sorted_priorities = available_purchases['priority'][sorted_order]
          sorted_returns = available_purchases['consumptive_fraction'][sorted_order]
          total_demand = 0.0
          #lease rights until total consumptive fraction equal the choke volume
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
    #create record of what rights were leased, the change in demand associated with the lease, and the consumptive volume that was leased
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

  def find_buyout_partners(self, last_right, last_structure, downstream_data, carrier_connections, divert_location, date_use):
    
    #final list of attributes for lease facilitator
    change_points_structure = []
    change_points_buyout_demand = []
    change_points_buyout_purchase = []
    change_points_date = []

    #set river node network column indices, find first line of network in file
    station_id_column_length = 12
    station_id_column_start = 0
    downstream_station_id_column_start = 36
    downstream_station_id_column_end = 48
    for j in range(0,len(downstream_data)):
      if downstream_data[j][0] != '#':
        first_line = int(j * 1)
        break
    
    #find priority of the informal lease buyer (facilitators are senior to this right)
    sri = self.structures_objects[divert_location].sorted_rights[0]
    end_priority = self.structures_objects[divert_location].rights_objects[sri].priority

    #find the priority of the junior-most lease seller (facilitators are junior to this right)
    use_buyouts = True
    try:
      initial_priority = self.structures_objects[last_structure].rights_objects[last_right].priority
    except:
      use_buyouts = False
      
    if use_buyouts:
      #loop through the 'water rights stack' - all water rights ordered by seniority, senior to junior
      for right_priority, right_id, struct_id in zip(self.rights_stack_priorities, self.rights_stack_ids, self.rights_stack_structure_ids):
        #facilitators are junior to the lease seller, senior to the lease buyer
        if right_priority >= initial_priority and right_priority < end_priority:
          total_decree = 0.0
          #find total water rights at the given structure (that are senior to lease buyer)
          for ind_right in self.structures_objects[struct_id].rights_list:
            if self.structures_objects[struct_id].rights_objects[ind_right].priority < end_priority:
              total_decree += self.structures_objects[struct_id].rights_objects[ind_right].decree_af[date_use.month-1]
          #find total deliveries made to this structure
          total_delivery_level = self.structures_objects[struct_id].historical_monthly_deliveries.loc[date_use, 'deliveries'] * 1.0
          #find unfulfilled demand at this structure (only for rights senior to the lease buyer)
          total_extra_demand = max(min(self.structures_objects[struct_id].adaptive_monthly_demand.loc[date_use, 'demand'], total_decree) - self.structures_objects[struct_id].historical_monthly_deliveries.loc[date_use, 'deliveries'], 0.0)
          #senior, unfulfilled demand at this structure is equal to the facilitated demand.  
          #if there is facilitated demand, need to find if it can make a claim to the informal lease
          if total_extra_demand > 0.0:
            toggle_divert = 0
            toggle_station = 0
            toggle_first = 0
            toggle_station_trib = 0
            use_carrier_divert = True
            #find if the diversion location has any connections to another structure
            try:
              carrier_list_divert = carrier_connections[divert_location]
            except:
              use_carrier_divert = False
            use_carrier_station = True
            #find if the potential facilitator location has any connections to another structure
            try:
              carrier_list_station = carrier_connections[str(struct_id)]
            except:
              use_carrier_station = False
          
            station_max_buyout = 1.0
            #loop through the river network
            for j in range(first_line, len(downstream_data)):
              #find the current structure in the loop and that structure downstream structures
              station_id = str(downstream_data[j][station_id_column_start:station_id_column_length].strip())
              downstream_id = str(downstream_data[j][downstream_station_id_column_start:downstream_station_id_column_end].strip())
              #need to look for both the diversion structure and the potential facilitator struture
              if toggle_first == 0:
                if use_carrier_divert:
                  #look to see if we've found the diversion location or one of its connected structures
                  if station_id == divert_location or station_id in carrier_list_divert:
                    #set toggles to show we've found the diversion structures
                    toggle_first = 1
                    toggle_divert = 1
                    #now only consider structures directly downstream
                    start_station = str(divert_location)
                else:
                  #look to see if we've found the diversion location
                  if station_id == divert_location:
                    #set toggles to show we've found the diversion structures
                    toggle_first = 1
                    toggle_divert = 1
                    #now only consider structures directly downstream
                    start_station = str(divert_location)

                if use_carrier_station:
                  #look to see if we've found the facilitator location or one of its connected structures
                  if station_id == str(struct_id) or station_id in carrier_list_station:
                    #set toggles to show we've found the facilitator structures
                    toggle_first = 1
                    toggle_station = 1
                    start_station = str(divert_location)
                else:
                  #look to see if we've found the facilitator location
                  if station_id == str(struct_id):
                    #set toggles to show we've found the facilitator structures
                    toggle_first = 1
                    toggle_station = 1
                    start_station = str(struct_id)
              #when one of the faciliatator/diverter is found, look for the other of the two that hasn't
              if toggle_first == 1:
                #if one is directly downstream of the other, then the structure serves as a facilitator
                #if this station is a downstream station, look for other of the diverter/faciltator pair
                if station_id == start_station:
                  toggle_station_trib = 0
                  start_station = str(downstream_id)
                  #if the diversion station is downstream, set divert toggle to 1
                  if use_carrier_divert:
                    if station_id == divert_location or station_id in carrier_list_divert:
                      toggle_divert = 1
                  else:
                    if station_id == divert_location:
                      toggle_divert = 1

                  #if the facilitator station is downstream, set divert toggle to 1
                  if use_carrier_station:
                    if station_id == str(struct_id) or station_id in carrier_list_station:
                      toggle_station = 1
                  else:
                    if station_id == str(struct_id):
                      toggle_station = 1
                      
                #if the facilitator structure is not directly downstream of the diverter structure, it can still be a facilitator
                elif station_id == str(struct_id) or toggle_station_trib == 1:
                  toggle_station_trib = 1
                  #check each node until the next node directly downstream of diversion for 'flow past the node'
                  #if there is non-zero flow at every node between the potential facilitator and the next downstream structure, the structure can be a facilitator
                  #note that toggle_statio_trib is set to zero in line 832 when the next downstream station is found
                  if self.structures_objects[struct_id].rights_objects[right_id].adaptive_monthly_demand.loc[date_use, 'demand'] > 0.0:             
                    station_max_buyout = min(station_max_buyout, max(self.structures_objects[str(station_id)].adaptive_monthly_deliveries.loc[date_use, 'flow'] / self.structures_objects[struct_id].rights_objects[right_id].adaptive_monthly_demand.loc[date_use, 'demand'], 0.0))
              if toggle_station == 1 and toggle_divert == 1:
                break
            #record structures that can serve as facilitators
            if station_max_buyout > 0.01:
              change_points_structure.append(struct_id)
              change_points_buyout_demand.append(total_delivery_level)
              change_points_buyout_purchase.append(total_extra_demand)
              change_points_date.append(date_use)              
    #save facilitator information          
    change_points_buyout_df = pd.DataFrame(columns = ['structure', 'demand', 'demand_purchase', 'date'])
    if len(change_points_structure) > 0:
      change_points_buyout_df['structure'] = change_points_structure
      change_points_buyout_df['demand'] = change_points_buyout_demand
      change_points_buyout_df['demand_purchase'] = change_points_buyout_purchase
      change_points_buyout_df['date'] = change_points_date

    return change_points_buyout_df, end_priority
                    
  def set_tunnel_changes(self, new_storage, diversion_current_demand, comp_year, comp_month, storage_id, diversion_id, storage_right, datetime_val):
    #this function calculates the change in storage between the adaptive and baseline scenario and creates a demand change dataframe to increase demands at the export tunnel
    storage_change = self.structures_objects[storage_id].simulated_reservoir_timeseries.loc[datetime_val, storage_id + '_end_storage'] - new_storage[storage_id + '_end_storage']
  
    #set dataframe with the parameters needed to change the StateMod demand input file
    #so that export demand is added to equal the increase in storage between the adaptive & baseline scenarios
    change_points_df = pd.DataFrame()    
    change_points_df['structure'] = [diversion_id,]
    change_points_df['demand'] = [storage_change,]
    change_points_df['right'] = [storage_right,]
    change_points_df['consumptive'] = [1.0,]
    change_points_df['date'] = [datetime(comp_year, comp_month, 1, 0, 0),]
  
    change_points_buyout_df = pd.DataFrame()
    change_points_buyout_df['structure'] = [diversion_id,]
    change_points_buyout_df['demand'] = [diversion_current_demand,]
    change_points_buyout_df['demand_purchase'] = [0.0,]
    change_points_buyout_df['date'] = [datetime(comp_year, comp_month, 1, 0, 0),]
  
    return change_points_df, change_points_buyout_df

  def compare_reservoir_storage(self, baseline_storage, scenario_storage):
    reservoir_list = []
    date_list = []
    difference_list = []
    for res_use in baseline_storage.columns:
      self.structures_objects[res_use].historical_monthly_storage = pd.DataFrame()
      self.structures_objects[res_use].adaptive_monthly_storage = pd.DataFrame()
    for index, row in baseline_storage.iterrows():
      for res_use in baseline_storage.columns:
        self.structures_objects[res_use].historical_monthly_storage.loc[index, 'storage'] = row[res_use]
        self.structures_objects[res_use].adaptive_monthly_storage.loc[index, 'storage'] = scenario_storage.loc[index, res_use]
        
  def find_lease_impacts(self, downstream_data, start_year, end_year, structure_print_list, folder_name):
    #this function calculates the total change in surface water deliveries to right holders in the basin as a result of informal leases (excluding lease buyers and sellers)
    #it also calculates the compensatory releases required to mitigate the shortfalls
    user_type_list = ['Exports', 'Environment', 'Other', 'Compensatory']
    leased_shortfalls = pd.DataFrame(index = user_type_list, columns = ['change',])
    count_shortfalls = pd.DataFrame(index = user_type_list, columns = ['change',])
    structure_purchases = pd.read_csv('results_' + folder_name + '/purchases_5104055.csv')
    purchase_structure_list = structure_purchases['structure'].astype(str).unique().tolist()
    purchase_structure_list.append('5103710')
    purchase_structure_list.append('5103695')
    purchase_structure_list.append('5103709')
    purchase_structure_list.append('5104055')
    purchase_structure_list.append('5003668')
    purchase_structure_list.append('5100958')

    delivery_scenarios = {}
    for x in user_type_list:
      leased_shortfalls.loc[x, 'change'] = 0.0
      count_shortfalls.loc[x, 'change'] = 0.0
    monthly_timesteps = 0
    for structure_name in self.structures_objects:
      monthly_timesteps = max(len(self.structures_objects[structure_name].historical_monthly_deliveries.index), monthly_timesteps)

      #calculate total annual deliveries under baseline scenario
      annual_deliveries_baseline = np.zeros(end_year - start_year + 1)
      total_annual_deliveries = 0.0
      year_counter = 0
      #loop through each month of baseline scenario
      for index, row in self.structures_objects[structure_name].historical_monthly_deliveries.iterrows():
        if index.year >= start_year and index.year <= end_year:
          if index.month == 10:
            annual_deliveries_baseline[year_counter] = total_annual_deliveries * 1.0
            year_counter += 1            
            total_annual_deliveries = 0.0
          total_annual_deliveries += row['deliveries']/1000.0#total deliveries (taf)
      annual_deliveries_baseline[year_counter] = total_annual_deliveries * 1.0
      #calculate total annual deliveries under leasing scenario
      annual_deliveries_adaptive = np.zeros(end_year - start_year + 1)
      total_annual_deliveries = 0.0
      year_counter = 0
      #loop through each month of adaptive scenario
      for index, row in self.structures_objects[structure_name].adaptive_monthly_deliveries.iterrows():
        if index.year >= start_year and index.year <= end_year:
          if index.month == 10:
            annual_deliveries_adaptive[year_counter] = total_annual_deliveries * 1.0
            year_counter += 1            
            total_annual_deliveries = 0.0
          total_annual_deliveries += row['deliveries']/1000.0#total deliveries (taf)
      annual_deliveries_adaptive[year_counter] = total_annual_deliveries * 1.0
      
      storage_change = np.zeros(end_year - start_year + 1)
      try:
        max_storage_change = 0.0
        for index, row in self.structures_objects[structure_name].historical_monthly_storage.iterrows():
          if index.year >= start_year and index.year <= end_year:
            if index.month == 10:
              storage_change[year_counter] = max_storage_change * 1.0
              year_counter += 1            
              max_storage_change = 0.0
            max_storage_change = max(max_storage_change, (row['storage'] - self.structures_objects[structure_name].adaptive_monthly_storage.loc[index, 'storage']) /1000.0)#total deliveries (taf)
        storage_change[year_counter] = max_storage_change * 1.0
      except:
        pass
      
      #find monthly difference in deliveries to the individual structure between baseline & lease scenarios
      delivery_change = annual_deliveries_adaptive - annual_deliveries_baseline
      #look at the change in deliveries at all structures NOT selling their leases - third parties but also lease facilitators
      if self.structures_objects[structure_name].struct_type == 'structure' and structure_name not in purchase_structure_list:
        #aggregate shortfalls by water user type
        #consumptive use - ag
        if 'Irrigation' in self.structures_objects[structure_name].structure_types:
          #find years in which deliveries drop (i.e., this finds the sum of shortfalls in the shortfall years only - it is NOT an 'average change' across the timeseries)
          change_years = delivery_change < 0.0
          if np.sum(delivery_change[change_years]) < 0.0:
            leased_shortfalls.loc['Other', 'change'] += np.sum(delivery_change[change_years])
            count_shortfalls.loc['Other', 'change'] += 1
        #instream flows
        elif 'Minimum Flow' in self.structures_objects[structure_name].structure_types:        
          change_years = delivery_change < 0.0
          if np.sum(delivery_change[change_years]) < 0.0:
            leased_shortfalls.loc['Environment', 'change'] += np.sum(delivery_change[change_years])
            count_shortfalls.loc['Environment', 'change'] += 1
        #change in transbasin diversions (lease buyers)
        elif 'Export' in self.structures_objects[structure_name].structure_types:
          change_years = delivery_change > 0.0
          if np.sum(delivery_change[change_years]) > 0.0:
            leased_shortfalls.loc['Exports', 'change'] += np.sum(delivery_change[change_years])
            count_shortfalls.loc['Exports', 'change'] += 1
        #consumptive use - m&i (including hydropower)
        elif 'Municipal' in self.structures_objects[structure_name].structure_types:
          change_years = delivery_change < 0.0
          if np.sum(delivery_change[change_years]) < 0.0:
            leased_shortfalls.loc['Other', 'change'] += np.sum(delivery_change[change_years])
            count_shortfalls.loc['Other', 'change'] += 1
        elif 'Reservoir' in self.structures_objects[structure_name].structure_types:
          leased_shortfalls.loc['Other', 'change'] -= np.sum(storage_change[change_years])
          count_shortfalls.loc['Other', 'change'] += 1
          
      #save full timeseries from selected structures
      
      if structure_name in structure_print_list:
        delivery_scenarios[structure_name + '_baseline'] = annual_deliveries_baseline
        delivery_scenarios[structure_name + '_reoperation'] = annual_deliveries_adaptive
    
    
    purchase_structure_list.append('5104634')
    month_name_list = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    #loop through downstream users to determine
    #the volume of leases required for compensatory flows
    station_id_column_length = 12
    station_id_column_start = 0
    tot_release = np.zeros(monthly_timesteps)
    running_returns = np.zeros(monthly_timesteps)
    #find first line in the downstream_data
    for j in range(0,len(downstream_data)):
      if downstream_data[j][0] != '#':
        first_line = int(j * 1)
        break
    #calculate the shortages at each station
    individual_shortage = []
    individual_compensatory = []
    individual_stations = []
    individual_dates = []
    reservoir_recal = {}
    for j in range(first_line, len(downstream_data)):
      station_id = str(downstream_data[j][station_id_column_start:station_id_column_length].strip())
      if station_id not in purchase_structure_list:
      #find the compensatory releases required in each month
        counter = 0
        for index, row in self.structures_objects[station_id].historical_monthly_deliveries.iterrows():  
        #find shortage relative to baseline at the structure/timestep
          month_id = month_name_list[index.month - 1]
          if 'Reservoir' in self.structures_objects[station_id].structure_types:
            if station_id in reservoir_recal:
              try:
                reservoir_recal[station_id] = max(min(reservoir_recal[station_id], self.structures_objects[station_id].historical_monthly_storage.loc[index, 'storage'] - self.structures_objects[station_id].adaptive_monthly_storage.loc[index, 'storage']), 0.0)
              except:
                reservoir_recal[station_id] = 0.0
                
            else:
              reservoir_recal[station_id] = 0.0
            try:
              shortage = (self.structures_objects[station_id].historical_monthly_storage.loc[index, 'storage'] - self.structures_objects[station_id].adaptive_monthly_storage.loc[index, 'storage'] - reservoir_recal[station_id]) / 1000.0
              if station_id == '3603543':
                dillon_shortage = (self.structures_objects['3604512'].historical_monthly_storage.loc[index, 'storage'] - self.structures_objects['3604512'].adaptive_monthly_storage.loc[index, 'storage']) / 1000.0
                if dillon_shortage < 0.0:
                  shortage += dillon_shortage
              if station_id == '3604512':
                gm_shortage = (self.structures_objects['3603543'].historical_monthly_storage.loc[index, 'storage'] - self.structures_objects['3603543'].adaptive_monthly_storage.loc[index, 'storage']) / 1000.0
                if gm_shortage < 0.0:
                  shortage += gm_shortage
                  
              reservoir_recal[station_id] = max(self.structures_objects[station_id].historical_monthly_storage.loc[index, 'storage'] - self.structures_objects[station_id].adaptive_monthly_storage.loc[index, 'storage'], 0.0)
            except:
              shortage = 0.0
            self.structures_objects[station_id].return_fraction.loc[month_id, station_id] = 0.0          
          else:
            shortage = (self.structures_objects[station_id].historical_monthly_deliveries.loc[index, 'deliveries'] - self.structures_objects[station_id].adaptive_monthly_deliveries.loc[index, 'deliveries']) / 1000.0
          if shortage > 0.0:
            return_use = min(shortage, running_returns[counter])#use return flows to meet shortage if possible
            running_returns[counter] -= return_use#subtract consumed return flows
            tot_release[counter] += shortage - return_use#releases are shortages above the available return flows
            running_returns[counter] += shortage * self.structures_objects[station_id].return_fraction.loc[month_id, station_id]#add return flows from this consumption to the available return flows
            individual_shortage.append(shortage)
            individual_compensatory.append(shortage - return_use)
            individual_stations.append(station_id)
            individual_dates.append(index)
          counter += 1
    #add compensatory releases to file
    leased_shortfalls.loc['Compensatory', 'change'] = np.sum(tot_release)
    individual_shortfalls = pd.DataFrame()
    individual_shortfalls['shortage'] = individual_shortage
    individual_shortfalls['compensatory'] = individual_compensatory
    individual_shortfalls['stations'] = individual_stations
    individual_shortfalls['dates'] = individual_dates
    individual_shortfalls.to_csv('results_' + folder_name + '/individual_changes.csv')
    
    delivery_scenarios = pd.DataFrame(delivery_scenarios)
    delivery_scenarios.to_csv('results_' + folder_name + '/total_export_deliveries.csv')
    leased_shortfalls.to_csv('results_' + folder_name + '/total_changes.csv')
    count_shortfalls.to_csv('results_' + folder_name + '/total_structures_shorted.csv')

  def find_return_flow_exposure(self, downstream_data, threshold_use):
    #this finds the potential shortfalls from uncertainty about return flows in each structure
    #it assumes that return flows for leased water are higher than the return flow assumptions in the leasing contract
    #higher than expected return flows for leased water reduces flows to the rest of the basin when that water is leased
    #this estimates which structures will be shorted for a given volume of return flow uncertainty in each
    #timestep that leases are exercised
    
    #find lease seller options exercised
    simulated_calls = pd.read_csv('results_' + str(int(threshold_use)) + '/purchases_5104055.csv', index_col = 0)
    simulated_calls['datetime_call'] = pd.to_datetime(simulated_calls['date'])
    
    #what is the uncertainty in return flows (pct of the consumption)
    uncertain_fraction = 0.1
    month_name_list = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    
    #get list of unique dates when lease seller options are called
    new_purchases = simulated_calls.drop_duplicates('datetime_call')
    dates_to_call = new_purchases['datetime_call']
    
    #find beginning of river node network list from data
    station_id_column_length = 12
    station_id_column_start = 0
    downstream_station_id_column_start = 36
    downstream_station_id_column_end = 48
    for j in range(0,len(downstream_data)):
      if downstream_data[j][0] != '#':
        first_line = int(j * 1)
        break
        
    #set structure, timestep, and magnitude of potential return flow risk
    structure_loss_list = []#structure
    structure_loss_volumes = []#magnitude
    structure_list_dates = []#timestep
    #loop through dates of informal purchases
    structure_purchase_loc = '5104055'
    trigger_downstream = False
    for index_use in dates_to_call:
      #find total assumed return flows from leased volume at this timestep
      this_simulated_values = simulated_calls[simulated_calls['datetime_call'] == index_use]
      total_return_flow = 0.0
      #for each water right that is informally leased, calculate the impact of return flow uncertainty
      upstream_index_vals = []
      for index, row in this_simulated_values.iterrows():
        total_return_flow = float(row['demand']) * (float(row['consumptive'])) * uncertain_fraction#potential return flow shortfalls
        next_station = str(row['structure'])#structure of lease purchase - impact only on downstream structures
        upstream_right_list = []#list of rights that can be 'called' by potentially shorted structure
        upstream_delivery_list = []#list of delivery volumes that can be 'called' by potentially shorted structure
        upstream_consumptive_list = []#list of consumptive fraction of the deliveries that can be 'called' by potentially shorted structure
        upstream_structure_list = []#list of structures that can be 'called' by potentially shorted structure
        for j in range(first_line, len(downstream_data)):
          #find current station and the station directly downstream
          station_id = str(downstream_data[j][station_id_column_start:station_id_column_length].strip())
          downstream_id = str(downstream_data[j][downstream_station_id_column_start:downstream_station_id_column_end].strip())
          #if we find the direct downstream station
          if station_id == structure_purchase_loc:
            trigger_downstream = True
          if station_id == next_station:
            #find any extra inflow to this station that is not diverted at this station or part of routed reservoir releases
            extra_flow = max(max(self.structures_objects[station_id].adaptive_monthly_inflows.loc[index_use, 'inflows'] - self.structures_objects[station_id].routed_plans.loc[index_use, 'plan'], 0.0) - self.structures_objects[str(station_id)].adaptive_monthly_deliveries.loc[index_use, 'deliveries'], 0.0)
            #if the extra flow is less than the reduction in flow from return flows, this structure could be at risk of shortfalls due to return flow uncertainty
            if extra_flow < total_return_flow:
              total_called = {}
              max_called = {}
              right_priority = {}
              #how much extra flow needs to be 'called' from upstream, junior users in order to mitigate potential shortfalls
              needed_flow = max(total_return_flow - extra_flow, 0.0)
              #how many deliveries will be made even if return flow uncertainty reduces flow
              protected_deliveries = max(self.structures_objects[station_id].adaptive_monthly_deliveries.loc[index_use, 'deliveries'] - needed_flow, 0.0)
              for sri in self.structures_objects[station_id].sorted_rights:
                total_called[sri] = 0.0#running total of the volume that is called by an individual right at this structure
                #dont need to 'call' water using most senior rights if they can be filled even when 
                this_right_protected_deliveries = min(protected_deliveries, self.structures_objects[station_id].rights_objects[sri].adaptive_monthly_deliveries.loc[index_use, 'deliveries'])
                #this is the maximum volume that can be called by an individual water right
                max_called[sri] = self.structures_objects[station_id].rights_objects[sri].adaptive_monthly_deliveries.loc[index_use, 'deliveries'] * 1.0 - this_right_protected_deliveries
                #what is the priority of the right doing the calling
                right_priority[sri] = float(self.structures_objects[station_id].rights_objects[sri].priority)
                protected_deliveries -= this_right_protected_deliveries
              
              #set lists as numpy arrays so they can be sorted
              upstream_right_np = np.asarray(upstream_right_list)
              upstream_delivery_np = np.asarray(upstream_delivery_list)
              upstream_consumptive_np = np.asarray(upstream_consumptive_list)
              sorted_index = np.argsort(upstream_right_np * (-1.0))
              #loop through the upstream rights from most junior to most senior
              for upstream_index in range(0, len(upstream_right_np)):
                this_right = upstream_right_np[sorted_index[upstream_index]] * 1.0
                this_delivery = upstream_delivery_np[sorted_index[upstream_index]] * 1.0
                this_consumption = upstream_consumptive_np[sorted_index[upstream_index]] * 1.0
                #loop through the water rights at the potentially shorted structures to see if they have
                #seniority to the upstream right potentially being 'called'
                for sri in reversed(self.structures_objects[station_id].sorted_rights):
                  if this_right > right_priority[sri]:
                    if this_delivery > 0.0 and this_consumption > 0.0:
                      #if the upstream right is junior but has deliveries, make a 'call', shorting that right instead
                      called_right = min(min(this_delivery * this_consumption, max_called[sri] - total_called[sri]), max(total_return_flow, 0.0))
                      total_called[sri] += called_right

                      #record structure name, call amount, and date of any potential shortfall due to return flow uncertainty
                      structure_loss_list.append(upstream_structure_list[sorted_index[upstream_index]])
                      structure_loss_volumes.append(called_right)
                      structure_list_dates.append(index_use)
                      #update return flows that can cause shortfalls
                      upstream_index_vals.append(sorted_index[upstream_index])

                      if this_consumption > 0.0:
                        this_delivery -= called_right / this_consumption
                      else:
                        this_delivery -= 0.0
                      total_return_flow -= called_right
                #when any potential return flow shortfalls are exhausted, end loop
                if total_return_flow <= 0.1:
                  break
              upstream_right_list_int = []
              upstream_delivery_list_int = []
              upstream_consumptive_list_int = []
              upstream_structure_list_int = []
              for xxx in range(0, len(upstream_right_list)):
                if xxx not in upstream_index_vals:
                  upstream_right_list_int.append(upstream_right_list[xxx])
                  upstream_delivery_list_int.append(upstream_delivery_list[xxx])
                  upstream_consumptive_list_int.append(upstream_consumptive_list[xxx])
                  upstream_structure_list_int.append(upstream_structure_list[xxx])
              upstream_right_list = []
              upstream_delivery_list = []
              upstream_consumptive_list = []
              upstream_structure_list = []
              for xxx in range(0, len(upstream_right_list_int)):
                upstream_right_list.append(upstream_right_list_int[xxx])
                upstream_delivery_list.append(upstream_delivery_list_int[xxx])
                upstream_consumptive_list.append(upstream_consumptive_list_int[xxx])
                upstream_structure_list.append(upstream_structure_list_int[xxx])
              #if the shortfalls from return flow uncertainty cant be mitigated by making 'calls' on upstream rights
              #the shortfalls come from the current structure
              if total_return_flow > 0.0:
                total_shorted = 0.0
                for sri in self.structures_objects[station_id].sorted_rights:
                  total_shorted += max_called[sri] - total_called[sri]
                consumptive_shorted = min(total_return_flow, total_shorted * (1.0 - self.structures_objects[station_id].return_fraction.loc[month_name_list[index_use.month - 1], station_id]))
                total_return_flow -= consumptive_shorted
                structure_loss_list.append(station_id)
                structure_loss_volumes.append(consumptive_shorted)
                structure_list_dates.append(index_use)
            next_station = str(downstream_id)

          #for each right at the current station - add to list of rights that can be potentially 'called'
          if trigger_downstream:
            for sri in self.structures_objects[station_id].sorted_rights:
              upstream_right_list.append(self.structures_objects[station_id].rights_objects[sri].priority)
              upstream_delivery_list.append(self.structures_objects[station_id].rights_objects[sri].adaptive_monthly_deliveries.loc[index_use, 'deliveries'])
              upstream_consumptive_list.append(1.0 - self.structures_objects[station_id].return_fraction.loc[month_name_list[index_use.month - 1], station_id])
              upstream_structure_list.append(station_id)
          if total_return_flow <= 0.1:
            break

    option_loss = pd.DataFrame()
    option_loss['structure'] = structure_loss_list
    option_loss['delivery'] = structure_loss_volumes
    option_loss['priority_delivery'] = structure_list_dates
    option_loss.to_csv('results_' + str(int(threshold_use)) + '/option_losses.csv')


  def find_option_price_seller(self, thresh_use, alfalfa_residuals, year_start, year_end):
    #this function calculates the annual upfront option fee to compensate for option seller risk
    #wang transform is used to risk-weight probability distributions
    #if you want to know more than that google it yourself
    num_years_shortfalls_obs = 64
    num_years_alfalfa_obs = len(alfalfa_residuals)
    wang_transform_multipliers = np.ones(num_years_shortfalls_obs * num_years_alfalfa_obs)
    #shift probability distributions by 0.25 on the z-score scale
    #first find the z-score for a distribution with 64 x 26 members (simulation observations of potential shortfalls * number of years of alfalfa distribution)
    #assuming that losses from shortfalls and alfalfa price change are independent distributions,
    #this is the wang transform for a joint distribution of alfalfa and delivery shortfalls
    wang_transform_multipliers = np.ones(num_years_shortfalls_obs * num_years_alfalfa_obs)
    for x in range(0, num_years_shortfalls_obs * num_years_alfalfa_obs):
      new_z_value = norm.ppf(float(x+1) / (float(num_years_shortfalls_obs * num_years_alfalfa_obs)))
      adjusted_pbs = norm.cdf(new_z_value + 0.25)
      wang_transform_multipliers[x] = adjusted_pbs / (float(x+1)/ (float(num_years_shortfalls_obs * num_years_alfalfa_obs)))

    total_cost_list = []
    #to calculate seller option prices, assume contract lasts one year and price is equal to alfalfa price at start of contract
    #if alfalfa price is higher when contract is exercised (~one year later) it is a potential loss for lease seller
    #calculate distribution of potential losses through joint distribution of sales volumes multiplied by one year increase in alfalfa prices
    #get lease options exercised
    price_options = pd.read_csv('results_' + thresh_use + '/purchases_5104055.csv')
    #find list of structrues where leases are purchased
    user_list = price_options['structure'].unique().tolist()
    price_options['datetime_use'] = pd.to_datetime(price_options['date'])
    price_options['year_use'] = pd.DatetimeIndex(price_options['datetime_use']).year
    #at each lease structure, price lease option
    loading_values = np.zeros(len(user_list))
    loading_counter = 0
    for x in user_list:
      #find leases exercised at this structure
      this_user_sales = price_options[price_options['structure'] == x]
      annual_loss = np.zeros(num_years_shortfalls_obs * num_years_alfalfa_obs)
      #find distribution of annual leases at this structure
      for year_num in range(year_start, year_end + 1):
        this_year_sales = this_user_sales[this_user_sales['year_use'] == year_num]
        total_yearly_sales = 0.0
        #find the consumptive fraction to convert af of water to tons of alfalfa
        for index, row in this_year_sales.iterrows():
          total_yearly_sales += row['demand'] * row['consumptive']
        
        #for each year of water lease sales, multiply volume leased by the distribution
        #of 1-year alfalfa change if the change is positive (i.e., losses when alfalfa prices drop are set to 0)
        for xxx in range(0, num_years_alfalfa_obs):
          annual_loss[(year_num - year_start) * num_years_alfalfa_obs + xxx] = total_yearly_sales * max(alfalfa_residuals[xxx], 0.0)

      #sort the loss distribution to use in the wang transform
      sorted_losses = np.sort(annual_loss * (-1.0))#sort from largest to smallest
      total_loading_loss = 0.0
      #total option price with loading is the expected value of the wang transform multiplier multiplied by the potential loss pairwise at each step of the distribution
      for year_num in range(0, num_years_shortfalls_obs * num_years_alfalfa_obs):
        total_loading_loss += wang_transform_multipliers[year_num] * sorted_losses[year_num] * (-1.0)#sorted losses are all negative so they can be sorted correctly
      #get expected value based on the joint distribution
      total_cost_list.append(total_loading_loss / (num_years_shortfalls_obs * num_years_alfalfa_obs))
      total_loss = np.sum(this_user_sales['demand'])
      #each structure has an independent 'loading' based on the frequency and magnitude of the potential losses
      if total_loss > 0.0:
        loading_values[loading_counter] = total_loading_loss / total_loss
      else:
        loading_values[loading_counter] = 0.0
      loading_counter += 1

    option_payments = pd.DataFrame(index = user_list)
    option_payments['annual payment'] = total_cost_list
    option_payments['loading'] = loading_values
    option_payments.to_csv('results_' + thresh_use + '/option_payments_sellers.csv')
    
  def find_option_price_facilitator(self, thresh_use, year_start, year_end):
    #this function calculates the annual upfront option fee to compensate for option seller risk
    #wang transform is used to risk-weight probability distributions
    #if you want to know more than that google it yourself
    num_years_shortfalls_obs = 64
    wang_transform_multipliers = np.ones(num_years_shortfalls_obs)
    #shift probability distributions by 0.25 on the z-score scale
    #this is for a distribution with 64 discrete annual observations - facilitator distribution
    for x in range(0, num_years_shortfalls_obs):
      new_z_value = norm.ppf(float(x+1) / float(num_years_shortfalls_obs))
      adjusted_pbs = norm.cdf(new_z_value + 0.25)
      wang_transform_multipliers[x] = adjusted_pbs / (float(x + 1)/float(num_years_shortfalls_obs))

    #load shortages from return flow uncertainty estimation
    all_losses = pd.read_csv('results_' + thresh_use + '/individual_changes.csv')  
    all_losses['datetime_use'] = pd.to_datetime(all_losses['dates'])
    all_losses['year_use'] = pd.DatetimeIndex(all_losses['datetime_use']).year
    user_list = all_losses['stations'].unique().tolist()
    
    #remove any shortages to Lake Granby
    try:
      user_list.remove('5104055')
    except:
      pass
    #contract loading for each facilitator option contract
    loading_values = np.zeros(len(user_list))
    loading_counter = 0
    cost_list = []
    #for each structure with return flow risk, calculate expected value + loading from wang transform
    for struct_id in user_list:
      #total losses
      this_user_losses = all_losses[all_losses['stations'] == struct_id]
      annual_loss = np.zeros(year_end + 1 - year_start)
      #annualize the return flow risk
      for year_num in range(year_start, year_end + 1):
        this_year_losses = this_user_losses[this_user_losses['year_use'] == year_num]
        annual_loss[year_num - year_start] = np.sum(this_year_losses['shortage']) * (-1.0) * 0.1#multiply by -1 to sort from high to low
      #sort losses from biggest to lowest (sorting the negative loss series)
      sorted_losses = np.sort(annual_loss)
      #to find 'loading' on contract, multiply each value in annual loss cdf (sorted from largest to smallest) by the wang transform
      #wang transform gives higher weight to more extreme (largest) losses as a risk-weighting
      total_loading_loss = 0.0
      for year_num in range(0, year_end + 1 - year_start):
        total_loading_loss += wang_transform_multipliers[year_num] * sorted_losses[year_num] * (-1.0)#now we want positive values
      #the 'loading' is just the expected value of the wang-transformed distribution divided by the expected value of the actual distribution
      total_loss = np.sum(this_user_losses['shortage'])
      #each structure has an independent 'loading' based on the frequency and magnitude of the potential losses
      if total_loss > 0.0:
        loading_values[loading_counter] = total_loading_loss / total_loss
      else:
        loading_values[loading_counter] = 0.0
      loading_counter += 1
      #the 'losses' are in units of AF and need to be translated into $ based on the marginal value of the use at the structure 
      if 'Irrigation' in self.structures_objects[struct_id].structure_types:
        cost_list.append(200.0 * total_loading_loss / float(year_end + 1 - year_start))
      elif 'Minimum Flow' in self.structures_objects[struct_id].structure_types:
        cost_list.append(115.0 * total_loading_loss / float(year_end + 1 - year_start))
      elif 'Municipal' in self.structures_objects[struct_id].structure_types:
        cost_list.append(900.0 * total_loading_loss / float(year_end + 1 - year_start))
      else:
        cost_list.append(900.0 * total_loading_loss / float(year_end + 1 - year_start))
      
    #write payments and loading for each structure to file 
    option_payments = pd.DataFrame(index = user_list)
    option_payments['annual payment'] = cost_list
    option_payments['loading'] = loading_values
    option_payments.to_csv('results_' + thresh_use + '/option_payments_facilitators.csv')
    
    
  def find_option_price_facilitator_2(self, input_data_dictionary, thresh_use, year_start, year_end):
    structures_ucrb = gpd.read_file(input_data_dictionary['structures'])
    irrigation_ucrb = gpd.read_file(input_data_dictionary['irrigation'])
    ditches_ucrb = gpd.read_file(input_data_dictionary['ditches'])
    other_structures = list(structures_ucrb['WDID'].astype(str))
    irrigation_structures = list(irrigation_ucrb['SW_WDID1'].astype(str))
    ditch_structures = list(ditches_ucrb['wdid'].astype(str))
    
    #get list of individual structure ids that make up each 'aggregated' structure in StateMod
    #the dictionary keys are the aggregated StateMod structure ids, the lists are the individual structures listed in the structure lists
    agg_diversions = pd.read_csv(input_data_dictionary['aggregated_diversions'])
    aggregated_diversions = {}
    for index, row in agg_diversions.iterrows():
      if row['statemod_diversion'] in aggregated_diversions:
        aggregated_diversions[row['statemod_diversion']].append(str(row['individual_diversion']))
      else:
        aggregated_diversions[row['statemod_diversion']] = [str(row['individual_diversion']), ]

    #this function calculates the annual upfront option fee to compensate for option seller risk
    #wang transform is used to risk-weight probability distributions
    #if you want to know more than that google it yourself
    num_years_shortfalls_obs = 64
    wang_transform_multipliers = np.ones(num_years_shortfalls_obs)
    #shift probability distributions by 0.25 on the z-score scale
    #this is for a distribution with 64 discrete annual observations - facilitator distribution
    for x in range(0, num_years_shortfalls_obs):
      new_z_value = norm.ppf(float(x+1) / float(num_years_shortfalls_obs))
      adjusted_pbs = norm.cdf(new_z_value + 0.25)
      wang_transform_multipliers[x] = adjusted_pbs / (float(x + 1)/float(num_years_shortfalls_obs))

    #load shortages from return flow uncertainty estimation
    all_losses = pd.read_csv('results_' + thresh_use + '/individual_changes.csv')  
    all_losses['datetime_use'] = pd.to_datetime(all_losses['dates'])
    all_losses['year_use'] = pd.DatetimeIndex(all_losses['datetime_use']).year
    user_list = all_losses['stations'].unique().tolist()
    
    #remove any shortages to Lake Granby
    try:
      user_list.remove('5104055')
    except:
      pass
    #contract loading for each facilitator option contract
    loading_values = np.zeros(len(user_list))
    loading_counter = 0
    cost_list = []


    #assign a 'type' to each structure object
    for structure_name in user_list:
      self.structures_objects = {}
      self.structures_objects[structure_name] = Structure(structure_name, 'unknown')
      self.structures_objects[structure_name].structure_types = []      
      #get list of individual structures (equal to the structure name for normal StateMod structure objects, list of many small structures for aggregated StateMod structure object)
      if structure_name in aggregated_diversions:
        ind_structure_list = aggregated_diversions[structure_name]
      else:
        ind_structure_list = [structure_name,]    
      #for each individual structure, assign type based on the 'list' its in, within specific values assigned to those not in particular structure files
      #if structure is irrigation, find total crop acreage
      for ind_structure in ind_structure_list:
        #if structure is in irrigation list, its an irrigation type
        if ind_structure in irrigation_structures:
          self.structures_objects[structure_name].structure_types.append('Irrigation')
        #if structure is in ditch list, its also an irrigation type
        elif ind_structure in ditch_structures:
          self.structures_objects[structure_name].structure_types.append('Irrigation')
        #if structure is in other structure types, assign type based on data about structure
        elif ind_structure in other_structures:
          this_structure = structures_ucrb[structures_ucrb['WDID'] == ind_structure]
          if this_structure.loc[this_structure.index[0], 'StructType'] == 'Reservoir':
            self.structures_objects[structure_name].structure_types.append('Reservoir')
          elif this_structure.loc[this_structure.index[0], 'StructType'] == 'Minimum Flow':
            self.structures_objects[structure_name].structure_types.append('Minimum Flow')
          elif this_structure.loc[this_structure.index[0], 'StructType'] == 'Power Plant':
            self.structures_objects[structure_name].structure_types.append('Municipal')            
          else:
            #assign remainder individually
            if structure_name == '36000662_D' or structure_name == '38000880_D' or structure_name == '5000734_D' or structure_name == '5300555_D' or structure_name == '3900532' or structure_name == '5100941':
              self.structures_objects[structure_name].structure_types.append('Irrigation')
            elif structure_name[:4] == '3600' or structure_name[:4] == '3601' or structure_name[:4] == '3700' or structure_name[:4] == '3800' or structure_name[:4] == '3801' or structure_name[:4] == '5300' or structure_name[:4] == '7200' or structure_name[:4] == '7201' or structure_name == '3900967' or structure_name == '5100958' or structure_name == '5101070':
              self.structures_objects[structure_name].structure_types.append('Municipal')
            elif structure_name[:4] == '3604' or structure_name[:4] == '3704' or structure_name[:4] == '3804' or structure_name[:4] == '5104' or structure_name[:4] == '7204':
              self.structures_objects[structure_name].structure_types.append('Export')
            elif structure_name[:6] == '36_ADC' or  structure_name[:6] == '37_ADC' or structure_name[:6] == '39_ADC' or structure_name[:6] == '45_ADC' or structure_name[:6] == '50_ADC' or structure_name[:6] == '51_ADC' or structure_name[:6] == '52_ADC' or structure_name[:6] == '53_ADC' or structure_name[:6] == '70_ADC' or structure_name[:6] == '72_ADC':
              self.structures_objects[structure_name].structure_types.append('Irrigation')
        #for others, assign types individually
        else:
          if ind_structure[3:6] == 'ARC' or ind_structure[3:6] == 'ASC' or ind_structure[-2:] == 'HU' or ind_structure == '7203904AG' or ind_structure == '7204033AG' or ind_structure == '36GMCON':
            self.structures_objects[structure_name].structure_types.append('Reservoir')
          elif ind_structure[3:6] == 'AMC' or ind_structure[-2:] == 'PL' or  ind_structure[:7] == '5003668' or ind_structure == '70FD1' or ind_structure == '70FD2' or ind_structure == 'CSULimitPLN' or ind_structure == 'HUPLimitPLN' or ind_structure == 'ColRivPln' or ind_structure == '3903508_Ex':
            self.structures_objects[structure_name].structure_types.append('None')
          elif ind_structure[-2:] == '_I':
            self.structures_objects[structure_name].structure_types.append('Irrigation')
          elif ind_structure[-2:] == '_M' or ind_structure == 'MoffatBF' or ind_structure == 'Baseflow' or ind_structure == '3702059_2' or ind_structure == '5300584P' or ind_structure == '3804625M2' or ind_structure == '7202001_2' or ind_structure[:2] == '09' or ind_structure[-3:] == 'Dwn':
            self.structures_objects[structure_name].structure_types.append('Minimum Flow')
          elif ind_structure == '3604683SU' or ind_structure == '3804625SU':
            self.structures_objects[structure_name].structure_types.append('Export')
          elif ind_structure == '36_KeyMun' or ind_structure[:6] == '37VAIL' or ind_structure[:7] == '3803713' or ind_structure == '4200520' or ind_structure == '4200541' or ind_structure == '72_GJMun' or ind_structure == '72_UWCD' or ind_structure == 'ChevDem':
            self.structures_objects[structure_name].structure_types.append('Municipal')
          elif ind_structure[:7] == '7200813':
            if ind_structure == '7200813':
              self.structures_objects[structure_name].structure_types.append('Irrigation')
            else:
              self.structures_objects[structure_name].structure_types.append('Municipal')              

      #total losses
      this_user_losses = all_losses[all_losses['stations'] == structure_name]
      annual_loss = np.zeros(year_end + 1 - year_start)
      #annualize the return flow risk
      for year_num in range(year_start, year_end + 1):
        this_year_losses = this_user_losses[this_user_losses['year_use'] == year_num]
        annual_loss[year_num - year_start] = np.sum(this_year_losses['shortage']) * (-1.0) * 0.1#multiply by -1 to sort from high to low
      #sort losses from biggest to lowest (sorting the negative loss series)
      sorted_losses = np.sort(annual_loss)
      #to find 'loading' on contract, multiply each value in annual loss cdf (sorted from largest to smallest) by the wang transform
      #wang transform gives higher weight to more extreme (largest) losses as a risk-weighting
      total_loading_loss = 0.0
      for year_num in range(0, year_end + 1 - year_start):
        total_loading_loss += wang_transform_multipliers[year_num] * sorted_losses[year_num] * (-1.0)#now we want positive values
      #the 'loading' is just the expected value of the wang-transformed distribution divided by the expected value of the actual distribution
      total_loss = np.sum(this_user_losses['shortage'])
      #each structure has an independent 'loading' based on the frequency and magnitude of the potential losses
      if total_loss > 0.0:
        loading_values[loading_counter] = total_loading_loss / total_loss
      else:
        loading_values[loading_counter] = 0.0
      loading_counter += 1
      #the 'losses' are in units of AF and need to be translated into $ based on the marginal value of the use at the structure 
      if 'Irrigation' in self.structures_objects[structure_name].structure_types:
        cost_list.append(200.0 * total_loading_loss / float(year_end + 1 - year_start))
      elif 'Minimum Flow' in self.structures_objects[structure_name].structure_types:
        cost_list.append(115.0 * total_loading_loss / float(year_end + 1 - year_start))
      elif 'Municipal' in self.structures_objects[structure_name].structure_types:
        cost_list.append(900.0 * total_loading_loss / float(year_end + 1 - year_start))
      else:
        cost_list.append(900.0 * total_loading_loss / float(year_end + 1 - year_start))
      
    #write payments and loading for each structure to file 
    option_payments = pd.DataFrame(index = user_list)
    option_payments['annual payment'] = cost_list
    option_payments['loading'] = loading_values
    option_payments.to_csv('results_' + thresh_use + '/option_payments_facilitators_updated.csv')

