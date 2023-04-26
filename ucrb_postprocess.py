import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import crss_reader as crss
from basin import Basin

# This script reads the raw StateMod output and creates a series of .csv files for figure creation
#################################################################
#### set postprocessing parameters
#################################################################
year_start_adaptive = 1950#year start analysis
year_end = 2013#year end analysis
basin_use = '14010001'#basin to use for snowpack data
reservoir_use = '5104055'#informal lease location
tunnel_transfer_to = '5104634'#export demand from informal leases
res_list = ['3603543', '3603570', '3603575', '3604512', '36_ARC001', '3703639', '3703699', '3704516', '37_ARC002', '3803713', '38_ARC003', '3903505', '3903508', '3903943', '39_ARC004', '45_ARC005', '5003668', '50_ARC006', '5103686', '5103695', '5103709', '5103710', '5104055', '51_ARC007', '52_ARC008', '53_ARC009']

#################################################################
#### load baseline StateMod data
#################################################################
ucrb_baseline = Basin()
input_data_dictionary = crss.create_input_data_dictionary('B', 'A')
#raw StateMod data
reservoir_rights_data = crss.read_text_file(input_data_dictionary['reservoir_rights'])#basin water rights (reservoirs)
structure_rights_data = crss.read_text_file(input_data_dictionary['structure_rights'])#basin water rights (other consumptive)
instream_rights_data = crss.read_text_file(input_data_dictionary['instream_rights'])#basin water rights (instream flows)
demand_data = crss.read_text_file(input_data_dictionary['structure_demand'])#baseline demands
delivery_data = crss.read_text_file(input_data_dictionary['deliveries'])#simulated (baseline) deliveries to diversion structures
reservoir_storage_data = crss.read_text_file(input_data_dictionary['reservoir_storage'])#simulated (baseline) reservoir storage, releases
return_flow_data = crss.read_text_file(input_data_dictionary['return_flows'])#simulated (baseline) return flows from each structure
operational_rights_data = crss.read_text_file(input_data_dictionary['operations'])#special links between water rights and diversion nodes
downstream_data = crss.read_text_file(input_data_dictionary['downstream'])#upstream/downstream structure of model nodes

#################################################################
#### link rights with structures
#################################################################
reservoir_rights_name, reservoir_rights_structure_name, reservoir_rights_priority, reservoir_rights_decree, reservoir_fill_rights, reservoir_rights_structure_common_title = crss.read_rights_data(reservoir_rights_data, structure_type = 'reservoir')
structure_rights_name, structure_rights_structure_name, structure_rights_priority, structure_rights_decree, structure_rights_structure_common_title = crss.read_rights_data(structure_rights_data)
instream_rights_name, instream_rights_structure_name, instream_rights_priority, instream_rights_decree, instream_rights_structure_common_title = crss.read_rights_data(instream_rights_data)
#adjust links between water rights 
new_structure_rights_structure_name, carrier_connections  = crss.read_operational_rights(operational_rights_data, structure_rights_structure_name, structure_rights_name)

#################################################################
#### get timeseries of demand and deliveries for each structure
#################################################################
#Create Basin class with the same extent as StateMod Basin
#as defined by the input_data_dictionary files
structure_demands = crss.read_structure_demands(demand_data,year_start_adaptive, year_end, read_from_file = False)
structure_deliveries = crss.read_structure_deliveries(delivery_data, year_start_adaptive, year_end, read_from_file = False)
# also get reservoir storage
baseline_reservoir_timeseries = crss.read_simulated_reservoirs_list(reservoir_storage_data, res_list, year_start_adaptive, year_end)
baseline_reservoir_single_timeseries = crss.read_simulated_reservoirs(reservoir_storage_data, reservoir_use, year_start_adaptive, year_end)
#get flow at each structure
structure_inflows = crss.read_structure_inflows(delivery_data, year_start_adaptive, year_end)

#################################################################
#### calculate snowpack regression and historical cbi values
#################################################################
historical_monthly_available = pd.DataFrame(list(structure_inflows[reservoir_use + '_available']), index = structure_inflows.index, columns = ['available',])
historical_monthly_control = pd.DataFrame(list(structure_inflows[reservoir_use + '_location']), index = structure_inflows.index, columns = ['location',])
ucrb_baseline.load_basin_snowpack(input_data_dictionary)
snow_coefs_tot = ucrb_baseline.make_snow_regressions(basin_use, historical_monthly_control, baseline_reservoir_single_timeseries, historical_monthly_available, reservoir_use, year_start_adaptive, year_end)
crss.get_cbi_timeseries(structure_deliveries, baseline_reservoir_single_timeseries, structure_inflows, ucrb_baseline.basin_snowpack[basin_use], snow_coefs_tot, tunnel_transfer_to, reservoir_use)

#################################################################
#### load informal leasing scenarios
#################################################################
ucrb = Basin()
ucrb2 = Basin()
ucrb3 = Basin()
ucrb4 = Basin()
basin_list = [ucrb, ucrb2, ucrb3, ucrb4]
scenario_list = ['550', '600', '650', '700']
#load output filenames from different informal leasing scenarios
for basin_use, scenario_name in zip(basin_list, scenario_list):
  input_data_dictionary_new = crss.create_input_data_dictionary('B', 'A', folder_name = 'results_' + scenario_name +'/')

  reservoir_storage_data_new = crss.read_text_file(input_data_dictionary_new['reservoir_storage_new'])#adaptive reservoir storage
  demand_data_new = crss.read_text_file(input_data_dictionary_new['structure_demand_new'])#adaptive demands
  delivery_data_new = crss.read_text_file(input_data_dictionary_new['deliveries_new'])#adaptive deliveries
    
  #create structure/reservoir objects within the basin object, create right objects within the structures
  basin_use.set_rights_to_reservoirs(reservoir_rights_name, reservoir_rights_structure_name, reservoir_rights_priority, reservoir_rights_decree, reservoir_fill_rights)
  basin_use.set_rights_to_structures(structure_rights_name, new_structure_rights_structure_name, structure_rights_priority, structure_rights_decree)
  basin_use.set_rights_to_structures(instream_rights_name, instream_rights_structure_name, instream_rights_priority, instream_rights_decree)
  #create 'rights stack' - all rights listed in the order of their priority, w/ structure names, decree amounts, etc. (single list for reservoir rights, diversion structure rights, and instream rights
  basin_use.combine_rights_data(structure_rights_name, new_structure_rights_structure_name, structure_rights_priority, structure_rights_decree, reservoir_rights_name, reservoir_rights_structure_name, reservoir_rights_priority, reservoir_rights_decree, instream_rights_name, instream_rights_structure_name, instream_rights_priority, instream_rights_decree)

  ###############################################################################################################################
  ####add water demands, deliveries, return flows, outflows, structure types, and reservoir release data to structure objects####
  print(scenario_name + ' apply demands to structures')
  #read demand data timeseries to dataframe
  structure_demands_adaptive = crss.read_structure_demands(demand_data_new,year_start_adaptive, year_end, read_from_file = False)
  #apply demands to structure objects, distribute across structure water rights (by seniority)
  basin_use.set_structure_demands(structure_demands, structure_demands_adaptive = structure_demands_adaptive)

  print(scenario_name + ' apply deliveries to structures')
  #read delivery data timeseries to dataframe
  structure_deliveries_adaptive = crss.read_structure_deliveries(delivery_data_new, year_start_adaptive, year_end, read_from_file = False)
  #apply deliveries to structure objects, distribute across structure water rights (by seniority)
  basin_use.set_structure_deliveries(structure_deliveries, structure_deliveries_adaptive = structure_deliveries_adaptive)

  #overwrite 'max' demands (=9999999.9) with maximum monthly deliveries in baseline simulation
  monthly_maximums = basin_use.adjust_structure_deliveries()

  print(scenario_name + ' apply return flows to structures')
  #read average monthly return flow data to dataframe
  structure_return_flows = crss.read_structure_return_flows(return_flow_data, year_start_adaptive, year_end, read_from_file = False)
  #apply return flow fractions to structure objects
  basin_use.set_return_fractions(structure_return_flows)
  #apply structure outflow timeseries to structure objects
  structure_inflows_new = crss.read_structure_inflows(delivery_data_new, year_start_adaptive, year_end)
  basin_use.set_structure_inflows(structure_inflows, structure_inflows_adaptive = structure_inflows_new)

  print(scenario_name + ' set structure types')
  #classify structures by water use type, add to structure object
  basin_use.set_structure_types(input_data_dictionary_new)

  #read stored releases from reservoirs
  plan_flows = crss.read_plan_flows(reservoir_storage_data, [reservoir_use,], year_start_adaptive, year_end)
  #apply timeseries of releases from upstream reservoirs to structures
  basin_use.set_plan_flows(plan_flows, downstream_data, [reservoir_use,],  '7202003')

  ###################################################################################################
  ####create water supply indicies with relationships between snowpack and inflow into reservoirs####
  print('create historical reservoir timeseries')
  simulated_reservoir_timeseries = crss.read_simulated_reservoirs_list(reservoir_storage_data_new, res_list, year_start_adaptive, year_end)
  basin_use.compare_reservoir_storage(baseline_reservoir_timeseries, simulated_reservoir_timeseries)

  print('calculate initial water supply metrics')
  basin_use.find_lease_impacts(downstream_data, year_start_adaptive, year_end, ['5104634',], scenario_name)
  basin_use.find_option_price_facilitator(scenario_name, year_start_adaptive, year_end)




