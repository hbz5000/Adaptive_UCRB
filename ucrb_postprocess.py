import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import crss_reader as crss
from basin import Basin
from plotter import Plotter

#load output filenames from different informal leasing scenarios
input_data_dictionary = crss.create_input_data_dictionary('B', 'A', folder_name = 'results_550/')
input_data_dictionary2 = crss.create_input_data_dictionary('B', 'A', folder_name = 'results_600/')
input_data_dictionary3 = crss.create_input_data_dictionary('B', 'A', folder_name = 'results_650/')
input_data_dictionary4 = crss.create_input_data_dictionary('B', 'A', folder_name = 'results_700/')
scenario_input_dictionaries = [input_data_dictionary, input_data_dictionary2, input_data_dictionary3, input_data_dictionary4]
scenario_names = ['550', '600', '650', '700']

#################################################################
######################FIGURE 2###################################
#################################################################
crop_types = Plotter('figure_2.png',figsize = (20,12))
crop_types.plot_crop_types(input_data_dictionary['irrigation'])
del crop_types

#################################################################
######################FIGURE 3###################################
#################################################################
reservoir_use = '5104055'#informal lease location
tunnel_transfer_to = '5104634'#export demand from informal leases
###################################
####read baseline inputs/output####
##load unformatted input file data
print('load input data')
downstream_data = crss.read_text_file(input_data_dictionary['downstream'])#upstream/downstream structure of model nodes
reservoir_rights_data = crss.read_text_file(input_data_dictionary['reservoir_rights'])#basin water rights (reservoirs)
structure_rights_data = crss.read_text_file(input_data_dictionary['structure_rights'])#basin water rights (other consumptive)
instream_rights_data = crss.read_text_file(input_data_dictionary['instream_rights'])#basin water rights (instream flows)
demand_data = crss.read_text_file(input_data_dictionary['structure_demand'])#baseline demands
demand_data_new = {}
for input_dictionary, sc_nm in zip(scenario_input_dictionaries, scenario_names):
  demand_data_new[sc_nm] = crss.read_text_file(input_dictionary['structure_demand_new'])#adaptive demands
operational_rights_data = crss.read_text_file('cm2015B.opr')#special links between water rights and diversion nodes
#load baseline simulation outputs
print('load output data')
reservoir_storage_data = crss.read_text_file(input_data_dictionary['reservoir_storage'])#simulated (baseline) reservoir storage, releases
reservoir_storage_data_new = {}
for input_dictionary, sc_nm in zip(scenario_input_dictionaries, scenario_names):
  reservoir_storage_data_new['550'] = crss.read_text_file(input_dictionary['reservoir_storage_new'])#adaptive reservoir storage

delivery_data = crss.read_text_file(input_data_dictionary['deliveries'])#simulated (baseline) deliveries to diversion structures
delivery_data_new = {}
for input_dictionary, sc_nm in zip(scenario_input_dictionaries, scenario_names):
  delivery_data_new[sc_nm] = crss.read_text_file(input_dictionary['deliveries_new'])#adaptive deliveries

return_flow_data = crss.read_text_file(input_data_dictionary['return_flows'])#simulated (baseline) return flows from each structure
return_flow_data_new = {}
for input_dictionary, sc_nm in zip(scenario_input_dictionaries, scenario_names):
  return_flow_data_new[sc_nm] = crss.read_text_file(input_dictionary['return_flows_new'])#adaptive return flow

#############################################
####format water rights data into objects####
print('apply rights to structures')
#link structure ids with water right ids, decrees, priorities
reservoir_rights_name, reservoir_rights_structure_name, reservoir_rights_priority, reservoir_rights_decree, reservoir_fill_rights, reservoir_rights_structure_common_title = crss.read_rights_data(reservoir_rights_data, structure_type = 'reservoir')
structure_rights_name, structure_rights_structure_name, structure_rights_priority, structure_rights_decree, structure_rights_structure_common_title = crss.read_rights_data(structure_rights_data)
instream_rights_name, instream_rights_structure_name, instream_rights_priority, instream_rights_decree, instream_rights_structure_common_title = crss.read_rights_data(instream_rights_data)
#adjust links between water rights 
new_structure_rights_structure_name, carrier_connections  = crss.read_operational_rights(operational_rights_data, structure_rights_structure_name, structure_rights_name)
#Create Basin class with the same extent as StateMod Basin
#as defined by the input_data_dictionary files
ucrb = Basin()
ucrb2 = Basin()
ucrb3 = Basin()
ucrb4 = Basin()
year_start = 1908
year_start_adaptive = 1950
year_end = 2013
basin_list = [ucrb, ucrb2, ucrb3, ucrb4]
scenario_list = ['550', '600', '650', '700']
structure_demands_adaptive = {}
structure_deliveries_adaptive = {}
structure_inflows_new = {}
structure_demands = crss.read_structure_demands(demand_data,year_start, year_end, read_from_file = False)
structure_deliveries = crss.read_structure_deliveries(delivery_data, year_start, year_end, read_from_file = False)
structure_inflows = crss.read_structure_inflows(delivery_data, year_start, year_end)
for basin_use, scenario_name in zip(basin_list, scenario_list):
  basin_use.load_basin_snowpack(input_data_dictionary)
  basin_use.reservoir_list = [reservoir_use,]

  #create structure/reservoir objects within the basin object, create right objects within the structures
  basin_use.set_rights_to_reservoirs(reservoir_rights_name, reservoir_rights_structure_name, reservoir_rights_priority, reservoir_rights_decree, reservoir_fill_rights)
  basin_use.set_rights_to_structures(structure_rights_name, new_structure_rights_structure_name, structure_rights_priority, structure_rights_decree)
  basin_use.set_rights_to_structures(instream_rights_name, instream_rights_structure_name, instream_rights_priority, instream_rights_decree)

  #create 'rights stack' - all rights listed in the order of their priority, w/ structure names, decree amounts, etc. (single list for reservoir rights, diversion structure rights, and instream rights
  basin_use.combine_rights_data(structure_rights_name, new_structure_rights_structure_name, structure_rights_priority, structure_rights_decree, reservoir_rights_name, reservoir_rights_structure_name, reservoir_rights_priority, reservoir_rights_decree, instream_rights_name, instream_rights_structure_name, instream_rights_priority, instream_rights_decree)

  ###############################################################################################################################
  ####add water demands, deliveries, return flows, outflows, structure types, and reservoir release data to structure objects####
  print('apply demands to structures')
  #read demand data timeseries to dataframe
  structure_demands_adaptive[scenario_name] = crss.read_structure_demands(demand_data_new[scenario_name],year_start, year_end, read_from_file = False)
  #apply demands to structure objects, distribute across structure water rights (by seniority)
  basin_use.set_structure_demands(structure_demands, structure_demands_adaptive = structure_demands_adaptive[scenario_name])

  print('apply deliveries to structures')
  #read delivery data timeseries to dataframe
  structure_deliveries_adaptive[scenario_name] = crss.read_structure_deliveries(delivery_data_new[scenario_name], year_start, year_end, read_from_file = False)
  #apply deliveries to structure objects, distribute across structure water rights (by seniority)
  basin_use.set_structure_deliveries(structure_deliveries, structure_deliveries_adaptive = structure_deliveries_adaptive[scenario_name])

  #overwrite 'max' demands (=9999999.9) with maximum monthly deliveries in baseline simulation
  monthly_maximums = basin_use.adjust_structure_deliveries()

  print('apply return flows to structures')
  #read average monthly return flow data to dataframe
  structure_return_flows = crss.read_structure_return_flows(return_flow_data, year_start, year_end, read_from_file = False)
  #apply return flow fractions to structure objects
  basin_use.set_return_fractions(structure_return_flows)

  #apply structure outflow timeseries to structure objects
  structure_inflows_new[scenario_name] = crss.read_structure_inflows(delivery_data_new[scenario_name], year_start, year_end)
  basin_use.set_structure_inflows(structure_inflows, structure_inflows_adaptive = structure_inflows_new[scenario_name])

  print('set structure types')
  #classify structures by water use type, add to structure object
  basin_use.set_structure_types(input_data_dictionary)

  #read stored releases from reservoirs
  plan_flows = crss.read_plan_flows(reservoir_storage_data, basin_use.reservoir_list, year_start, year_end)
  #apply timeseries of releases from upstream reservoirs to structures
  basin_use.set_plan_flows(plan_flows, downstream_data, basin_use.reservoir_list,  '7202003')

  ###################################################################################################
  ####create water supply indicies with relationships between snowpack and inflow into reservoirs####
  print('create historical reservoir timeseries')
  #create timeseries of total inflow impounded by reservoirs (for use in snowpack/flow regressions)
  for res in basin_use.reservoir_list:
    basin_use.structures_objects[res].simulated_reservoir_timeseries = crss.read_simulated_reservoirs(reservoir_storage_data, res, year_start, year_end)
  #make a copy of reservoir storage values that can be edited by adaptive simulations & compared to the original
    basin_use.structures_objects[res].adaptive_reservoir_timeseries = basin_use.structures_objects[res].simulated_reservoir_timeseries.copy(deep = True)

  print('calculate initial water supply metrics')
  #create linear regression coefficients for the relationship between observed snowpack in each month and remaining impounded inflow to reservoirs
  snow_coefs_tot = {}
  for res in basin_use.reservoir_list:
  #univariate linear regression coefficients between the observed snowpack and future inflow to a reservoir, conditional on month of the year and any existing 'calls' on the river
    snow_coefs_tot[res] = basin_use.make_snow_regressions('14010001', res, year_start_adaptive, year_end)

available_water = Plotter('figure_3.png', figsize = (16,6))
available_water.plot_available_water(structure_deliveries, ucrb.structures_objects[reservoir_use].simulated_reservoir_timeseries, structure_inflows, ucrb.basin_snowpack['14010001'], snow_coefs_tot[reservoir_use], tunnel_transfer_to, reservoir_use)
del available_water
#################################################################
######################FIGURE 4##################################
#################################################################
informal_lease_frequency = Plotter('figure_4.png', figsize = (12, 6))
informal_lease_frequency.plot_informal_leases(['700', '650', '600', '550'])
del informal_lease_frequency
#################################################################
######################FIGURE 5##################################
#################################################################
third_party_impacts = Plotter('figure_5.png', figsize = (10, 10))
x_cnt = 0
for basin_use, folder_name in zip(basin_list, scenario_list):
  basin_use.find_lease_impacts(downstream_data, year_start_adaptive, year_end, ['5104634',], x_cnt, folder_name)
  x_cnt += 1
third_party_impacts.plot_third_party_impacts(scenario_list)
del third_party_impacts
#################################################################
######################FIGURE 6##################################
#################################################################
transaction_costs = Plotter('figure_6.png', figsize = (16,12))
transaction_costs.plot_informal_price('700')
del transaction_costs

#################################################################
######################FIGURE 7##################################
#################################################################
option_fees = Plotter('figure_7.png', figsize = (10, 10))
alfalfa_price_residuals = crss.get_alfalfa_residuals()
total_option_seller = {}
total_option_facilitator = {}
for basin_use, scenario_name in zip(basin_list, scenario_list):
  basin_use.find_return_flow_exposure(downstream_data, scenario_name)
  basin_use.find_option_price_seller(scenario_name, alfalfa_price_residuals, year_start_adaptive, year_end)
  basin_use.find_option_price_facilitator(scenario_name, year_start_adaptive, year_end)
option_fees.plot_annual_option_payments(scenario_list)
del option_fees