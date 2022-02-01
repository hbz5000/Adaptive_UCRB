##postprocessing
reservoir_data_new = crss.read_text_file(input_data_dictionary['reservoir_storage_new'])
for res in ucrb.reservoir_list:
  reservoir_figure = Plotter('results/informal_transfers_to' + ucrb.structures_objects[res].name + '.png', figsize = (16,6))
  informal_transfer, structure_purchase_list, structure_buyout_list = reservoir_figure.plot_informal_purchases(res)

#plot historical reseroir data
for res in ['5104055',]:
  ucrb.structures_objects[res].simulated_reservoir_timeseries_new = crss.read_simulated_reservoirs(reservoir_data_new, res, year_start, year_end)
  reservoir_figure = Plotter('results/reservoir_validation' + ucrb.structures_objects[res].name + '.png', figsize = (8,8))
  reservoir_figure.plot_reservoir_figures(ucrb.structures_objects[res].historical_reservoir_timeseries, ucrb.structures_objects[res].simulated_reservoir_timeseries, ucrb.structures_objects[res].simulated_reservoir_timeseries_new, ucrb.structures_objects[res].name, informal_transfer)
  del reservoir_figure
  reservoir_figure = Plotter('results/transfer_cost' + ucrb.structures_objects[res].name + '.png', figsize = (8,8))
  reservoir_figure.plot_cost_distribution(res, ucrb.structures_objects[res].simulated_reservoir_timeseries, ucrb.structures_objects[res].simulated_reservoir_timeseries_new, informal_transfer)
  del reservoir_figure
  
  
  
#Plot reservoir rights priority onto the cumulative demands of the rights stack
rights_stack_figure = Plotter('results/rights_stack_prob.png', figsize = (16,10))
rights_stack_figure.plot_rights_stack_prob(ucrb.rights_stack_ids, ucrb.rights_stack_structure_ids, ucrb.structures_objects, ucrb.reservoir_list)

for res in ucrb.reservoir_list:
  reservoir_figure = Plotter('results/flow_past_station_' + res + '.png', nr = 3)
  reservoir_figure.plot_release_simulation(simulated_release_timeseries, ucrb.basin_snowpack['14010001'], res, 1950, 2013)
  del reservoir_figure
  reservoir_figure = Plotter('results/controlled_flow_past_station_' + res + '.png', nr = 3, nc = 2, figsize = (12, 18))
  snow_coefs_location = reservoir_figure.plot_release_simulation_controlled(simulated_release_timeseries, ucrb.basin_snowpack['14010001'], res, snow_coefs_tot, 1950, 2013)
  del reservoir_figure
  reservoir_figure = Plotter('results/available_water_' + res + '.png', nr = 2, figsize = (16,9))
  total_water[res], date_index = reservoir_figure.plot_available_water(ucrb.structures_objects[res].simulated_reservoir_timeseries, ucrb.basin_snowpack['14010001'], simulated_release_timeseries, snow_coefs_tot, snow_coefs_location, res, 1950, 2013)
  del reservoir_figure
  reservoir_figure = Plotter('results/physical_supply_availablility_' + res + '.png', figsize = (20,6))
  reservoir_figure.plot_physical_supply(simulated_release_timeseries, informal_transfer_network, res, ucrb.structures_objects, 2000, 2010)
  del reservoir_figure


fnf_station_list = ['09163500',]
full_natural_flow_data = crss.read_text_file(input_data_dictionary['natural flows'])
full_natural_flows = crss.read_full_natural_flow(full_natural_flow_data, fnf_station_list, year_start, year_end)
full_natural_figure = Plotter(project_folder + 'Adaptive_experiment/results/snow_flow.png', figsize = (16,6))
full_natural_figure.plot_snow_fnf_relationship(ucrb.basin_snowpack['14010001'], full_natural_flows, '09163500')


##Historical Data for Adams Tunnel
reservoir_list = ['5104055', '5103710', '5103695']#GRANBY, WILLOW CREEK, SHADOW MNT
reservoir_names = ['GRANBY', 'WILLOW CREEK', 'SHADOW MNT']
adams_tunnel = '5104634'
lake_granby = '5104055'
fnf_station_list = ['09163500',]


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
snow_coefs = reservoir_figure.plot_release_simulation(simulated_release_timeseries, ucrb.basin_snowpack['14010001'], '5104055', '5104634', 1950, 2013)
del reservoir_figure

#Make Historical Plots - Snowpack indices vs. available water, grouped by month & current controlling call

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


##graveyard