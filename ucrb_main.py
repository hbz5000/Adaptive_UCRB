import os
import pandas as pd
from datetime import datetime
from basin import Basin
import crss_reader as crss
import shutil

###############################################
####initialize baseline statemod simulation####
#create dictionary of input filenames
input_data_dictionary = crss.create_input_data_dictionary('B', 'A')

for thresh_use in [550,]:
    
  #create new basin (this will hold all the structure/reservoir objects for statemod simluation input/output)
  ucrb = Basin()
  ucrb.load_basin_snowpack(input_data_dictionary)
  ucrb.reservoir_list = []
  ucrb.create_reservoir('Green Mountain', '3603543', 154645.0)
  ucrb.create_reservoir('Dillon', '3604512', 257000.0)
  ucrb.create_reservoir('Homestake', '3704516', 43600.0)
  ucrb.create_reservoir('Wolford', '5003668', 65985.0)
  ucrb.create_reservoir('Williams Fork', '5103709', 96822.0)
  ucrb.create_reservoir('Granby', '5104055', 539758.0)

  #set simulation period
  year_start = 1908
  year_start_adaptive = 1950
  year_end = 2013

  #set informal targets (diversion point & demand point)
  #this is the location of the right that is used to divert water (when this right is out-of-priority, informal transfers make it in-priority)
  reservoir_use = '5104055'
  res_thres = {}
  res_thres[reservoir_use] = float(thresh_use)
  results_dir = 'results_' + str(int(res_thres[reservoir_use]))
  if not os.path.isdir(results_dir):
    os.makedirs(results_dir)
  #this is the location of the demand that is filled by the diversion above
  tunnel_transfer_to = '5104634'

  #create simulation file that will run baseline conditions during the 'simulation period'
  #start simulations with a 10-year 'spin-up' period
  crss.make_control_file('cm2015', 'B', year_start_adaptive - 10, 2013)
  #this runs the 'cm2015' baseline scenario with statemod executable - using control file above
  #executable files, statemod input files and output files are in the working directory 
  os.system("StateMod_Model_15.exe cm2015B -simulate")    

  ###################################
  ####read baseline inputs/output####
  ##load unformatted input file data
  print('load input data')
  downstream_data = crss.read_text_file(input_data_dictionary['downstream'])#upstream/downstream structure of model nodes
  reservoir_rights_data = crss.read_text_file(input_data_dictionary['reservoir_rights'])#basin water rights (reservoirs)
  structure_rights_data = crss.read_text_file(input_data_dictionary['structure_rights'])#basin water rights (other consumptive)
  instream_rights_data = crss.read_text_file(input_data_dictionary['instream_rights'])#basin water rights (instream flows)
  demand_data = crss.read_text_file(input_data_dictionary['structure_demand'])#baseline demands
  operational_rights_data = crss.read_text_file('cm2015B.opr')#special links between water rights and diversion nodes
  #load baseline simulation outputs
  print('load output data')
  historical_reservoir_data = crss.read_text_file(input_data_dictionary['historical_reservoirs'])#historical reservoir storage, releases
  reservoir_storage_data_b = crss.read_text_file(input_data_dictionary['reservoir_storage'])#simulated (baseline) reservoir storage, releases
  delivery_data = crss.read_text_file(input_data_dictionary['deliveries'])#simulated (baseline) deliveries to diversion structures
  return_flow_data = crss.read_text_file(input_data_dictionary['return_flows'])#simulated (baseline) return flows from each structure
  plan_data = crss.read_text_file(input_data_dictionary['plan_releases'])#releases for deliveries outside prior appropriation rules 

  #############################################
  ####format water rights data into objects####
  print('apply rights to structures')
  #link structure ids with water right ids, decrees, priorities
  reservoir_rights_name, reservoir_rights_structure_name, reservoir_rights_priority, reservoir_rights_decree, reservoir_fill_rights, reservoir_rights_structure_common_title = crss.read_rights_data(reservoir_rights_data, structure_type = 'reservoir')
  structure_rights_name, structure_rights_structure_name, structure_rights_priority, structure_rights_decree, structure_rights_structure_common_title = crss.read_rights_data(structure_rights_data)
  instream_rights_name, instream_rights_structure_name, instream_rights_priority, instream_rights_decree, instream_rights_structure_common_title = crss.read_rights_data(instream_rights_data)
  #adjust links between water rights 
  new_structure_rights_structure_name, carrier_connections  = crss.read_operational_rights(operational_rights_data, structure_rights_structure_name, structure_rights_name)

  #create structure/reservoir objects within the basin object, create right objects within the structures
  ucrb.set_rights_to_reservoirs(reservoir_rights_name, reservoir_rights_structure_name, reservoir_rights_priority, reservoir_rights_decree, reservoir_fill_rights)
  ucrb.set_rights_to_structures(structure_rights_name, new_structure_rights_structure_name, structure_rights_priority, structure_rights_decree)
  ucrb.set_rights_to_structures(instream_rights_name, instream_rights_structure_name, instream_rights_priority, instream_rights_decree)

  #create 'rights stack' - all rights listed in the order of their priority, w/ structure names, decree amounts, etc. (single list for reservoir rights, diversion structure rights, and instream rights
  ucrb.combine_rights_data(structure_rights_name, new_structure_rights_structure_name, structure_rights_priority, structure_rights_decree, reservoir_rights_name, reservoir_rights_structure_name, reservoir_rights_priority, reservoir_rights_decree, instream_rights_name, instream_rights_structure_name, instream_rights_priority, instream_rights_decree)

  ###############################################################################################################################
  ####add water demands, deliveries, return flows, inflows, structure types, and reservoir release data to structure objects####
  print('apply demands to structures')
  #read demand data timeseries to dataframe
  structure_demands = crss.read_structure_demands(demand_data,year_start, year_end, read_from_file = False)
  #apply demands to structure objects, distribute across structure water rights (by seniority)
  ucrb.set_structure_demands(structure_demands)

  print('apply deliveries to structures')
  #read delivery data timeseries to dataframe
  structure_deliveries = crss.read_structure_deliveries(delivery_data, year_start, year_end, read_from_file = False)
  #apply deliveries to structure objects, distribute across structure water rights (by seniority)
  ucrb.set_structure_deliveries(structure_deliveries)

  #overwrite 'max' demands (=9999999.9) with maximum monthly deliveries in baseline simulation
  monthly_maximums = ucrb.adjust_structure_deliveries()

  print('apply return flows to structures')
  #read average monthly return flow data to dataframe
  structure_return_flows = crss.read_structure_return_flows(return_flow_data, year_start, year_end, read_from_file = False)
  #apply return flow fractions to structure objects
  ucrb.set_return_fractions(structure_return_flows)

  #apply structure inflow timeseries to structure objects
  structure_inflows = crss.read_structure_inflows(delivery_data, year_start, year_end)
  ucrb.set_structure_inflows(structure_inflows)

  print('set structure types')
  #classify structures by water use type, add to structure object
  ucrb.set_structure_types(input_data_dictionary)

  #read stored releases from reservoirs
  plan_flows = crss.read_plan_flows(reservoir_storage_data_b, ucrb.reservoir_list, year_start, year_end)
  #apply timeseries of releases from upstream reservoirs to structures
  ucrb.set_plan_flows(plan_flows, downstream_data, ucrb.reservoir_list,  '7202003')

  ###################################################################################################
  ####create water supply indicies with relationships between snowpack and inflow into reservoirs####
  print('create historical reservoir timeseries')
  #create timeseries of total inflow impounded by reservoirs (for use in snowpack/flow regressions)
  for res in ucrb.reservoir_list:
    ucrb.structures_objects[res].simulated_reservoir_timeseries = crss.read_simulated_reservoirs(reservoir_storage_data_b, res, year_start, year_end)
    #make a copy of reservoir storage values that can be edited by adaptive simulations & compared to the original
    ucrb.structures_objects[res].adaptive_reservoir_timeseries = ucrb.structures_objects[res].simulated_reservoir_timeseries.copy(deep = True)

  print('calculate initial water supply metrics')
  #create linear regression coefficients for the relationship between observed snowpack in each month and remaining impounded inflow to reservoirs
  snow_coefs_tot = {}
  for res in ucrb.reservoir_list:
    #univariate linear regression coefficients between the observed snowpack and future inflow to a reservoir, conditional on month of the year and any existing 'calls' on the river
    simulated_reservoir_timeseries = crss.read_simulated_reservoirs(reservoir_storage_data_b, res, year_start, year_end)
    historical_monthly_available = pd.DataFrame(list(structure_inflows[res + '_available']), index = structure_inflows.index, columns = ['available',])
    historical_monthly_control = pd.DataFrame(list(structure_inflows[res + '_location']), index = structure_inflows.index, columns = ['location',])
    snow_coefs_tot[res] = ucrb.make_snow_regressions('14010001', historical_monthly_control, simulated_reservoir_timeseries, historical_monthly_available, res, 1950, 2013)

###################################
####simulate informal transfers####
#set water supply index threshold for informal leases
#initialize informal lease dataframes
  all_change1 = pd.DataFrame()
  all_change2 = pd.DataFrame()
  all_change3 = pd.DataFrame()
  all_change4 = pd.DataFrame()
  for res in [reservoir_use,]:
    adaptive_toggle = 0#toggles informal transfers - don't need to update anything before they are triggered in the simulation
    remaining_storage = 0.0#check for any 'additional' water in the reservoir that has been leased but not diverted
    #loop through simulation years
    for year_num in range(year_start_adaptive, year_end):
      year_add = 0
      month_start = 10#start simulations in october
      trigger_purchases = False
      #set a new control file for adaptive simulations - only run simulations out through the current timestep (plus 2 years)
      #each year the adpative simulation gets one year longer
      crss.make_control_file('cm2015', 'A', year_start_adaptive - 10, min(year_num + 2, year_end))
      #loop through each month of the year
      for month_num in range(0, 12):
        if month_start + month_num == 13:
          month_start -= 12
          year_add = 1
        #set current timestep
        datetime_val = datetime(year_num + year_add, month_start + month_num, 1, 0, 0)

        #######
        #step 1 - update structure objects with simulation outputs
        #only trigger updates to adaptive simulation data once the informal leases have been triggered (until they are first triggered, adaptive runs = baseline run)
        if adaptive_toggle == 1:
          #if its been two years since the last adaptive run, we need to extend the simulation (because each simulation only generates results through the current timestep + 2 years)
          if last_year_use == year_num and month_num == 0:
            #execute adaptive StateMod simulation
            os.system("StateMod_Model_15.exe cm2015A -simulate")
            #extract unformatted data
            demand_data_new = crss.read_text_file(input_data_dictionary['structure_demand_new'])
            delivery_data_new = crss.read_text_file(input_data_dictionary['deliveries_new'])
            reservoir_storage_data_a = crss.read_text_file(input_data_dictionary['reservoir_storage_new'])
            #extend simulation for two years
            last_year_use = year_num + 2    
          #read unformatted data and extract demand & delivery values from current timestep          
          #these return dictionaries with single values for each variable
          structure_deliveries = crss.update_structure_deliveries(delivery_data_new, datetime_val.year, datetime_val.month, read_from_file = False)
          structure_demands = crss.update_structure_demands(demand_data_new, datetime_val.year, datetime_val.month, read_from_file = False)
          #read unformatted data and extract reservoir stored releases from current timestep
          #these return dictionaries with single values for each variable
          new_plan_flows = crss.update_plan_flows(reservoir_storage_data_a, ucrb.reservoir_list, datetime_val.year, datetime_val.month)
          #read unformatted data and extract output and call location data from current timestep
          new_releases = crss.update_inflows(delivery_data_new, datetime_val.year, datetime_val.month)
          new_storage = crss.update_simulated_reservoirs(reservoir_storage_data_a, ucrb.reservoir_list, datetime_val.year, datetime_val.month)

        #update structure objects with adaptive simulation output for this timestep        
          ucrb.update_structure_demand_delivery(structure_demands, structure_deliveries, monthly_maximums, datetime_val)
          ucrb.update_structure_plan_flows(new_plan_flows, datetime_val)
          ucrb.update_structure_inflows(new_releases, datetime_val)
          ucrb.update_structure_storage(new_storage, datetime_val)

      #######  
      #step 2 - calculate updated water supply index and compare with leasing threshold
        total_water = ucrb.find_available_water(snow_coefs_tot[res], tunnel_transfer_to, res, '14010001', datetime_val, 10)
        print(datetime_val, end =  " ")
        print(total_water)      
      #only lease water from April - September when lease threshold is triggered
        if total_water < res_thres[res] and datetime_val.month >= 4 and datetime_val.month <= 6:
          trigger_purchases = True
        if trigger_purchases:
        #######  
        #step 3 - find informal leasing partners
        #find volumes leased from specific water rights
        #last_right and last_structure are the most junior water right that is leased
          change_points_purchase_1, change_points_buyout_1, last_right, last_structure = ucrb.find_adaptive_purchases(downstream_data, res, datetime_val)
        #find facilitators and facilitated demands
        #end priority is the priority of the reservoir making diversions
          change_points_buyout_2, end_priority = ucrb.find_buyout_partners(last_right, last_structure, downstream_data, carrier_connections, res, datetime_val)
          change_points_buyout = pd.concat([change_points_buyout_1, change_points_buyout_2])

        #######  
        #step 4 - create adaptive demand inputs for StateMod simulations & run simulation
        #if informal leases have already been triggered, only 'update' the demand input file
          if adaptive_toggle == 1:
            crss.writepartialDDM(demand_data_new, change_points_purchase_1, change_points_buyout, month_num, year_num + 1, year_start_adaptive, year_num + 1, scenario_name = 'A')
        #if this is the first time informal leases are triggered, create a new demand input file
          else:
            crss.writenewDDM(demand_data, change_points_purchase_1, change_points_buyout, year_num + 1, month_num, scenario_name = 'A')
        #run StateMod simulation
          os.system("StateMod_Model_15.exe cm2015A -simulate")        

        #######  
        #step 5 - calculate the actual yield of the informal lease (extra storage above baseline), increase tunnel diversions to export this water from storage, and re-run simulation
        #extract unformatted reservoir data from new simulation
          reservoir_storage_data_a = crss.read_text_file(input_data_dictionary['reservoir_storage_new'])
        #get initial tunnel exports
          initial_diversion_demand = ucrb.structures_objects[tunnel_transfer_to].adaptive_monthly_deliveries.loc[datetime_val, 'deliveries'] * 1.0
        #compare adaptive reservoir storage to baseline conditions to determine how much extra water to divert through export tunnel
          new_storage = crss.update_simulated_reservoirs(reservoir_storage_data_a, ucrb.reservoir_list, datetime_val.year, datetime_val.month)
          change_points_purchase_3, change_points_buyout_3 = ucrb.set_tunnel_changes(new_storage, initial_diversion_demand, datetime_val.year, datetime_val.month, res, tunnel_transfer_to, end_priority, datetime_val)
          change_points_purchase = pd.concat([change_points_purchase_1, change_points_purchase_3])
          change_points_buyout = pd.concat([change_points_buyout, change_points_buyout_3])
        
        #make export adjustment to the original demand file
          if adaptive_toggle == 1:
            crss.writepartialDDM(demand_data_new, change_points_purchase, change_points_buyout, month_num, year_num + 1, year_start_adaptive, year_num + 1, scenario_name = 'A')
          else:
            crss.writenewDDM(demand_data, change_points_purchase, change_points_buyout, year_num + 1, month_num, scenario_name = 'A')
          os.system("StateMod_Model_15.exe cm2015A -simulate")        
        
        #######  
        #step 6 - read adaptive simulation output and update structure objects 
        #read unformatted demand, delivery, and reservoir data from adaptive simulation
          demand_data_new = crss.read_text_file(input_data_dictionary['structure_demand_new'])
          delivery_data_new = crss.read_text_file(input_data_dictionary['deliveries_new'])
          reservoir_storage_data_a = crss.read_text_file(input_data_dictionary['reservoir_storage_new'])
        
        #apply simulated delivery, demand, storage, and inflow data to model structures
          structure_deliveries = crss.update_structure_deliveries(delivery_data_new, datetime_val.year, datetime_val.month, read_from_file = False)
          structure_demands = crss.update_structure_demands(demand_data_new, datetime_val.year, datetime_val.month, read_from_file = False)
          new_releases = crss.update_inflows(delivery_data_new, datetime_val.year, datetime_val.month)
          new_storage = crss.update_simulated_reservoirs(reservoir_storage_data_a, ucrb.reservoir_list, datetime_val.year, datetime_val.month)
          ucrb.update_structure_demand_delivery(structure_demands, structure_deliveries, monthly_maximums, datetime_val)
          ucrb.update_structure_inflows(new_releases, datetime_val)
          ucrb.update_structure_storage(new_storage, datetime_val)

        #calculate if any leased water could not be exported through the tunnel
          remaining_storage = ucrb.structures_objects[res].simulated_reservoir_timeseries.loc[datetime_val, res + '_end_storage'] - new_storage[res + '_end_storage']
        #subtract any un-exported water from the 'diverted' water timeseries (this already-leased water will be counted in later timesteps)
          if remaining_storage < 0:
            change_points_purchase_3.loc[change_points_purchase_3.index[0], 'demand'] -= remaining_storage
        #save daily records of informal leases in simulation timeseries
          all_change1 = pd.concat([all_change1, change_points_purchase_1])
          all_change2 = pd.concat([all_change2, change_points_buyout_1])
          all_change3 = pd.concat([all_change3, change_points_buyout_2])
          all_change4 = pd.concat([all_change4, change_points_purchase_3])
          all_change1.to_csv(results_dir + '/purchases_' + res + '.csv')
          all_change2.to_csv(results_dir + '/buyouts_' + res + '.csv')
          all_change3.to_csv(results_dir + '/buyouts_2_' + res + '.csv')
          all_change4.to_csv(results_dir + '/diversions_' + res + '.csv')
        #trigger leases, set the trigger to update adaptive simulation forward 2 years        
          adaptive_toggle = 1
          last_year_use = year_num + 2

        elif remaining_storage < 0.0:
        #if there are no informal leases but there is still extra water in the reservoir, export it through the tunnel (no new leases)
        #get baseline tunnel export
          initial_diversion_demand = ucrb.structures_objects[tunnel_transfer_to].adaptive_monthly_deliveries.loc[datetime_val, 'deliveries'] * 1.0
        #get current adaptive storage
          new_storage = crss.update_simulated_reservoirs(reservoir_storage_data_a, ucrb.reservoir_list, datetime_val.year, datetime_val.month)
        #set demand change dataframes
          change_points_purchase, change_points_buyout = ucrb.set_tunnel_changes(new_storage, initial_diversion_demand, datetime_val.year, datetime_val.month, res, tunnel_transfer_to, end_priority, datetime_val)
        #make export adjustment to the original demand file
          if adaptive_toggle == 1:
            crss.writepartialDDM(demand_data_new, change_points_purchase, change_points_buyout, month_num, year_num + 1, year_start_adaptive, year_num + 1, scenario_name = 'A')
            last_year_use = year_num + 2
          else:
            crss.writenewDDM(demand_data, change_points_purchase, change_points_buyout, year_num + 1, month_num, scenario_name = 'A')
          os.system("StateMod_Model_15.exe cm2015A -simulate")
        #update diversion record        
          all_change4 = pd.concat([all_change4, change_points_purchase])
          all_change4.to_csv(results_dir + '/diversions_' + res + '.csv')

        #update new unformatted data
          demand_data_new = crss.read_text_file(input_data_dictionary['structure_demand_new'])
          delivery_data_new = crss.read_text_file(input_data_dictionary['deliveries_new'])
          reservoir_storage_data_a = crss.read_text_file(input_data_dictionary['reservoir_storage_new'])
        
        #apply simulated delivery, demand, storage, and inflow data to model structures
          structure_deliveries = crss.update_structure_deliveries(delivery_data_new, datetime_val.year, datetime_val.month, read_from_file = False)
          structure_demands = crss.update_structure_demands(demand_data_new, datetime_val.year, datetime_val.month, read_from_file = False)
          new_releases = crss.update_inflows(delivery_data_new, datetime_val.year, datetime_val.month)
          new_storage = crss.update_simulated_reservoirs(reservoir_storage_data_a, ucrb.reservoir_list, datetime_val.year, datetime_val.month)
          ucrb.update_structure_demand_delivery(structure_demands, structure_deliveries, monthly_maximums, datetime_val)
          ucrb.update_structure_inflows(new_releases, datetime_val)
          ucrb.update_structure_storage(new_storage, datetime_val)
        #recalculate remaining storage
          remaining_storage = ucrb.structures_objects[res].simulated_reservoir_timeseries.loc[datetime_val, res + '_end_storage'] - new_storage[res + '_end_storage']

  copy_file_list = ['.ddm', '.xca', '.xdd', '.xir', '.xop', '.xpl', '.xre', '.xrp', '.xss']   
  file_base = 'cm2015A'   
  for ender in copy_file_list:
    shutil.copyfile(file_base + ender, results_dir + '/' + file_base + ender)