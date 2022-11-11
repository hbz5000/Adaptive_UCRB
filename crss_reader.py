import numpy as np 
import pandas as pd
from datetime import datetime

def create_input_data_dictionary(baseline_scenario, adaptive_scenario, folder_name = ''):
  input_data_dictionary = {}
  ###geographic layout
  input_data_dictionary['hydrography'] = 'Shapefiles_UCRB/NHDPLUS_H_1401_HU4_GDB.gdb'
  input_data_dictionary['structures'] = 'Shapefiles_UCRB/Div_5_structures.shp'
  ##basin labels
  input_data_dictionary['HUC4'] = ['1401',]
  input_data_dictionary['HUC8'] = ['14010001', '14010002', '14010003', '14010004', '14010005']
  ##locations of large agricultural aggregations
  ###snow data
  input_data_dictionary['snow'] = 'Snow_Data/'

  ###statemod input data
  ##monthly demand data
  input_data_dictionary['structure_demand'] = 'input_files/cm2015' + baseline_scenario + '.ddm'
  ##water rights data
  input_data_dictionary['structure_rights'] = 'input_files/cm2015' + baseline_scenario + '.ddr'
  ##reservoir fill rights data
  input_data_dictionary['reservoir_rights'] = 'input_files/cm2015' + baseline_scenario + '.rer'
  ##reservoir fill rights data
  input_data_dictionary['instream_rights'] = 'cm2015.ifr'
  ##full natural flow data
  input_data_dictionary['natural flows'] = 'cm2015x.xbm'
  ##flow/node network
  input_data_dictionary['downstream'] = 'input_files/cm2015.rin'
  ##historical reservoir data
  input_data_dictionary['historical_reservoirs'] = 'input_files/cm2015.eom'
  ##call data
  input_data_dictionary['calls'] = 'cm2015' + baseline_scenario + '.xca'

  ###statemod output data
  ##reservoir storage data
  input_data_dictionary['reservoir_storage'] = 'cm2015' + baseline_scenario + '.xre'
  ##diversion data
  input_data_dictionary['deliveries'] = 'cm2015' + baseline_scenario + '.xdd'
  ##return flow data
  input_data_dictionary['return_flows'] = 'cm2015' + baseline_scenario + '.xss'
  ##plan flow data
  input_data_dictionary['plan_releases'] = 'cm2015' + baseline_scenario + '.xpl'
  
  ##adaptive reservoir output data
  input_data_dictionary['reservoir_storage_new'] = folder_name + 'cm2015' + adaptive_scenario + '.xre'
  ##adaptive diversion data
  input_data_dictionary['deliveries_new'] = folder_name + 'cm2015' + adaptive_scenario + '.xdd'
  ##adaptive demand data
  input_data_dictionary['structure_demand_new'] = folder_name + 'cm2015' + adaptive_scenario + '.ddm'
  ##adaptive return flows
  input_data_dictionary['return_flows_new'] = folder_name + 'cm2015' + adaptive_scenario + '.xss'
  ##plan flow data
  input_data_dictionary['plan_releases'] = folder_name + 'cm2015' + adaptive_scenario + '.xpl'

  input_data_dictionary['snow'] = 'Snow_Data/'
  input_data_dictionary['irrigation'] = 'Shapefiles_UCRB/Div5_Irrigated_Lands_2015/Div5_Irrig_2015.shp'
  input_data_dictionary['ditches'] = 'Shapefiles_UCRB/Div5_Irrigated_Lands_2015/Div5_2015_Ditches.shp'
  input_data_dictionary['aggregated_diversions'] = 'output_files/aggregated_diversions.txt'


  return input_data_dictionary

def create_datetime_index(initial_year, end_year, start_month):
  datetime_index = []
  for year_count in range(initial_year, end_year):
    month_num = start_month
    year_add = 0
    for month_count in range(0, 12):
      datetime_index.append(datetime(year_count + year_add, month_num, 1, 0, 0))
      month_num += 1
      if month_num == 13:
        month_num = 1
        year_add = 1
        
  return datetime_index
  
def make_month_num_dict():
  month_num_dict = {}
  counter = 10
  for month_name in ['OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP']:
    month_num_dict[month_name] = counter
    counter += 1
    if counter == 13:
      counter = 1
  return month_num_dict  
    

def make_control_file(old_filename, scenario_name, year_start, year_end):
  #read existing baseline control file and adjust simulation time period
  #read existing baseline control file
  with open(old_filename + '.ctl','r') as f:
    all_split_data = [x for x in f.readlines()]       
  f.close()
  #create new control file with adjusted simulation time period
  f = open(old_filename + scenario_name + '.ctl','w')
  # write firstLine # of rows as in initial file
  for i in range(0, len(all_split_data)):
    #changes to simulation period line
    if i == 7:
      f.write('    ' + str(year_start) + '     : iystr   STARTING YEAR OF SIMULATION\n')
    elif i == 8:
      f.write('    ' + str(year_end) + '     : iyend   ENDING YEAR OF SIMULATION\n')
    #otherwise copy exactly
    else:
      f.write(all_split_data[i])
  #new file is written and can be closed
  f.close()


def read_text_file(filename):
  #extract unformatted data from StateMod
  with open(filename,'r') as f:
    all_split_data = [x for x in f.readlines()]       
  f.close()
  return all_split_data

def writenewDDM(demand_data, structures_purchase, structures_buyout, change_year, change_month, scenario_name = 'A'):
  #this function writes a new StateMod demand input file
  #create a list that stores each row of the file as a list
  new_data = []
  
  use_value = 0
  start_loop = 0
  #demand data is the unformatted data from the baseline demand 
  for i in range(0, len(demand_data)):
    #find first line of data
    if use_value == 1:
      start_loop = 1
    if demand_data[i][0] != '#':
      use_value = 1
    #first line of data
    if start_loop == 1:
      #get line date and structure name
      monthly_values = demand_data[i].split('.')
      first_data = monthly_values[0].split()
      use_line = True
      try:
        year_num = int(first_data[0])
        structure_name = str(first_data[1]).strip()
      except:
        use_line = False      
      
      #list of values (strings) in each row
      row_data = []
      #only change the lines associated with this year
      if use_line and year_num == change_year:
        #check if this structure is a facilitator
        this_buyout_structure = structures_buyout[structures_buyout['structure'] == structure_name]
        #check if this structure is a lease seller
        this_purchase_structure = structures_purchase[structures_purchase['structure'] == structure_name]
        #each line has 12 monthly demand values (starting with October) plus an annual total
        new_demands = np.zeros(13)
        #add the first three columns to the new row data (year, structure id, and October demand)
        row_data.extend(first_data)
        toggle_use = False
        #if this structure is a lease facilitator, change the demand to 
        #the value given in this_buyout_structure
        if len(this_buyout_structure) > 0:
          toggle_use = True
          for index, row in this_buyout_structure.iterrows():
            new_demands[change_month] = row['demand'] * 1.0
        #if this structure is a lease seller, reduce the demand to 
        #the value given in this_purchase_structure
        if len(this_purchase_structure) > 0:
          toggle_use = True
          for index, row in this_purchase_structure.iterrows():
            new_demands[change_month] -= row['demand'] * 1.0
        #if there are any demand changes in this row, append the changes to the row_data list
        if toggle_use:
          total_new_demand = 0.0
          #october demands are already appended to the list, so 
          #if october is changed the 3 index location in the list needs to be changed
          if change_month == 0:
            #keep track of total annual demand
            total_new_demand += new_demands[change_month]
            #change value in list
            row_data[2] = str(int(new_demands[change_month]))
          else:
            #keep track of total annual demand
            total_new_demand += float(row_data[2])
          #loop through the remaining months
          for j in range(0, len(monthly_values)-3):
            #if this is the month to change, append new demand
            if j+1 == change_month:
              total_new_demand += new_demands[change_month]
              row_data.append(str(int(new_demands[change_month])))
            #if not append demands from original data
            else:
              total_new_demand += float(monthly_values[j+1])
              row_data.append(str(int(float(monthly_values[j+1]))))
          #append the total demand to the last position
          row_data.append(str(int(total_new_demand)))
        #if there is no data to change at this structure, append old demand data from that row
        else:
          for j in range(len(monthly_values)-2):
            row_data.append(str(int(float(monthly_values[j+1]))))    
      #if the line is not from the year to be changed, append old data
      elif use_line:
        row_data.extend(first_data)
        for j in range(len(monthly_values)-2):
          row_data.append(str(int(float(monthly_values[j+1]))))
      #if line is not correct data format, append old data but dont convert to int first
      else:
        row_data.extend(first_data)
        for j in range(len(monthly_values)-2):
          row_data.append(str(monthly_values[j+1]))
      #append list of values in this line to list of lines
      new_data.append(row_data)                
  
  #once all data has been written or rewritten to the list, write list to file
  # write new demand data file with new scenario appended on the end of the filename (in place of B for baseline)
  f = open('cm2015' + scenario_name + '.ddm','w')
  # write firstLine # of rows as in initial file
  i = 0
  #write old file up to first line of data
  while demand_data[i][0] == '#':
    f.write(demand_data[i])
    i += 1
  #write first line of data header file
  f.write(demand_data[i])
  i+=1
  #write remainder of file from the list of row lists compiled above
  for i in range(len(new_data)):
    # write year, ID and first month of adjusted data w/ appropriate white space 
    f.write(new_data[i][0] + ' ' + new_data[i][1] + (19-len(new_data[i][1])-len(new_data[i][2]))*' ' + new_data[i][2] + '.')
    # write all but last month of adjusted data
    for j in range(len(new_data[i])-4):
      f.write((7-len(new_data[i][j+3]))*' ' + new_data[i][j+3] + '.')                
        # write last month of adjusted data
    f.write((9-len(new_data[i][-1]))*' ' + new_data[i][-1] + '.' + '\n')            
  f.close()
    
  return None

def writepartialDDM(demand_data, structures_purchase, structures_buyout, change_month, change_year, begin_year, end_year, scenario_name = 'A', structure_list = 'all'):    
  #this function writes a new StateMod demand input file by adjusting the 
  #unformatted data extracted from a previous demand input file
  
  #create a list that stores each row of the file as a list
  new_data = []
  use_value = 0
  start_loop = 0
  
  for i in range(0, len(demand_data)):
    #find first line of baseline demands
    if use_value == 1:
      start_loop = 1
    if demand_data[i][0] != '#':
      use_value = 1
    #after the first non-# line, read data
    if start_loop == 1:
      #split unformatted data into columns
      monthly_values = demand_data[i].split('.')
      first_data = monthly_values[0].split()
      use_line = True
      #create a list for all values in the current row
      row_data = []
      #check to see if line has property formatted data
      try:
        year_num = int(first_data[0])
        structure_name = str(first_data[1]).strip()
      except:
        use_line = False
      if use_line:
        if structure_name in structure_list or structure_list == 'all':
          #if the data file line is within the adaptive simulation period, check for changes        
          if year_num >= begin_year and year_num <= end_year and year_num == change_year:
            #find if current structure is lease facilitator
            this_buyout_structure = structures_buyout[structures_buyout['structure'] == structure_name]
            #find if current structure is lease seller
            this_purchase_structure = structures_purchase[structures_purchase['structure'] == structure_name]
            #each line contains 12 monthly demand values (starting in october) plus an annual total
            new_demands = np.zeros(13)
            #append the first three lines of the row to the new list
            row_data.extend(first_data)
            toggle_use = False
            #check if any buyouts for this structure, calculate the new demands
            if len(this_buyout_structure) > 0:
              toggle_use = True
              for index, row in this_buyout_structure.iterrows():
                new_demands[change_month] = row['demand'] * 1.0
            #check if any purchases for this structure, calculate the new demands
            if len(this_purchase_structure) > 0:
              toggle_use = True
              for index, row in this_purchase_structure.iterrows():
                new_demands[change_month] -= row['demand'] * 1.0
            #append new demands to the list of row values
            if toggle_use:
              total_new_demand = 0.0
              #if changes happen to october - need to change the value that has already been appended to list of row values
              if change_month == 0:
                total_new_demand += new_demands[change_month]
                row_data[2] = str(int(new_demands[change_month]))
              else:
                total_new_demand += float(row_data[2])
              #append monthly values, including the values for the changed month
              for j in range(0, len(monthly_values)-3):
                if j+1 == change_month:
                  total_new_demand += new_demands[change_month]
                  row_data.append(str(int(new_demands[change_month])))
                else:
                  total_new_demand += float(monthly_values[j+1])
                  row_data.append(str(int(float(monthly_values[j+1]))))   
              row_data.append(str(int(total_new_demand)))
            else:
              for j in range(len(monthly_values)-2):
                row_data.append(str(int(float(monthly_values[j+1]))))
          #if line is outside of the adaptive simulation period, use baseline demands                
          else:
            row_data.extend(first_data)
            for j in range(len(monthly_values)-2):
              row_data.append(str(int(float(monthly_values[j+1]))))
        #if line is a structure that is not subject to changes, use baseline demands 
        else:
          row_data.extend(first_data)
          for j in range(len(monthly_values)-2):
            row_data.append(str(int(float(monthly_values[j+1]))))
      #if line is not data, copy original demand file (no change to int)
      else:
        row_data.extend(first_data)
        for j in range(len(monthly_values)-2):
          row_data.append(str(monthly_values[j+1]))
      new_data.append(row_data)
    # write new data to file
  f = open('cm2015' + scenario_name + '.ddm','w')
  # write firstLine # of rows as in initial file
  j = 0
  while demand_data[j][0] == '#':
    f.write(demand_data[j])
    j += 1
  f.write(demand_data[j])
  j+=1
  for i in range(len(new_data)):
    # write year, ID and first month of adjusted data
    f.write(new_data[i][0] + ' ' + new_data[i][1] + (19-len(new_data[i][1])-len(new_data[i][2]))*' ' + new_data[i][2] + '.')
    # write all but last month of adjusted data
    for j in range(len(new_data[i])-4):
      f.write((7-len(new_data[i][j+3]))*' ' + new_data[i][j+3] + '.')                
        # write last month of adjusted data
    f.write((9-len(new_data[i][-1]))*' ' + new_data[i][-1] + '.' + '\n')
  for j in range(j+len(new_data), len(demand_data)):
    f.write(demand_data[j])
    
  f.close()
    
  return None

def read_rights_data(all_data_DDR, structure_type = 'structure'):
  #reads unformatted water rights data and links right ids, right priority, and right decree volume to structure ids

  #reservoir rights are formatted differently than other rights (extra columns)
  if structure_type == 'reservoir':
    column_lengths=[12,24,12,16,8,8,8,8,8]
    all_rights_fill_type = []
  else:
    column_lengths=[12,24,12,16,8,8]  

  #set start/end location of individual data columns
  split_line = ['']*len(column_lengths)
  character_breaks=np.zeros(len(column_lengths),dtype=int)
  character_breaks[0]=column_lengths[0]
  for i in range(1,len(column_lengths)):
    character_breaks[i]=character_breaks[i-1]+column_lengths[i]
  
  #link right, structure data in ordered lists
  all_rights_name = []
  all_rights_priority = []
  all_rights_decree = []
  all_rights_structure_name = []
  all_rights_structure_common_title = []
  #loop through unformatted data
  for j in range(0,len(all_data_DDR)):
    #extract values from unformatted data
    if all_data_DDR[j][0] == '#':
      first_line = int(j * 1)
    else:
      split_line[0]=all_data_DDR[j][0:character_breaks[0]]
      for i in range(1,len(split_line)):
        split_line[i]=all_data_DDR[j][character_breaks[i-1]:character_breaks[i]]
      structure_name = str(split_line[2].strip())
      right_name = str(split_line[0].strip())
      right_priority = float(split_line[3].strip())
      structure_common_name = str(split_line[1].strip())
      if int(split_line[5].strip()) == 1:
        right_decree = float(split_line[4].strip())
      else:
        right_decree = 0.0
        
      #set extra reservoir variables
      if structure_type == 'reservoir':
        fill_type = int(split_line[8].strip())
        all_rights_fill_type.append(fill_type)
      #set list of right/structure attributes (structure & reservoirs)
      all_rights_name.append(right_name)
      all_rights_priority.append(right_priority) 
      all_rights_decree.append(right_decree)
      all_rights_structure_name.append(structure_name)
      all_rights_structure_common_title.append(structure_common_name)

  if structure_type == 'reservoir':
    return all_rights_name, all_rights_structure_name, all_rights_priority, all_rights_decree, all_rights_fill_type, all_rights_structure_common_title
  else:
    return all_rights_name, all_rights_structure_name, all_rights_priority, all_rights_decree, all_rights_structure_common_title

def read_operational_rights(operational_plans, structure_rights_structure_name, structure_rights_name):
  #this reads the operational plan input file to extract
  #structures that are 'using' water rights officially associated with different structures
  #this can represent water transfers, conveyance structures, etc.
  
  #column breaks for right change lines in the operational file
  col_start = [0, 12, 53, 70, 79, 81, 99, 102, 121, 123, 142, 149, 152, 165, 185, 193, 201, 206, 211] 
  right_list = []
  structure_list = []
  
  for i in range(0, len(operational_plans)):
    if operational_plans[i][0] == '#':
      pass
    else:
      #check format to find a specific 'kind' of rule
      check_row = True
      monthly_vals = operational_plans[i].split()
      if len(monthly_vals) == 13:
        check_row = False
        for x in range(0, len(monthly_vals)):
          try:
            new_val = float(monthly_vals[x])
          except:
            check_row = True
      elif len(monthly_vals) == 3 or len(monthly_vals) == 12 or len(monthly_vals) == 1 or len(monthly_vals) == 2:
        check_row = False
      #extract unformatted rule data
      if check_row:
        for col_loc in range(0, len(col_start)):
          if col_loc == len(col_start) - 1:
            value_use = operational_plans[i][col_start[col_loc]:].strip()
          else:          
            value_use = operational_plans[i][col_start[col_loc]:col_start[col_loc+1]].strip()
          if col_loc == 5:
            #structure column
            this_structure = str(value_use)
          elif col_loc == 7:
            #right column
            this_right = str(value_use)
          elif col_loc == 11:
            #rule type column
            if str(value_use) == '11' or str(value_use) == '14' or str(value_use) == '45':
              if this_right not in right_list:
                right_list.append(this_right)
                structure_list.append(this_structure)
  
    
  #create a copy of the list of original structure names
  #this is the list we will change one line at a time
  new_structure_rights_structure_name = []
  for xxx in structure_rights_structure_name:
    new_structure_rights_structure_name.append(xxx)
  #create
  carrier_connections = {}
  #loop through linked list of structures and their alternative water rights
  for new_struct, right_find in zip(structure_list, right_list):
    new_structure_rights_structure_name_int = []
    #loop through existing connections between structures and rights to find the changed right
    #use the copied list of structure ids
    for old_struct, right_look in zip(new_structure_rights_structure_name, structure_rights_name):
      if right_look == right_find:
        #if you find the changed right, put the new structure in the structure list
        new_structure_rights_structure_name_int.append(new_struct)
        #take new structure and link it to the old structures from which it is using the water right
        if new_struct in carrier_connections:
          carrier_connections[new_struct].append(old_struct)
        else:
          carrier_connections[new_struct] = [old_struct,]
      else:
        #if its a different right, keep the structure the same
        new_structure_rights_structure_name_int.append(old_struct)
    #re-write the 'copied' structure id list, but with new structures in place
    new_structure_rights_structure_name = []
    for xxx in new_structure_rights_structure_name_int:
      new_structure_rights_structure_name.append(xxx)
  
  
  return new_structure_rights_structure_name, carrier_connections


def read_structure_demands(demand_data, initial_year, end_year, read_from_file = False, read_filename = ''):
  #reading from csv file cuts down on reading time
  if read_from_file:
    structure_demand = pd.read_csv(read_filename, index_col = 0)
    structure_demand.index = pd.to_datetime(structure_demand.index)
  else:
    #create monthly index for demand timeseries at each structure
    datetime_index = create_datetime_index(initial_year, end_year, 10)
    year_num = 0
    i = 0
    #create a list of structure headers
    structure_list = []
    while year_num < 1910:
      if demand_data[i][0] != '#':
        monthly_values = demand_data[i].split('.')
        first_data = monthly_values[0].split()
        use_line = True
        try:
          year_num = int(first_data[0])
          structure_name = str(first_data[1])
          if year_num < 1910:
            structure_list.append(structure_name)
        except:
          use_line = False
      i += 1    
    structure_demand = pd.DataFrame(index = datetime_index, columns = structure_list)
  
    for x in structure_list:
      structure_demand[x] = np.zeros(len(datetime_index))

    #read unformatted data 
    for i in range(len(demand_data)):
      if demand_data[i][0] != '#':
        monthly_values = demand_data[i].split('.')
        first_data = monthly_values[0].split()
        use_line = True
        try:
          year_num = int(first_data[0])
          structure_name = str(first_data[1])
        except:
          use_line = False
        if use_line:
          #each line is one year of monthly data
          datetime_val = datetime(year_num - 1, 10, 1, 0, 0)
          structure_demand.loc[datetime_val, structure_name] = float(first_data[2])     
          for month_num in range(0, 11):
            if month_num < 2:
              datetime_val = datetime(year_num - 1, month_num + 11, 1, 0, 0)
            else:
              datetime_val = datetime(year_num, month_num - 1, 1, 0, 0)
            structure_demand.loc[datetime_val, structure_name] = float(monthly_values[month_num + 1])     

    if len(read_filename) > 0:
      structure_demand.to_csv(read_filename)
  
  return structure_demand  


def read_structure_deliveries(delivery_data, initial_year, end_year, read_from_file = False, read_filename = ''):
  if read_from_file:
    structure_deliveries = pd.read_csv(read_filename, index_col = 0)
    structure_deliveries.index = pd.to_datetime(structure_deliveries.index)
  else:
    month_num_dict = make_month_num_dict()#dictionary to translate StateMod month keys into integers
    datetime_index = create_datetime_index(initial_year, end_year, 10)#datetime index for dataframe
  
    #read unformatted simulation output to get different types of water deliveries to each user
    structure_deliveries = pd.DataFrame(index = datetime_index)
    for line in delivery_data:
      data = line.split()
      if data:
        if len(data) > 1 and data[0] !='#':
          struct_id = str(data[0].strip())
          if struct_id == 'Baseflow' or struct_id == 'NA':
            struct_id = str(data[1].strip())
          try:
            month_id = data[3].strip()
            year_id = int(data[2].strip())
            month_number = month_num_dict[month_id]
            use_line = True
          except:
            use_line = False            
          if use_line:             
            datetime_val = datetime(year_id, month_number, 1, 0, 0)      
            structure_deliveries.loc[datetime_val, struct_id] = float(data[16].strip()) - float(data[15].strip())
            structure_deliveries.loc[datetime_val, struct_id + '_priority'] = float(data[6].strip()) + float(data[11].strip())
            structure_deliveries.loc[datetime_val, struct_id + '_return'] = float(data[21].strip())
            structure_deliveries.loc[datetime_val, struct_id + '_flow'] = float(line[250:257].strip())
              
    if len(read_filename) > 0:
      structure_deliveries.to_csv(read_filename)
  
  return structure_deliveries  

def read_structure_return_flows(return_flow_data, initial_year, end_year, read_from_file = False, read_filename = ''):
  #this function calculates return flows in each month as an average % of the diversions in that month, using baseline simulation output
  if read_from_file:
    structure_return_flows = pd.read_csv(read_filename, index_col = 0)
    structure_return_flows.index = pd.to_datetime(structure_return_flows.index)
  else:
    month_name_list = ['OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP']
    structure_return_flows = pd.DataFrame(index = month_name_list)
    structure_diversions = pd.DataFrame(index = month_name_list)
    structure_returns = pd.DataFrame(index = month_name_list)
    #find total structure diversions & total structure return flows in the baseline simulation
    #aggregated to each month of the year (sum of all simulation years)
    for line in return_flow_data:
      data = line.split()
      if data:
        if len(data) > 1 and data[0] !='#':
          struct_id = str(data[0].strip())
          try:
            month_id = data[2].strip()
            year_id = int(data[1].strip())
            if month_id in month_name_list:
              use_line = True
            else:
              use_line = False
          except:
            use_line = False            
          if use_line:
            if struct_id in structure_diversions.columns:           
              structure_diversions.loc[month_id, struct_id] += float(data[12].strip())
            else:
              structure_diversions[struct_id] = np.zeros(len(month_name_list))
              structure_diversions.loc[month_id, struct_id] += float(data[12].strip())

            if struct_id in structure_returns.columns:           
              structure_returns.loc[month_id, struct_id] += float(data[17].strip())
            else:
              structure_returns[struct_id] = np.zeros(len(month_name_list))
              structure_returns.loc[month_id, struct_id] += float(data[17].strip())
    #take % of diversions 'returned' in each month of the year
    for struct_id in structure_diversions.columns:
      structure_return_flows[struct_id] = np.zeros(len(month_name_list))
      for month_id in month_name_list:
        if structure_diversions.loc[month_id, struct_id] > 0.0:
          structure_return_flows.loc[month_id, struct_id] = structure_returns.loc[month_id, struct_id] / structure_diversions.loc[month_id, struct_id]
    if len(read_filename) > 0:        
      structure_return_flows.to_csv(read_filename)
  
  return structure_return_flows  

def read_structure_inflows(delivery_data, initial_year, end_year):
  #read unformatted data to get dataframes with timeseries of flow at structure locations and timeseries of control locations & flow at all other stations
  month_num_dict = make_month_num_dict()#dictionary to translate StateMod month keys into integers
  datetime_index = create_datetime_index(initial_year, end_year, 10)#datetime index for dataframe
  simulated_releases = pd.DataFrame(index = datetime_index)  
  #extract unformatted data from diversion output (.xdd)            
  for line in delivery_data:
    monthly_values = line.split('.')
    if len(monthly_values) > 0:
      first_data = monthly_values[0].split()
      if len(monthly_values) > 1 and first_data[0] !='#':
        use_line = True
        struct_id = first_data[0].strip()
        if struct_id == 'Baseflow' or struct_id == 'NA':
          struct_id = first_data[1].strip()
        try:
          month_id = first_data[3].strip()
          year_id = int(first_data[2].strip())
          month_number = month_num_dict[month_id]
        except:
          use_line = False            
        if use_line:             
          datetime_val = datetime(year_id, month_number, 1, 0, 0)
          last_values = monthly_values[29].split()
          #location of water right call
          simulated_releases.loc[datetime_val, struct_id + '_location'] = str(last_values[0].strip())
          #total available flow for diversions
          simulated_releases.loc[datetime_val, struct_id + '_available'] = float(monthly_values[28].strip())
          #total flow past structure 
          simulated_releases.loc[datetime_val, struct_id + '_inflow'] = float(monthly_values[24].strip())
            
  return simulated_releases

def read_plan_flows(reservoir_data, reservoir_list, initial_year, end_year):
  #this function reads unformatted reservoir data to determine releases from storage to storage account owners
  #becuase this water was previously diverted (impounded) by right holders, senior right holders cannot make a 'call' on these releaess
  month_num_dict = make_month_num_dict()#dictionary to translate StateMod month keys into integers
  datetime_index = create_datetime_index(initial_year, end_year, 10)#datetime index for dataframe
      
  simulated_storage = pd.DataFrame(index = datetime_index, columns = reservoir_list)
  for res in reservoir_list:
    simulated_storage[res] = np.zeros(len(datetime_index))
  #extract unformatted data
  for i in range(len(reservoir_data)):
    use_line = True
    monthly_values = reservoir_data[i].split('.')
    first_data = monthly_values[0].split()
    if len(first_data) > 1:
      structure_name = str(first_data[0])
      if structure_name in reservoir_list:
        try:
          account_num = int(first_data[1])
          year_num = int(first_data[2])
          month_num = month_num_dict[str(first_data[3])]
        except:
          use_line = False
        if use_line and account_num == 0:
          datetime_val = datetime(year_num, month_num, 1, 0, 0)
          simulated_storage.loc[datetime_val, structure_name] = float(monthly_values[19])

  return simulated_storage  

def read_simulated_reservoirs(reservoir_data, reservoir_list, initial_year, end_year, year_read = 'all'):
  #this reads simulated output for reservoir storage & how much water is impounded by reservoirs

  #set datetime_index
  datetime_index = create_datetime_index(initial_year, end_year, 10)
  month_num_dict = make_month_num_dict()#dictionary to translate StateMod month keys into integers
        
  #create dataframe for formatted data
  simulated_storage = pd.DataFrame(index = datetime_index, columns = [reservoir_list,])
  simulated_storage[reservoir_list] = np.zeros(len(datetime_index))
  #loop through lines & extract storage and diversion data
  for i in range(len(reservoir_data)):
    use_line = True
    monthly_values = reservoir_data[i].split('.')
    first_data = monthly_values[0].split()
    if len(first_data) > 1:
      structure_name = str(first_data[0])
      if structure_name == reservoir_list:
        try:
          account_num = int(first_data[1])
          year_num = int(first_data[2])
          month_num = month_num_dict[str(first_data[3])]
        except:
          use_line = False
        if use_line and account_num == 0:
          datetime_val = datetime(year_num, month_num, 1, 0, 0)
          simulated_storage.loc[datetime_val, structure_name] = float(first_data[4])
          simulated_storage.loc[datetime_val, structure_name + '_diversions'] = float(monthly_values[8])
          simulated_storage.loc[datetime_val, structure_name + '_end_storage'] = float(monthly_values[15])
        else:
          datetime_val = datetime(year_num, month_num, 1, 0, 0)
          simulated_storage.loc[datetime_val, structure_name + '_account_' + str(account_num)] = float(first_data[4])
          simulated_storage.loc[datetime_val, structure_name + '_diversions_' + str(account_num)] = float(monthly_values[8])
          simulated_storage.loc[datetime_val, structure_name + '_end_storage_' + str(account_num)] = float(monthly_values[15])

  return simulated_storage  

def update_structure_deliveries(delivery_data, update_year, update_month, read_from_file = False):

  month_num_dict = make_month_num_dict()#dictionary to translate StateMod month keys into integers
  structure_deliveries = {}
  #read unformatted data and extract delivery updates from a single timestep
  for line in delivery_data:
    data = line.split()
    if data:
      use_line = True
      if len(data) > 1 and data[0] !='#':
        try:
          month_id = data[3].strip()
          year_number = int(data[2].strip())
          month_number = month_num_dict[month_id]
        except:
          use_line = False
        if use_line:
          #only update for specified day
          if year_number == update_year and month_number == update_month:
            struct_id = str(data[0].strip())
            if struct_id == 'Baseflow' or struct_id == 'NA':
              struct_id = str(data[1].strip())

            structure_deliveries[struct_id] = float(data[16].strip()) - float(data[15].strip())
            structure_deliveries[struct_id + '_priority'] = float(data[6].strip()) + float(data[11].strip())
            structure_deliveries[struct_id + '_return'] = float(data[21].strip())
            structure_deliveries[struct_id + '_flow'] = float(line[250:257].strip())
                    
  return structure_deliveries  

def update_structure_demands(demand_data, update_year, update_month, read_from_file = False):

  structure_demand = {}
  #read unformatted data and extract demand updates from a single timestep
  for i in range(len(demand_data)):
    if demand_data[i][0] != '#':
      monthly_values = demand_data[i].split('.')
      first_data = monthly_values[0].split()
      use_line = True
      try:
        year_num = int(first_data[0])
        structure_name = str(first_data[1])
      except:
        use_line = False
      if use_line and year_num == update_year:
        if update_month == 10:
          structure_demand[structure_name] = float(first_data[2])
        elif update_month == 11 or update_month == 12:
          structure_demand[structure_name] = float(monthly_values[update_month - 10])
        else:
          structure_demand[structure_name] = float(monthly_values[update_month + 2])
  
  return structure_demand  
  
def update_plan_flows(reservoir_storage_data_b, reservoir_list, update_year, update_month):
        
  month_num_dict = make_month_num_dict()#dictionary to translate StateMod month keys into integers
      
  #extract data on reservoir releass
  new_plan_flows = {}
  for i in range(len(reservoir_storage_data_b)):
    use_line = True
    monthly_values = reservoir_storage_data_b[i].split('.')
    first_data = monthly_values[0].split()
    if len(first_data) > 1:
      structure_name = str(first_data[0])
      if structure_name in reservoir_list:
        try:
          account_num = int(first_data[1])
          year_num = int(first_data[2])
          month_num = month_num_dict[str(first_data[3])]
        except:
          use_line = False
        #only use current timestep
        if use_line and account_num == 0 and year_num == update_year and month_num == update_month:
          datetime_val = datetime(year_num, month_num, 1, 0, 0)
          new_plan_flows[structure_name] = float(monthly_values[19])
  
  return new_plan_flows

def update_inflows(delivery_data, update_year, update_month):

  month_num_dict = make_month_num_dict()#dictionary to translate StateMod month keys into integers

  simulated_releases = {}
  #extract data on inflow and control location for a specific timestep  
  for line in delivery_data:
    monthly_values = line.split('.')
    if len(monthly_values) > 0:
      first_data = monthly_values[0].split()
      use_line = True
      if len(monthly_values) > 1 and first_data[0] !='#':
        try:
          month_id = first_data[3].strip()
          year_id = int(first_data[2].strip())
          month_number = month_num_dict[month_id]
        except:
          use_line = False
        if use_line:
          if month_number == update_month and year_id == update_year:
            struct_id = str(first_data[1].strip())
            last_values = monthly_values[29].split()
            simulated_releases[struct_id + '_location'] = str(last_values[0].strip())
            simulated_releases[struct_id + '_available'] = float(monthly_values[28].strip())
            simulated_releases[struct_id + '_inflow'] = float(monthly_values[24].strip())

  return simulated_releases

def update_simulated_reservoirs(reservoir_data, reservoir_list, update_year, update_month):
  #this reads simulated output for reservoir storage & how much water is impounded by reservoirs

  simulated_storage = {}#dictionary to store updated structure data
  month_num_dict = make_month_num_dict()#dictionary to translate StateMod month keys into integers
  for i in range(len(reservoir_data)):
    use_line = True
    monthly_values = reservoir_data[i].split('.')
    first_data = monthly_values[0].split()
    if len(first_data) > 1:
      structure_name = str(first_data[0])
      try:
        account_num = int(first_data[1])
        year_num = int(first_data[2])
        month_num = month_num_dict[str(first_data[3])]
      except:
        use_line = False
      if use_line and account_num == 0 and structure_name in reservoir_list:
        if update_year == year_num and update_month == month_num:
          simulated_storage[structure_name] = float(first_data[4])
          simulated_storage[structure_name + '_diversions'] = float(monthly_values[8])
          simulated_storage[structure_name + '_end_storage'] = float(monthly_values[15])
        else:
          simulated_storage[structure_name + '_account_' + str(account_num)] = float(first_data[4])
          simulated_storage[structure_name + '_diversions_' + str(account_num)] = float(monthly_values[8])
          simulated_storage[structure_name + '_end_storage_' + str(account_num)] = float(monthly_values[15])

  return simulated_storage  

def find_historical_max_deliveries(structure_deliveries, structure_id):
  #this function is used in the generation of alternate tunnel 
  month_list = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
  annual_deliveries = []
  total_annual_deliveries = 0.0
  monthly_deliveries = {}
  for x in month_list:
    monthly_deliveries[x] = []
  #loop through all deliveries for the selected structure  
  for index, row in structure_deliveries.iterrows():
    if index.month == 10 and total_annual_deliveries > 0.0:
      annual_deliveries.append(total_annual_deliveries)#make list of total annual deliveries
      total_annual_deliveries = 0.0
    monthly_deliveries[month_list[index.month - 1]].append(row[structure_id])#make list of monthly deliveries for each month
    total_annual_deliveries += row[structure_id]
      
  #get max annual and monthly delivery
  max_annual = np.max(annual_deliveries)
  max_monthly = np.zeros(12)
  for x_cnt, x in enumerate(month_list):
    max_monthly[x_cnt] = np.max(monthly_deliveries[x])
    
  return max_monthly, max_annual

def get_alfalfa_residuals():
  #this calculates annual change in alfalfa prices 
  alfalfa_hist_prices = pd.read_csv('input_files/crop_costs/alfalfa_prices_historical.csv')#load annual alfalfa price timeseries
  residual = np.zeros(len(alfalfa_hist_prices['year']))
  counter = 0
  #put prices in annual order
  sorted_hist_prices = alfalfa_hist_prices.sort_values(by = ['year'])
  for index, row in sorted_hist_prices.iterrows():
    if counter > 0:
      #find annual change in alfalfa prices
      residual[counter] = row['alfalfa'] - last_year_price
    last_year_price = float(row['alfalfa']) * 1.0
    counter += 1
  #return timeseries of annual changes in alfalfa price
  return residual










