import numpy as np 
import pandas as pd
from rights import Rights

class Reservoir():

  def __init__(self, station_name, station_id, storage_capacity):
    #initialize the structure object
    self.name = station_name
    self.idnum = station_id
    self.capacity = storage_capacity
    self.struct_type = 'reservoir'
    self.rights_list = []#list of potential right ids
    self.rights_objects = {}#right objects associated with the structure
    self.use_adaptive = False
    
  def initialize_right(self, right_name, right_priority, right_decree, fill_type = 'none'):
    #this creates rights associated with a spcific structure
    #right_name = id
    #right_priority = 'age' of the right, lowest number is olders
    #right_decree = maximum diversion using the right
    #fill_type = either primary or secondary fill right for a reservoir
    self.rights_list.append(right_name)
    if fill_type == 'none':
      self.rights_objects[right_name] =  Rights(right_name, right_priority, right_decree)
    else:
      self.rights_objects[right_name] = Rights(right_name, right_priority, right_decree, fill_type = fill_type)

  def make_sorted_rights_list(self):
    #this function sorts water rights associated with the structure from most senior to most junior
    priority_rank = []
    #priority is the seniority, with lowest number most senior
    for right_name in self.rights_list:
      priority_rank.append(self.rights_objects[right_name].priority)
    #sort the order and append them to sorted rights list
    priority_order = np.argsort(priority_rank)
    self.sorted_rights = []
    for priority_counter in range(0, len(priority_order)):
      self.sorted_rights.append(self.rights_list[priority_order[priority_counter]])

  def assign_demand_rights(self):
    #this function takes total demands at a structure in a given timestep and distributes it among the water rights at that structure
    for ind_right in self.sorted_rights:
      self.rights_objects[ind_right].initialize_demands(self.historical_monthly_demand.index)
    #total demand at the structure is already assigned
    #baseline demands are a timeseries at each structure
    structure_demands = np.asarray(self.historical_monthly_demand['demand'])
    rights_demands = np.zeros((len(self.historical_monthly_demand.index), len(self.sorted_rights)))
    for month_step in range(0, len(self.historical_monthly_demand.index)):
      remaining_demand = structure_demands[month_step]
      rights_counter = 0
      #in each timestep, assign demand to each right, starting with the most senior
      #each right is assigned demand up to the maximum decree amount or until there is no more demand to distribute
      for ind_right in self.sorted_rights:
        rights_demands[month_step, rights_counter] = min(max(remaining_demand, 0.0), self.rights_objects[ind_right].decree_af[self.historical_monthly_demand.index[month_step].month - 1])
        remaining_demand -=  min(max(remaining_demand, 0.0), self.rights_objects[ind_right].decree_af[self.historical_monthly_demand.index[month_step].month - 1])
        rights_counter += 1
    counter = 0
    #assign the calculated demand to the right objects
    for ind_right in self.sorted_rights:
      self.rights_objects[ind_right].historical_monthly_demand['demand'] = rights_demands[:,counter]
      counter += 1
      
  def assign_delivery_rights(self):
    #this function takes deliveries assigned to a structure and distributes those deliveries to specific water rights at that structure
    #deliveries are distributed to most senior water rights first
    #also assigns return flows & priority diversions (only for plotting purposes)
    for ind_right in self.sorted_rights:
      #set up dataframes in the right objects at this structure
      self.rights_objects[ind_right].initialize_delivery(self.historical_monthly_deliveries.index)
    
    for delivery_type in ['deliveries', 'priority']:
      #structure deliveries have already been assigned
      structure_deliveries = np.asarray(self.historical_monthly_deliveries[delivery_type])
      return_flows = np.asarray(self.historical_monthly_deliveries['return'])
      #set up arrays to store deliveries by right
      if delivery_type == 'deliveries':
        #divide return flows as a function of the total deliveries
        rights_return = np.zeros((len(self.historical_monthly_deliveries.index), len(self.sorted_rights)))
      rights_delivery = np.zeros((len(self.historical_monthly_deliveries.index), len(self.sorted_rights)))
      #loop through delivery timeseries
      for month_step in range(0, len(self.historical_monthly_deliveries.index)):
        #start with overall structure delivery and distribute it to water rights starting with most senior until there is no delivery left
        #maximum delivery to each right is just the right decree amount
        remaining_delivery = structure_deliveries[month_step] * 1.0
        #also assign the 'consumptive' use at each structure - assign the consumptive use to the most senior right
        if delivery_type == 'deliveries':
          remaining_consumptive = structure_deliveries[month_step] - return_flows[month_step]
        rights_counter = 0
        #loop through the rights, most senior first, and assign deliveries up to the right decree amount until all structure deliveries have been assigned
        for ind_right in self.sorted_rights:
          this_right_delivery = min(max(remaining_delivery, 0.0), self.rights_objects[ind_right].decree_af[self.historical_monthly_deliveries.index[month_step].month - 1])
          if delivery_type == 'deliveries':
            rights_return[month_step, rights_counter] = max(this_right_delivery - remaining_consumptive, 0.0)
            remaining_consumptive -= min(this_right_delivery, remaining_consumptive)
          rights_delivery[month_step, rights_counter] = this_right_delivery * 1.0
          remaining_delivery -=  this_right_delivery
          rights_counter += 1
        
      if self.use_adaptive:
        #also assign deliveries to adaptive uses
        structure_deliveries_adaptive = np.asarray(self.adaptive_monthly_deliveries[delivery_type])
        return_flows_adaptive = np.asarray(self.historical_monthly_deliveries['return'])
        if delivery_type == 'deliveries':
          rights_return_adaptive = np.zeros((len(self.historical_monthly_deliveries.index), len(self.sorted_rights)))
        rights_delivery_adaptive = np.zeros((len(self.adaptive_monthly_deliveries.index), len(self.sorted_rights)))
        for month_step in range(0, len(self.adaptive_monthly_deliveries.index)):
          remaining_delivery = structure_deliveries_adaptive[month_step]
          if delivery_type == 'deliveries':
            remaining_consumptive = structure_deliveries_adaptive[month_step] - return_flows_adaptive[month_step]
          rights_counter = 0
          for ind_right in self.sorted_rights:
            this_right_delivery = min(max(remaining_delivery, 0.0), self.rights_objects[ind_right].decree_af[self.adaptive_monthly_deliveries.index[month_step].month - 1])
            if delivery_type == 'deliveries':
              rights_return_adaptive[month_step, rights_counter] =  max(this_right_delivery - remaining_consumptive, 0.0)
              remaining_consumptive -= min(this_right_delivery, remaining_consumptive)
            rights_delivery_adaptive[month_step, rights_counter] = this_right_delivery * 1.0
            remaining_delivery -= this_right_delivery * 1.0
            rights_counter += 1
    
      counter = 0
      for ind_right in self.sorted_rights:
      #assign right-based deliveries to the right objects
        self.rights_objects[ind_right].historical_monthly_deliveries[delivery_type] = rights_delivery[:,counter]
        if delivery_type == 'deliveries':
          self.rights_objects[ind_right].historical_monthly_deliveries['return'] = rights_return[:,counter]
        if self.use_adaptive:
          self.rights_objects[ind_right].adaptive_monthly_deliveries[delivery_type] = rights_delivery_adaptive[:,counter]
          if delivery_type == 'deliveries':
            self.rights_objects[ind_right].adaptive_monthly_deliveries['return'] = rights_return_adaptive[:,counter]
        else:
          self.rights_objects[ind_right].adaptive_monthly_deliveries[delivery_type] = rights_delivery[:,counter]
          if delivery_type == 'deliveries':
            self.rights_objects[ind_right].adaptive_monthly_deliveries['return'] = rights_return[:,counter]
        counter += 1

  def update_demand_rights(self, date_use):
    structure_demands = self.adaptive_monthly_demand.loc[date_use, 'demand']
    rights_counter = 0
    rights_demands = np.zeros(len(self.sorted_rights))
    for ind_right in self.sorted_rights:
      rights_demands[rights_counter] = min(max(structure_demands, 0.0), self.rights_objects[ind_right].decree_af[date_use.month-1])
      structure_demands -=  min(max(structure_demands, 0.0), self.rights_objects[ind_right].decree_af[date_use.month-1])
      rights_counter += 1
      
    counter = 0
    for ind_right in self.sorted_rights:
      self.rights_objects[ind_right].adaptive_monthly_demand[date_use, 'demand'] = rights_demands[counter]
      counter += 1

  def update_delivery_rights(self, date_use):
    structure_deliveries = self.adaptive_monthly_deliveries.loc[date_use, 'deliveries']
    rights_counter = 0
    rights_demands = np.zeros(len(self.sorted_rights))
    for ind_right in self.sorted_rights:
      rights_demands[rights_counter] = min(max(structure_deliveries, 0.0), self.rights_objects[ind_right].decree_af[date_use.month-1])
      structure_deliveries -=  min(max(structure_deliveries, 0.0), self.rights_objects[ind_right].decree_af[date_use.month-1])
      rights_counter += 1
      
    counter = 0
    for ind_right in self.sorted_rights:
      self.rights_objects[ind_right].adaptive_monthly_demand[date_use, 'demand'] = rights_demands[counter]
      counter += 1
