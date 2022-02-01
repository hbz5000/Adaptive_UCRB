from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
import collections as cl
import pandas as pd
import json
from rights import Rights

class Reservoir():
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


  def __init__(self, station_name, station_id, storage_capacity):
    self.name = station_name
    self.idnum = station_id
    self.capacity = storage_capacity
    self.struct_type = 'reservoir'
    self.rights_list = []
    self.rights_objects = {}
    
  def initialize_right(self, right_name, right_priority, right_decree, fill_type):
    self.rights_list.append(right_name)
    self.rights_objects[right_name] = Rights(right_name, right_priority, right_decree, fill_type = fill_type)

  def make_sorted_rights_list(self):
    priority_rank = []
    for right_name in self.rights_list:
      priority_rank.append(self.rights_objects[right_name].priority)
    priority_order = np.argsort(priority_rank)
    self.sorted_rights = []
    for priority_counter in range(0, len(priority_order)):
      self.sorted_rights.append(self.rights_list[priority_order[priority_counter]])

  def assign_demand_rights(self):
    for ind_right in self.sorted_rights:
      self.rights_objects[ind_right].initialize_timeseries(self.historical_monthly_demand.index)
    structure_demands = np.asarray(self.historical_monthly_demand['demand'])
    rights_demands = np.zeros((len(self.historical_monthly_demand.index), len(self.sorted_rights)))
    for month_step in range(0, len(self.historical_monthly_demand.index)):
      remaining_demand = structure_demands[month_step]
      rights_counter = 0
      for ind_right in self.sorted_rights:
        rights_demands[month_step, rights_counter] = min(max(remaining_demand, 0.0), self.rights_objects[ind_right].decree_af)
        remaining_demand -=  min(max(remaining_demand, 0.0), self.rights_objects[ind_right].decree_af)
        rights_counter += 1
    counter = 0
    for ind_right in self.sorted_rights:
      self.rights_objects[ind_right].historical_monthly_demand['demand'] = rights_demands[:,counter]
      counter += 1
      
  def assign_delivery_rights(self):
    structure_deliveries = np.asarray(self.historical_monthly_deliveries['deliveries'])
    rights_delivery = np.zeros((len(self.historical_monthly_deliveries.index), len(self.sorted_rights)))
    for month_step in range(0, len(self.historical_monthly_deliveries.index)):
      remaining_delivery = structure_deliveries[month_step]
      rights_counter = 0
      for ind_right in self.sorted_rights:
        rights_delivery[month_step, rights_counter] = min(max(remaining_delivery, 0.0), self.rights_objects[ind_right].decree_af)
        remaining_delivery -=  min(max(remaining_delivery, 0.0), self.rights_objects[ind_right].decree_af)
        rights_counter += 1
    counter = 0
    for ind_right in self.sorted_rights:
      self.rights_objects[ind_right].historical_monthly_deliveries['deliveries'] = rights_delivery[:,counter]
      counter += 1

  def update_demand_rights(self, date_use):
    structure_demands = self.adaptive_monthly_demand.loc[date_use, 'demand']
    rights_counter = 0
    rights_demands = np.zeros(len(self.sorted_rights))
    for ind_right in self.sorted_rights:
      rights_demands[rights_counter] = min(max(structure_demands, 0.0), self.rights_objects[ind_right].decree_af)
      structure_demands -=  min(max(structure_demands, 0.0), self.rights_objects[ind_right].decree_af)
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
      rights_demands[rights_counter] = min(max(structure_deliveries, 0.0), self.rights_objects[ind_right].decree_af)
      structure_deliveries -=  min(max(structure_deliveries, 0.0), self.rights_objects[ind_right].decree_af)
      rights_counter += 1
      
    counter = 0
    for ind_right in self.sorted_rights:
      self.rights_objects[ind_right].adaptive_monthly_demand[date_use, 'demand'] = rights_demands[counter]
      counter += 1
