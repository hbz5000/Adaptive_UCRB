from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
import collections as cl
import pandas as pd
import json
from rights import Rights

class Structure():
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


  def __init__(self, station_name, basin):
    self.name = station_name
    self.basin = basin
    self.msdr = -1
    self.rights_list = []
    self.rights_objects = {}
    self.downstream_structures = []
    
  def initialize_right(self, right_name, right_priority, right_decree):
    self.rights_list.append(right_name)
    self.rights_objects[right_name] = Rights(right_name, right_priority, right_decree)
  
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
      self.rights_objects[ind_right].initialize_timeseries(len(self.monthly_demand))
      
    for month_step in range(0, len(self.monthly_demand)):
      remaining_demand = self.monthly_demand[month_step] * 1.0
      remaining_delivery = self.monthly_deliveries[month_step] * 1.0
      for ind_right in self.sorted_rights:
        self.rights_objects[ind_right].monthly_demand[month_step] = min(max(remaining_demand, 0.0), self.rights_objects[ind_right].decree_af)
        self.rights_objects[ind_right].monthly_deliveries[month_step] = min(max(remaining_delivery, 0.0), self.rights_objects[ind_right].decree_af)
        remaining_demand -=  min(max(remaining_demand, 0.0), self.rights_objects[ind_right].decree_af)
        remaining_delivery -=  min(max(remaining_delivery, 0.0), self.rights_objects[ind_right].decree_af)

  def find_senior_downstream_call(self, calling_structures, calling_rights):
    most_senior = 99999999999.999
    for ind_structure, ind_right in zip(calling_structures, calling_rights):
      if ind_structure in self.downstream_structures:
        most_senior = min(most_senior, ind_right)
    return most_senior



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



##################################SENSITIVITY ANALYSIS#################################################################
  def set_sensitivity_factors(self, et_factor, acreage_factor, irr_eff_factor, recharge_decline_factor):
    wyt_list = ['W', 'AN', 'BN', 'D', 'C']
    for wyt in wyt_list:
      for i,v in enumerate(self.crop_list):
        self.acreage[wyt][i] = self.acreage[wyt][i]*acreage_factor
        for monthloop in range(0,12):
          self.irrdemand.etM[v][wyt][monthloop] = self.irrdemand.etM[v][wyt][monthloop]*et_factor
    self.seepage = 1.0 + irr_eff_factor
    for recharge_count in range(0, len(self.recharge_decline)):
      self.recharge_decline[recharge_count] = 1.0 - recharge_decline_factor*(1.0 - self.recharge_decline[recharge_count])

      
#####################################################################################################################
##################################DEMAND CALCULATION#################################################################
#####################################################################################################################

  def find_baseline_demands(self,wateryear, non_leap_year, days_in_month):
    self.monthlydemand = {}
    wyt_list = ['W', 'AN', 'BN', 'D', 'C']
    crop_wyt_list = ['AN', 'AN', 'BN', 'D', 'C']
    
    for wyt, cwyt in zip(wyt_list, crop_wyt_list):
      self.monthlydemand[wyt] = np.zeros(12)
      for monthloop in range(0,12):
        self.monthlydemand[wyt][monthloop] += self.urban_profile[monthloop]*self.MDD/days_in_month[non_leap_year][monthloop]
        if self.has_pesticide == 1:
          for i,v in enumerate(self.acreage_by_year):
            self.monthlydemand[wyt][monthloop] += max(self.irrdemand.etM[v][cwyt][monthloop],0.0)*(self.acreage_by_year[v][wateryear]-self.private_acreage[v][wateryear])/(12.0*days_in_month[non_leap_year][monthloop])
            #self.monthlydemand[wyt][monthloop] += max(self.irrdemand.etM[v][cwyt][monthloop] - self.irrdemand.etM['precip'][cwyt][monthloop],0.0)*(self.acreage_by_year[v][wateryear]-self.private_acreage[v][wateryear])/(12.0*days_in_month[non_leap_year][monthloop])
        elif self.has_pmp == 1:
          for crop in self.pmp_acreage:
            self.monthlydemand[wyt][monthloop] += max(self.irrdemand.etM[crop][cwyt][monthloop],0.0)*max(self.pmp_acreage[crop]-self.private_acreage[crop], 0.0)/(12.0*days_in_month[non_leap_year][monthloop])
            #self.monthlydemand[wyt][monthloop] += max(self.irrdemand.etM[crop][cwyt][monthloop] - self.irrdemand.etM['precip'][cwyt][monthloop],0.0)*max(self.pmp_acreage[crop]-self.private_acreage[crop], 0.0)/(12.0*days_in_month[non_leap_year][monthloop])
        else:
          for i,v in enumerate(self.crop_list):
            self.monthlydemand[wyt][monthloop] += max(self.irrdemand.etM[v][cwyt][monthloop],0.0)*(self.acreage[cwyt][i]-self.private_acreage[v])/(12.0*days_in_month[non_leap_year][monthloop])
            #self.monthlydemand[wyt][monthloop] += max(self.irrdemand.etM[v][cwyt][monthloop] - self.irrdemand.etM['precip'][cwyt][monthloop],0.0)*(self.acreage[cwyt][i]-self.private_acreage[v])/(12.0*days_in_month[non_leap_year][monthloop])
          #self.monthlydemand[wyt][monthloop] += max(self.irrdemand.etM[v][wyt][monthloop] ,0.0)*self.acreage[wyt][i]/(12.0*days_in_month[non_leap_year][monthloop])
	  	
