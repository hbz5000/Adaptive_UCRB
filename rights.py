import numpy as np 
import pandas as pd

class Rights():

  def __init__(self, right_name, right_priority, right_decree, fill_type = 0):
    self.name = right_name
    self.priority = right_priority
    self.decree = right_decree
    days_in_month = [31.0, 28.0, 31.0, 30.0, 31.0, 30.0, 31.0, 31.0, 30.0, 31.0, 30.0, 31.0]
    self.decree_af = np.zeros(len(days_in_month))
    for x in range(0, len(days_in_month)):
      self.decree_af[x] = right_decree * 1.983 * days_in_month[x]
    self.constraining_call = -1
    self.fill_type = fill_type
    
  def initialize_demands(self, timesteps):
    #this function initializes the demand dataframes for each water right (for baseline and adaptive simulation)
    self.historical_monthly_demand = pd.DataFrame(np.zeros(len(timesteps)), index = timesteps, columns = ['demand',])
    self.adaptive_monthly_demand = pd.DataFrame(np.zeros(len(timesteps)), index = timesteps, columns = ['demand',])
  
  def initialize_delivery(self, timesteps):
    #initialize the delivery dataframes for each water right (for both baseline and adaptive simulation)
    self.historical_monthly_deliveries = pd.DataFrame(np.zeros((len(timesteps),3)), index = timesteps, columns = ['deliveries','priority', 'return'])
    self.adaptive_monthly_deliveries = pd.DataFrame(np.zeros((len(timesteps), 3)), index = timesteps, columns = ['deliveries','priority', 'return'])
    self.distance_from_call = np.zeros(len(timesteps))
    self.monthly_snowpack = np.zeros(len(timesteps))
    self.percent_filled = np.zeros(len(timesteps))
    self.percent_filled_single = np.zeros(len(timesteps))
      