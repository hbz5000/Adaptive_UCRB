import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from plotter import Plotter
from basin import Basin
import crss_reader as crss


input_data_dictionary = crss.create_input_data_dictionary('B', 'A')
#################################################################
######################FIGURE 3###################################
#################################################################
#crop values
crop_types = Plotter('figure_3.png',figsize = (20,12))
crop_types.plot_crop_types(input_data_dictionary['irrigation'])
del crop_types

#################################################################
######################FIGURE 6###################################
#################################################################
#leasing stages/volumes/net benefits
data_figure = Plotter('figure_6.png', figsize = (20,12))
data_figure.plot_trigger_stages()
del data_figure

#################################################################
######################FIGURE 9###################################
#################################################################
#average lease volumes
for ani_plot in range(0, 6):
  third_party_impacts = Plotter('figure_5_animated_all_' + str(ani_plot) + '.png', figsize = (28, 12))
  third_party_impacts.plot_third_party_impacts(['550', '600', '650', '700'], years_leased, ani_plot)
  del third_party_impacts

#################################################################
######################FIGURE 10###################################
#################################################################
#average transaction costs
scenario_list = ['550', '600', '650', '700']
for ani_plot in range(0, 6):
  informal_lease_frequency = Plotter('figure_10.png', figsize = (28, 12))
  years_leased = informal_lease_frequency.plot_informal_leases(scenario_list, ani_plot)
  del informal_lease_frequency

#################################################################
######################FIGURE S3###################################
#################################################################
#option prices by structure
data_figure = Plotter('figure_S3.png', figsize = (16, 12), nr = 2)
data_figure.plot_option_payments(['700', '650', '600', '550'])
del data_figure
