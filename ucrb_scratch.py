def plot_exposures_all():
    read_from_file = True
    if read_from_file:
      #self.plot_all_rights(read_from_file)
      exposure_base, premium_base, demand_base = self.plot_exposure(end_year - start_year - 1, 'all', -0.1, read_from_file)
      exposure_gradual_1, premium_gradual_1, demand_gradual_1 = self.plot_exposure(end_year - start_year - 1, 0.041616, -0.1, read_from_file)
      exposure_gradual_2, premium_gradual_2, demand_gradual_2 = self.plot_exposure(end_year - start_year - 1, 0.0833, -0.1, read_from_file)
      exposure_gradual_3, premium_gradual_3, demand_gradual_3 = self.plot_exposure(end_year - start_year - 1, 0.16, -0.1, read_from_file)
      exposure_limited_1, premium_limited_1, demand_limited_1 = self.plot_exposure(end_year - start_year - 1, 'all', -0.1, read_from_file)
      exposure_limited_2, premium_limited_2, demand_limited_2 = self.plot_exposure(end_year - start_year - 1, 'all', 0.1, read_from_file)
      exposure_limited_3, premium_limited_3, demand_limited_3 = self.plot_exposure(end_year - start_year - 1, 'all', 0.3, read_from_file)
      exposure_limited_4, premium_limited_4, demand_limited_4 = self.plot_exposure(end_year - start_year - 1, 'all', 0.5, read_from_file)
      exposure_limited_5, premium_limited_5, demand_limited_5 = self.plot_exposure(end_year - start_year - 1, 'all', 0.7, read_from_file)
    else:
      self.plot_demand_index(start_year, end_year)
      self.plot_all_rights(read_from_file)
      
      
    #self.plot_forecast_index(end_year - start_year - 1)  
    current_val = 0.0
    current_index = -1
    while current_val == 0.0:
      current_index += 1
      current_val = exposure_base['irrigation']['0'][current_index]
  
      
    fig, ax = plt.subplots(3)
    exposure_colors = sns.color_palette('gnuplot_r', 6)
    for mn in range(0, 6):
      exposure_vals = np.zeros(len(exposure_base['transbasin'][str(mn)]))
      total_demand = np.zeros(len(demand_base['transbasin'][str(mn)]))
      premium_timeseries = np.zeros(len(premium_base['transbasin'][str(mn)]))
      for right_type in ['transbasin', 'mi', 'env', 'irrigation']:
        exposure_vals += exposure_base[right_type][str(mn)]
        premium_timeseries += premium_base[right_type][str(mn)]
        total_demand += demand_base[right_type][str(mn)]
      pos = np.linspace(np.min(exposure_vals[current_index:]), np.max(exposure_vals[current_index:]), 101)
      kde_vals = stats.gaussian_kde(exposure_vals[current_index:])
      norm_mult = np.max(kde_vals(pos))
      
      
      ax[0].fill_between(pos, np.zeros(len(pos)), kde_vals(pos)/norm_mult, edgecolor = 'black', alpha = 0.6, facecolor = exposure_colors[mn])
      ax[1].plot(np.arange(start_year + current_index, end_year - 1), premium_timeseries[current_index:], color = exposure_colors[mn], linewidth = 3.0)
      ax[2].plot(np.arange(start_year + current_index, end_year - 1), total_demand[current_index:], color = exposure_colors[mn], linewidth = 3.0)
    plt.savefig('UCRB_analysis-master/index_figs/exposure_1.png', dpi = 150, bbox_inches = 'tight', pad_inches = 0.0)
    plt.show()
    plt.close()

    fig, ax = plt.subplots(3)
    exposure_colors = sns.color_palette('gnuplot_r', 6)
    for mn in range(0, 6):
      exposure_vals = np.zeros(len(exposure_base['transbasin'][str(mn)]))
      total_demand = np.zeros(len(demand_base['transbasin'][str(mn)]))
      premium_timeseries = np.zeros(len(premium_base['transbasin'][str(mn)]))
      right_type_counter = 1.0
      for right_type in ['transbasin', 'mi', 'irrigation']:
        exposure_vals += exposure_base[right_type][str(mn)]
        premium_timeseries = premium_base[right_type][str(mn)]
        total_demand = demand_base[right_type][str(mn)]
        if len(np.unique(exposure_vals)) > 1:
          ax[1].plot(np.arange(start_year + current_index, end_year - 1), premium_timeseries[current_index:], color = exposure_colors[mn], linewidth = 3.0, alpha = right_type_counter/3.0)
          ax[2].plot(np.arange(start_year + current_index, end_year - 1), total_demand[current_index:], color = exposure_colors[mn], linewidth = 3.0, alpha = right_type_counter/3.0)
        right_type_counter += 1.0

      pos = np.linspace(np.min(exposure_vals[current_index:]), np.max(exposure_vals[current_index:]), 101)
      kde_vals = stats.gaussian_kde(exposure_vals[current_index:])
      norm_mult = np.max(kde_vals(pos))
      ax[0].fill_between(pos, np.zeros(len(pos)), kde_vals(pos)/norm_mult, edgecolor = 'black', facecolor = exposure_colors[mn])
    plt.savefig('UCRB_analysis-master/index_figs/exposure_2.png', dpi = 150, bbox_inches = 'tight', pad_inches = 0.0)
    plt.show()
    plt.close()



    fig, ax = plt.subplots(3)
    exposure_colors = sns.color_palette('gnuplot_r', 4)
    exposure_cnt = 0
    legend_elements = []
    maxx = 0
    for right_type, label_name in zip(['transbasin', 'mi', 'env', 'irrigation'], ['transbasin', 'M&I', 'ENV', 'irrigation']):
      exposure_vals = exposure_base[right_type]['2']/1000000.0
      premium_timeseries = premium_base[right_type]['2']/1000000.0
      total_demand = demand_base[right_type]['2']/1000000.0
      pos = np.linspace(np.min(exposure_vals[current_index:]), np.max(exposure_vals[current_index:]), 101)
      if len(np.unique(exposure_vals)) > 1:
        kde_vals = stats.gaussian_kde(exposure_vals[current_index:])
        norm_mult = np.max(kde_vals(pos))
        ax[0].fill_between(pos, np.zeros(len(pos)), kde_vals(pos)/norm_mult, edgecolor = 'black', alpha = 0.6, facecolor = exposure_colors[exposure_cnt])
        ax[1].plot(np.arange(start_year + current_index, end_year - 1), premium_timeseries[current_index:], color = exposure_colors[exposure_cnt], linewidth = 3.0)
        ax[2].plot(np.arange(start_year + current_index, end_year - 1), total_demand[current_index:], color = exposure_colors[exposure_cnt], linewidth = 3.0)
        legend_elements.append(Line2D([0], [0], color=exposure_colors[exposure_cnt], lw = 2, label=label_name))
        maxx = max(maxx, np.max(exposure_vals))

      exposure_cnt += 1
      
    ax[0].set_xlim([0, maxx])
    ax[1].set_xlim([1980, 2011])
    ax[2].set_xlim([1980, 2011])
    ax[0].set_yticks([])
    ax[0].set_yticklabels('')
    legend_location = 'upper right'
    ax[0].legend(handles=legend_elements, loc=legend_location, prop={'family':'Gill Sans MT','weight':'bold','size':8}, framealpha = 1.0)
    ax[0].set_ylabel('Annual\nExposure\nPDF ($MM)')
    ax[1].set_ylabel('Annual\nPremiums\n($MM)')
    ax[2].set_ylabel('Annual\nDemand\n(MAF)')
    
    ax[0].set_yticks([])
    ax[0].set_yticklabels('')

    plt.savefig('UCRB_analysis-master/index_figs/exposure_3.png', dpi = 150, bbox_inches = 'tight', pad_inches = 0.0)
    plt.show()
    plt.close()

    fig, ax = plt.subplots(3)
    exposure_colors = sns.color_palette('gnuplot_r', 3)
    exposure_cnt = 0
    for exposure_dict, premium_dict, demand_dict in zip([exposure_gradual_1, exposure_gradual_2, exposure_gradual_3], [premium_gradual_1, premium_gradual_2, premium_gradual_3], [demand_gradual_1, demand_gradual_2, demand_gradual_3]):
      total_gradual = np.zeros(len(exposure_dict['transbasin']['2']))
      total_premium = np.zeros(len(premium_dict['transbasin']['2']))
      total_demand = np.zeros(len(demand_dict['transbasin']['2']))
      for mn in range(0, 6):
        for right_type in ['transbasin', 'mi', 'env', 'irrigation']:
          total_gradual += exposure_dict[right_type][str(mn)]
          total_premium += premium_dict[right_type][str(mn)]
          total_demand += demand_dict[right_type][str(mn)]
      pos = np.linspace(np.min(total_gradual[current_index:]), np.max(total_gradual[current_index:]), 101)
      kde_vals = stats.gaussian_kde(total_gradual[current_index:])
      ax[0].fill_between(pos, np.zeros(len(pos)), kde_vals(pos), edgecolor = 'black', alpha = 0.6, facecolor = exposure_colors[exposure_cnt])
      ax[1].plot(np.arange(start_year + current_index, end_year - 1), total_premium[current_index:], color = exposure_colors[exposure_cnt], linewidth = 3.0)
      ax[2].plot(np.arange(start_year + current_index, end_year - 1), total_demand[current_index:], color = exposure_colors[exposure_cnt], linewidth = 3.0)
      exposure_cnt += 1      
    plt.savefig('UCRB_analysis-master/index_figs/exposure_4.png', dpi = 150, bbox_inches = 'tight', pad_inches = 0.0)
    plt.show()
    plt.close()

    fig, ax = plt.subplots(3)
    exposure_colors = sns.color_palette('gnuplot_r', 5)
    exposure_cnt = 0
    legend_elements = []
    for exposure_dict, premium_dict, demand_dict, label_name in zip([exposure_limited_1, exposure_limited_2, exposure_limited_3, exposure_limited_4, exposure_limited_5], [premium_limited_1, premium_limited_2, premium_limited_3, premium_limited_4, premium_limited_5], [demand_limited_1, demand_limited_2, demand_limited_3, demand_limited_4, demand_limited_5], ['All', '>10% Filled', '>30% filled', '>50% filled', '>70% filled']):
      total_gradual = np.zeros(len(exposure_dict['transbasin']['0']))
      total_premium = np.zeros(len(premium_dict['transbasin']['0']))
      total_demand = np.zeros(len(demand_dict['transbasin']['0']))
      for right_type in ['transbasin', 'mi', 'env', 'irrigation']:
        total_gradual += exposure_dict[right_type]['0']/1000000.0
        total_premium += premium_dict[right_type]['0']/1000000.0
        total_demand += demand_dict[right_type]['0']/1000000.0
      pos = np.linspace(np.min(total_gradual[current_index:]), np.max(total_gradual[current_index:]), 101)
      kde_vals = stats.gaussian_kde(total_gradual[current_index:])
      ax[0].fill_between(pos, np.zeros(len(pos)), kde_vals(pos), edgecolor = 'black', alpha = 0.6, facecolor = exposure_colors[exposure_cnt])
      ax[1].plot(np.arange(start_year + current_index, end_year - 1), total_premium[current_index:], color = exposure_colors[exposure_cnt], linewidth = 3.0)
      ax[2].plot(np.arange(start_year + current_index, end_year - 1), total_demand[current_index:], color = exposure_colors[exposure_cnt], linewidth = 3.0)
      legend_elements.append(Line2D([0], [0], color=exposure_colors[exposure_cnt], lw = 2, label=label_name))
      if exposure_cnt == 0:
        ax[0].set_xlim([0, np.max(total_gradual)])
      
      exposure_cnt += 1
    ax[1].set_xlim([1980, 2011])
    ax[2].set_xlim([1980, 2011])
    ax[0].set_yticks([])
    ax[0].set_yticklabels('')
    legend_location = 'upper right'
    ax[0].legend(handles=legend_elements, loc=legend_location, prop={'family':'Gill Sans MT','weight':'bold','size':8}, framealpha = 1.0)
    ax[0].set_ylabel('Annual\nExposure\nPDF ($MM)')
    ax[1].set_ylabel('Annual\nPremiums\n($MM)')
    ax[2].set_ylabel('Annual\nDemand\n(MAF)')
    
    ax[0].set_yticks([])
    ax[0].set_yticklabels('')
    plt.savefig('UCRB_analysis-master/index_figs/exposure_5.png', dpi = 150, bbox_inches = 'tight', pad_inches = 0.0)
    plt.show()
    plt.close()





structures_ucrb = gpd.read_file(project_folder + '/Shapefiles_UCRB/Div_5_structures.shp')
for x in structures_ucrb:
  print(x)
print(list(set(structures_ucrb['WDID'])))

all_structures_plot = list(set(structures_ucrb['WDID']))

start_year = 1908
end_year = 2013
control_call = ucrb.read_ind_structure_deliveries(input_data_dictionary, start_year, end_year)
area_ucrb = gpd.read_file(project_folder + '/Shapefiles_UCRB/DIV3CO.shp')
area_ucrb = area_ucrb[area_ucrb['DIV'] == 5]
districts_ucrb = gpd.read_file(project_folder + '/Shapefiles_UCRB/Water_Districts.shp')
ditches_ucrb = gpd.read_file(project_folder + '/Shapefiles_UCRB/Div5_Irrigated_Lands_2015/Div5_2015_Ditches.shp')
irrigation_ucrb = gpd.read_file(project_folder + '/Shapefiles_UCRB/Div5_Irrigated_Lands_2015/Div5_Irrig_2015.shp')
streams_ucrb = gpd.read_file(project_folder + '/Shapefiles_UCRB/UCRBstreams.shp')
flowlines_ucrb = gpd.read_file(project_folder + 'Shapefiles_UCRB/flowline.shp')
area_ucrb = area_ucrb.to_crs(epsg = 3857)
districts_ucrb = districts_ucrb.to_crs(epsg = 3857)
ditches_ucrb = ditches_ucrb.to_crs(epsg = 3857)
irrigation_ucrb = irrigation_ucrb.to_crs(epsg = 3857)
streams_ucrb = streams_ucrb.to_crs(epsg = 3857)
flowlines_ucrb = flowlines_ucrb.to_crs(epsg = 3857)
perennial_crops = irrigation_ucrb[irrigation_ucrb['PERENNIAL'] == 'YES']
streams_ucrb = gpd.sjoin(streams_ucrb, area_ucrb, how = 'inner', op = 'intersects')
adams_tunnel = '5104634'
granby_res = '5104055'

for month_name in ['OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP']:
  call_loc_list = control_call[month_name]
  geom_x = []
  geom_y = []
  total_freq = {}
  total_freq['frequency'] = []
  for st_id in call_loc_list:
    if st_id[0:7] in all_structures_plot:
      this_struct = structures_ucrb[structures_ucrb['WDID'] == st_id[0:7]]
      total_freq['frequency'].append(float(call_loc_list[st_id]))
      geom_x.append(this_struct['xcoord'])
      geom_y.append(this_struct['ycoord'])
  
  geometry = [Point(xy) for xy in zip(geom_x, geom_y)]
  freq_call = pd.DataFrame(total_freq)
  freq_call_gdf = gpd.GeoDataFrame(freq_call, geometry = geometry, crs = 'EPSG:4326')
  freq_call_gdf = freq_call_gdf.to_crs(epsg = 3857)
  print(freq_call_gdf)
  ana_struct = structures_ucrb[structures_ucrb['WDID'] == '5104634']
  ana_struct = gpd.GeoDataFrame(ana_struct, geometry = ana_struct['geometry'], crs = structures_ucrb.crs)
  ana_struct = ana_struct.to_crs(epsg = 3857)
  data_figure = Mapper()
  projection_string = 'EPSG:3857'#project raster to data projection
  ##raster file names
  raster_name_pt1 = 'LC08_L1TP_'
  raster_name_pt2 = '_02_T1'
  raster_band_list = ['_B4', '_B3', '_B2']
  raster_id_list = {}
  raster_id_list['034033'] = ['20200702_20200913',]
  raster_id_list['034032'] = ['20200702_20200913',]
  raster_id_list['036032'] = ['20200817_20200920',]
  raster_id_list['036033'] = ['20200817_20200920',]
  raster_id_list['035032'] = ['20200709_20200912',]
  raster_id_list['035033'] = ['20200709_20200912',]

  #data_figure.load_batch_raster(project_folder + '/stitched_satellite/', raster_id_list, raster_name_pt1, raster_name_pt2, raster_band_list, projection_string, max_bright = (100.0, 20000.0), use_gamma = 0.8)
  data_figure.plot_scale(area_ucrb, 'depth', type = 'polygons', solid_color = 'none', solid_alpha = 0.5, linewidth_size = 2.0, outline_color = 'black')
  data_figure.plot_scale(streams_ucrb, 'depth', type = 'polygons', solid_color = 'steelblue', solid_alpha = 1.0, linewidth_size = 1.0, outline_color = 'steelblue')
  data_figure.plot_scale(irrigation_ucrb, 'depth', type = 'polygons', solid_color = 'forestgreen', solid_alpha = 1.0, linewidth_size = 0.2, outline_color = 'forestgreen')
  #ata_figure.plot_scale(ditches_ucrb, 'depth', type = 'points', solid_color = 'black', solid_alpha = 0.5, linewidth_size = 0.0, outline_color = 'black', markersize = 2)
  data_figure.plot_scale(freq_call_gdf, 'frequency', type = 'points', solid_color = 'scaled', colorscale = 'RdYlBu', markersize = 75, solid_alpha = 1.0, linewidth_size = 0.2, outline_color = 'black', value_lim = (0.0, 1.0), zorder = 3)
  data_figure.plot_scale(ana_struct, 'frequency', type = 'points', solid_color = 'black', markersize = 75, solid_alpha = 1.0, linewidth_size = 0.2, outline_color = 'black', value_lim = (0.0, 1.0), zorder = 3)
  legend_location = 'upper left'
  legend_element = [Patch(facecolor='forestgreen', edgecolor='black', label='Field Crops'), 
                  Line2D([0], [0], markerfacecolor='steelblue', markeredgecolor='black',  lw = 2, label='River'),
                  Line2D([0], [0], markerfacecolor='black', markeredgecolor='black',  lw = 0, marker = 'o', markersize = 10, label='Transboundary Diversion Station'), 
                  Line2D([0], [0], markerfacecolor='blue', markeredgecolor='blue',  lw = 0, marker = 'o', markersize = 10, label='Call Location (100% of Years)'), 
                  Line2D([0], [0], markerfacecolor='yellow', markeredgecolor='yellow',  lw = 0, marker = 'o', markersize = 10, label='Call Location (50% of Years)'), 
                  Line2D([0], [0], markerfacecolor='red', markeredgecolor='red',  lw = 0, marker = 'o', markersize = 10, label='Call Location (0% of Years)')]
  legend_properties = {'family':'Gill Sans MT','weight':'bold','size':8}
  xrange = area_ucrb.total_bounds[2] - area_ucrb.total_bounds[0]
  yrange = area_ucrb.total_bounds[3] - area_ucrb.total_bounds[1] 
  data_figure.format_plot(xlim = (area_ucrb.total_bounds[0] - xrange*0.025, area_ucrb.total_bounds[2] + xrange*0.025), ylim = (area_ucrb.total_bounds[1]  -yrange*0.025, area_ucrb.total_bounds[3] + yrange*0.025))
  data_figure.add_legend(legend_location, legend_element, legend_properties)

  plt.savefig(project_folder + 'Shapefiles_UCRB/Calls_' + month_name +  '.png', dpi = 300, bbox_inches = 'tight', pad_inches = 0.0)
  plt.close()
  del data_figure
structures_ucrb = gpd.read_file('BraysBayou/Shapefiles_UCRB/Div_5_structures.shp')
flowlines_ucrb = gpd.read_file('BraysBayou/Shapefiles_UCRB/flowline.shp')
area_ucrb = area_ucrb.to_crs(epsg = 3857)
districts_ucrb = districts_ucrb.to_crs(epsg = 3857)
ditches_ucrb = ditches_ucrb.to_crs(epsg = 3857)
irrigation_ucrb = irrigation_ucrb.to_crs(epsg = 3857)
streams_ucrb = streams_ucrb.to_crs(epsg = 3857)
structures_ucrb = structures_ucrb.to_crs(epsg = 3857)
flowlines_ucrb = flowlines_ucrb.to_crs(epsg = 3857)
print(list(set(structures_ucrb['StructType'])))
print(list(set(structures_ucrb['FeatureTyp'])))
print(list(set(structures_ucrb['CurrInUse'])))
for x in irrigation_ucrb:
  print(x)
  print(irrigation_ucrb[x])
crop_list = list(set(irrigation_ucrb['CROP_TYPE']))
print(crop_list)
plot_type = 'fig1'
perennial_crops = irrigation_ucrb[irrigation_ucrb['PERENNIAL'] == 'YES']
streams_ucrb = gpd.sjoin(streams_ucrb, area_ucrb, how = 'inner', op = 'intersects')
data_figure.plot_scale(area_ucrb, 'depth', type = 'polygons', solid_color = 'none', solid_alpha = 0.5, linewidth_size = 2.0, outline_color = 'black')
data_figure.plot_scale(streams_ucrb, 'depth', type = 'polygons', solid_color = 'steelblue', solid_alpha = 1.0, linewidth_size = 1.0, outline_color = 'steelblue')
data_figure.plot_scale(irrigation_ucrb, 'depth', type = 'polygons', solid_color = 'forestgreen', solid_alpha = 1.0, linewidth_size = 0.2, outline_color = 'forestgreen')
data_figure.plot_scale(ditches_ucrb, 'depth', type = 'points', solid_color = 'black', solid_alpha = 0.5, linewidth_size = 0.0, outline_color = 'black', markersize = 2)
if plot_type == 'fig1':
  data_figure.plot_scale(transboundary_stations_gdf, 'depth', type = 'points', solid_color = 'indianred', solid_alpha = 1.0, linewidth_size = 0.0, outline_color = 'black', markersize = 15)
  data_figure.plot_scale(snow_stations_gdf, 'depth', type = 'points', solid_color = 'navy', solid_alpha = 1.0, linewidth_size = 0.0, outline_color = 'black', markersize = 15)
  data_figure.plot_scale(perennial_crops, 'depth', type = 'polygons', solid_color = 'goldenrod', solid_alpha = 1.0, linewidth_size = 0.2, outline_color = 'goldenrod')
if plot_type == 'fig2':
  data_figure.plot_scale(snow_stations_gdf, 'depth', type = 'points', solid_color = 'navy', solid_alpha = 1.0, linewidth_size = 0.0, outline_color = 'black', markersize = 15)
  data_figure.plot_scale(perennial_crops, 'depth', type = 'polygons', solid_color = 'goldenrod', solid_alpha = 1.0, linewidth_size = 0.2, outline_color = 'goldenrod')
if plot_type == 'fig3':
  data_figure.plot_scale(snow_stations_gdf, 'depth', type = 'points', solid_color = 'navy', solid_alpha = 1.0, linewidth_size = 0.0, outline_color = 'black', markersize = 15)

#plot colorbar w/perimeter
xrange = area_ucrb.total_bounds[2] - area_ucrb.total_bounds[0]
yrange = area_ucrb.total_bounds[3] - area_ucrb.total_bounds[1] 
data_figure.format_plot(xlim = (area_ucrb.total_bounds[0] - xrange*0.025, area_ucrb.total_bounds[2] + xrange*0.025), ylim = (area_ucrb.total_bounds[1]  -yrange*0.025, area_ucrb.total_bounds[3] + yrange*0.025))
legend_location = 'upper left'
if plot_type == 'fig1':
  legend_element = [Patch(facecolor='forestgreen', edgecolor='black', label='Field Crops'), 
                  Patch(facecolor='goldenrod', edgecolor='black', label='Perennial Crops'),
                  Line2D([0], [0], markerfacecolor='steelblue', markeredgecolor='black',  lw = 2, label='River'),
                  Line2D([0], [0], markerfacecolor='black', markeredgecolor='black',  lw = 2, marker = 'o', markersize = 10, label='Irrigation Diversion'),
                  Line2D([0], [0], markerfacecolor='navy', markeredgecolor='black',  lw = 2, marker = 'o', markersize = 10, label='Snowpack Station'),
                  Line2D([0], [0], markerfacecolor='indianred', markeredgecolor='black',  lw = 2, marker = 'o', markersize = 10, label='Transboundary Diversion')]
if plot_type == 'fig2':
  legend_element = [Patch(facecolor='forestgreen', edgecolor='black', label='Field Crops'), 
                  Patch(facecolor='goldenrod', edgecolor='black', label='Perennial Crops'),
                  Line2D([0], [0], markerfacecolor='steelblue', markeredgecolor='black', lw = 2, label='River'),
                  Line2D([0], [0], markerfacecolor='black', markeredgecolor='black', lw = 2, marker = 'o', markersize = 10, label='Irrigation Diversion'),
                  Line2D([0], [0], markerfacecolor='navy', markeredgecolor='black', lw = 2, marker = 'o', markersize = 10, label='Snowpack Station')]
if plot_type == 'fig3':
  legend_element = [Patch(facecolor='forestgreen', edgecolor='black', label='Irrigation'),
                  Line2D([0], [0], markerfacecolor='steelblue', markeredgecolor='black', lw = 2, label='River'),
                  Line2D([0], [0], markerfacecolor='black', markeredgecolor='black', lw = 2, marker = 'o', markersize = 10, label='Irrigation Diversion'),
                  Line2D([0], [0], markerfacecolor='navy', markeredgecolor='black', lw = 2, marker = 'o', markersize = 10, label='Snowpack Station')]
if plot_type == 'fig4':
  legend_element = [Patch(facecolor='forestgreen', edgecolor='black', label='Irrigation'),
                  Line2D([0], [0], markerfacecolor='steelblue', markeredgecolor='black', lw = 2, label='River'),
                  Line2D([0], [0], markerfacecolor='black', markeredgecolor='black', lw = 2, marker = 'o', markersize = 10, label='Irrigation Diversion')]
                  
legend_properties = {'family':'Gill Sans MT','weight':'bold','size':8}
data_figure.add_legend(legend_location, legend_element, legend_properties)
plt.savefig('BraysBayou/Shapefiles_UCRB/' + plot_type + '.png', dpi = 300, bbox_inches = 'tight', pad_inches = 0.0)




rights_data = ucrb.read_text_file(input_data_dictionary['structure_rights'])
downstream_data = ucrb.read_text_file(input_data_dictionary['downstream'])
ucrb.read_rights_data(rights_data)
ucrb.read_downstream_structure(downstream_data)
ucrb.create_rights_stack()

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

for index, row in ucrb.basin_huc8.iterrows():
  watershed_name = row['HUC8']

nhd_database_filename = 'UCRB_analysis-master/Shapefiles_UCRB/NHDPLUS_H_1401_HU4_GDB.gdb'
structures_ucrb = gpd.read_file('UCRB_analysis-master/Shapefiles_UCRB/Div_5_structures.shp')

extended_table = gpd.read_file(nhd_database_filename, layer = 'WBDHU4')
ucrb = extended_table[extended_table['HUC4'] == '1401']
extended_table8 = gpd.read_file(nhd_database_filename, layer = 'WBDHU8')
ucrb_huc8 = gpd.sjoin(extended_table8, ucrb, how = 'inner', op = 'within')
basin_areas = {}
for index, row in ucrb_huc8.iterrows():
  watershed_name = row['HUC8']
  basin_snowpack = pd.read_csv('UCRB_analysis-master/Sobol_sample/Snow_Data/Basin_' + watershed_name + '/basinwide_snowpack.csv')
  this_watershed = extended_table8[extended_table8['HUC8'] == watershed_name]
  this_watershed_structures = gpd.sjoin(structures_ucrb, this_watershed, how = 'inner', op = 'within')
  basin_areas[watershed_name] = row['AreaAcres_right']
huc_8_watersheds = ['14010001', '14010002', '14010003', '14010004', '14010005']
self.basin_huc8
nhd_database_filename = 'UCRB_analysis-master/Shapefiles_UCRB/NHDPLUS_H_1401_HU4_GDB.gdb'
structures_ucrb = gpd.read_file('UCRB_analysis-master/Shapefiles_UCRB/Div_5_structures.shp')
structures_ucrb = structures_ucrb.to_crs(epsg = 4326)
extended_table = gpd.read_file(nhd_database_filename, layer = 'WBDHU4')
extended_table = extended_table.to_crs(epsg = 4326)
ucrb = extended_table[extended_table['HUC4'] == '1401']
extended_table8 = gpd.read_file(nhd_database_filename, layer = 'WBDHU8')
#extended_table8 = extended_table8.to_crs(epsg = 4326)
fig, ax = plt.subplots()
extended_table8.plot(ax = ax, facecolor = 'steelblue', edgecolor = 'black', alpha = 0.7)
ucrb.plot(ax = ax, facecolor = 'indianred', edgecolor = 'black', alpha = 0.4)
plt.show()
ucrb_huc8 = gpd.sjoin(extended_table8, ucrb, how = 'inner', op = 'intersects')
structures_list_huc8 = {}
for index, row in ucrb_huc8.iterrows():
  watershed_name = row['HUC8']
  basin_snowpack = pd.read_csv('UCRB_analysis-master/Sobol_sample/Snow_Data/Basin_' + watershed_name + '/basinwide_snowpack.csv')
  this_watershed = extended_table8[extended_table8['HUC8'] == watershed_name]
  this_watershed_structures = gpd.sjoin(structures_ucrb, this_watershed, how = 'inner', op = 'within')
  structures_list_huc8[watershed_name] = []
  for index_s, row_s in this_watershed_structures.iterrows():
    structures_list_huc8[watershed_name].append(row_s['WDID'])


aggregate_structures = {}
aggregate_structures['14010001'] = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '020', '021', '022', '023', '024', '025', '026', '027', '028', '032'] 
aggregate_structures['14010002'] = ['017', '018', '019']
aggregate_structures['14010003'] = ['029', '030', '031']
aggregate_structures['14010004'] = ['033', '034', '035', '036', '037', '038', '039', '040']
aggregate_structures['14010005'] = ['041', '042', '043', '044', '045', '046', '047', '048', '049', '050', '051', '052', '053', '054', '055', '056', '057', '058', '059', '060', '061', '062', '063', '064', '065']
 
demand_filename = 'UCRB_analysis-master\Sobol_sample\Experiment_files\cm2015B.ddm'
with open(demand_filename,'r') as f:
    all_split_data_DDM = [x for x in f.readlines()]       
f.close()

demand_cols = ['structure',]
for x in range(1, 13):
  demand_cols.append('demand ' + str(x))
demand_list = pd.DataFrame(columns = demand_cols)
toggle_on = 0
for j in range(0, len(all_split_data_DDM)):
  if all_split_data_DDM[j][0] == '#':
    toggle_on = 1
  elif toggle_on == 1:
    first_line = int(j * 1)
    toggle_on = 0    
  else:
    this_row = all_split_data_DDM[j].split('.')
    row_data = []
    row_data.extend(this_row[0].split())
    if row_data[0] == '2013':
      demand_list.loc[j - first_line, 'structure'] = row_data[1].strip()
      demand_list.loc[j - first_line, 'demand ' + str(10)] = float(row_data[2].strip())
      for x in range(1, 12):
        month_no = x + 10
        if month_no > 12:
          month_no -= 12
        demand_list.loc[j - first_line, 'demand ' + str(month_no)] = float(this_row[x].strip())
# For DDR
# get unsplit data to rewrite everything that's unchanged
diversion_filename = 'UCRB_analysis-master\Sobol_sample\Experiment_files\cm2015B.ddr'
with open(diversion_filename,'r') as f:
  all_data_DDR = [x for x in f.readlines()]       
f.close() 
column_lengths=[12,24,12,16,8,8]
split_line = ['']*len(column_lengths)
character_breaks=np.zeros(len(column_lengths),dtype=int)
character_breaks[0]=column_lengths[0]
for i in range(1,len(column_lengths)):
  character_breaks[i]=character_breaks[i-1]+column_lengths[i]

diversion_cols = ['structure', 'priority', 'cfs', 'watershed',]
for x in range(1, 13):
  diversion_cols.append('demand ' + str(x))
  diversion_cols.append('prob ' + str(x))

copy_all_data = np.copy(all_data_DDR)
# Change only for specific samples
  # Change only the specific lines
  
rights_list = pd.DataFrame(columns = diversion_cols)
for j in range(0,len(all_data_DDR)):
  if all_data_DDR[j][0] == '#':
    first_line = int(j * 1)
  else:
    split_line[0]=all_data_DDR[j][0:character_breaks[0]]
    for i in range(1,len(split_line)):
      split_line[i]=all_data_DDR[j][character_breaks[i-1]:character_breaks[i]]
    rights_list.loc[j - first_line, 'structure'] = split_line[2].strip()
    rights_list.loc[j - first_line, 'priority'] = float(split_line[3].strip())
    if int(split_line[5].strip()) == 1:
      rights_list.loc[j - first_line, 'cfs'] = float(split_line[4].strip()) * 1.98 * 30.0
    else:
      rights_list.loc[j - first_line, 'cfs'] = 0.0
    counter = 0
    if split_line[2].strip()[2:6] == '_ADC':
      for watershed_name in huc_8_watersheds:
        if split_line[2].strip()[6:9] in aggregate_structures[watershed_name]:
          rights_list.loc[j - first_line, 'watershed'] = float(counter)
          break
        counter += 1
    else:
      for watershed_name in huc_8_watersheds:
        if split_line[2].strip()[0:7] in structures_list_huc8[watershed_name]:
          rights_list.loc[j - first_line, 'watershed'] = float(counter)
        counter += 1
    if pd.isna(rights_list.loc[j - first_line, 'watershed']):
      rights_list.loc[j - first_line, 'watershed'] = len(huc_8_watersheds)

rights_list = rights_list[pd.notnull(rights_list['watershed'])]
rights_list = rights_list.reset_index()
rights_list_sorted = rights_list.sort_values(by = ['priority'])
rights_decree = np.asarray(rights_list_sorted['cfs'])
rights_watershed = np.asarray(rights_list_sorted['watershed'])
rights_priority = np.asarray(rights_list_sorted['priority'])
for index, row in demand_list.iterrows():
  rights_structure = rights_list_sorted[rights_list_sorted['structure'] == row['structure']]
  for month_no in range(1, 13):
    remaining_demand = row['demand ' + str(month_no)] * 1.0
    for index_rs, row_rs in rights_structure.iterrows():
      right_demand = min(max(remaining_demand, 0.0), rights_list_sorted.loc[index_rs, 'cfs'])
      rights_list_sorted.loc[index_rs, 'demand ' + str(month_no)] = right_demand * 1.0
      remaining_demand -= right_demand * 1.0
      if row['structure'] == '51_ADC001':
        print(month_no, end = " ")
        print(x, end = " ")
        print(index_rs, end = " ")
        print(right_demand, end = " ")
        print(remaining_demand, end = " ")
        print(rights_list_sorted.loc[index_rs, 'demand ' + str(month_no)])

#deliveries_dict = {}
#with open ('UCRB_analysis-master/Sobol_sample/Experiment_files/cm2015B.xdd', 'rt') as xdd_file:
#  for line in xdd_file:
#    data = line.split()
#    if data:
#      if len(data) > 1 and data[0] != '#':
#        struct_id = data[1].strip()
#        if '(' in struct_id or ')' in struct_id or struct_id[0] == '*':
#          struct_id = 'donotusethis'
#        if rights_list['structure'].str.contains(struct_id).any():
#          month_id = data[3].strip()
#          total_delivery = float(data[4].strip()) - float(data[17].strip())
#          if struct_id in deliveries_dict:
#            if month_id in deliveries_dict[struct_id]:
#              deliveries_dict[struct_id][month_id].append(total_delivery)
#            else:
#              deliveries_dict[struct_id][month_id] = [total_delivery,]          
#          else:
#            deliveries_dict[struct_id] = {}
#            deliveries_dict[struct_id][month_id] = [total_delivery,]
#            print(len(deliveries_dict), end = " ")
#            print(struct_id)
            
#deliveries_df = pd.DataFrame()
#for x in deliveries_dict:
#  for y in deliveries_dict[x]:
#    deliveries_df[x + '_' + y] = np.asarray(deliveries_dict[x][y])
#deliveries_df.to_csv('delivery_by_struct.csv')

call_structs_df = pd.DataFrame(columns = ['year', 'month', 'structure', 'right'])
data_col = False
call_struct = []
year_call = []
month_call = []
call_right = []
with open ('UCRB_analysis-master/Sobol_sample/Experiment_files/cm2015B.xca', 'rt') as xca_file:
  for line in xca_file:
    data = line.split()
    if data:
      if data[0] != '#' and data_col:
        call_struct.append(data[5].strip())
        call_right.append(float(data[6].strip()))
        year_call.append(int(data[1].strip()))
        month_call.append(data[2].strip())
      else:
        data_col = True
        try:
          int(data[1])
        except:
          data_col = False

call_structs_df['year'] = year_call
call_structs_df['month'] = month_call
call_structs_df['structure'] = call_struct
call_structs_df['right'] = call_right
call_structs_df.to_csv('UCRB_analysis-master/Sobol_sample/Experiment_files/call_times.csv')
print('write calls')

baseflow_filename = 'UCRB_analysis-master\Sobol_sample\Experiment_files\cm2015.rin'
with open(baseflow_filename,'r') as f:
  all_data_XBM = [x for x in f.readlines()]       
f.close() 

column_lengths=[12,24,13,17,4]
split_line = ['']*len(column_lengths)
character_breaks=np.zeros(len(column_lengths),dtype=int)
character_breaks[0]=column_lengths[0]
for i in range(1,len(column_lengths)):
  character_breaks[i]=character_breaks[i-1]+column_lengths[i]

copy_all_data = np.copy(all_data_XBM)
# Change only for specific samples
  # Change only the specific lines

id_list = []
upstream_list = []
for j in range(0,len(all_data_XBM)):
  if all_data_XBM[j][0] == '#':
    first_line = int(j * 1)
  else:
    split_line[0]=all_data_XBM[j][0:character_breaks[0]]
    
    for i in range(1,len(split_line)):
      split_line[i]=all_data_XBM[j][character_breaks[i-1]:character_breaks[i]]
    id_list.append(split_line[0].strip())
    upstream_list.append(split_line[2].strip())
    print(j, end = " ")
    print(split_line[0].strip(), end = " ")
    print(split_line[2].strip())
flow_network = pd.DataFrame(index = id_list, columns = ['downstream_station',])
flow_network['downstream_station'] = upstream_list

for month_no in range(1, 13):
  rights_list_sorted['downstream_senior_demand ' + str(month_no)] = np.zeros(len(rights_list_sorted.index))
  rights_list_sorted['upstream_senior_demand ' + str(month_no)] = np.zeros(len(rights_list_sorted.index))

relative_rights_stack = {}
counter = 0
rights_stack = {}
for index, row in flow_network.iterrows():
  print(index, end = " ")
  print(counter)
  counter += 1
  rights_structure = rights_list_sorted[rights_list_sorted['structure'] == index]
  downstream_index = row['downstream_station']
  rights_id = []
  priority_order = []
  priority_volume = []
  rights_id.extend(rights_structure.index)
  priority_order.extend(rights_structure['priority'])
  priority_volume.extend(rights_structure['cfs'])
  
  while downstream_index != 'coloup_end' and index != 'coloup_end':
    downstream_rights = rights_list_sorted[rights_list_sorted['structure'] == downstream_index]
    rights_id.extend(downstream_rights.index)
    priority_order.extend(downstream_rights['priority'])
    priority_volume.extend(downstream_rights['cfs'])
    downstream_index = flow_network.loc[downstream_index, 'downstream_station']
  downstream_order = np.argsort(np.asarray(priority_order))
  priority_volume = np.asarray(priority_volume)
  ordered_volumes = priority_volume[downstream_order]
  rights_id = np.asarray(rights_id)
  ordered_ids = rights_id[downstream_order]
  rights_stack[index] = pd.DataFrame(index = ordered_ids, columns = rights_structure.index)
  
  for index_rs, row_rs in rights_structure.iterrows():
    total_stack = 0.0
    rights_distance = np.zeros(len(ordered_ids))
    for stack_counter in range(0, len(ordered_ids)):
      if ordered_ids[stack_counter] == index_rs:
        break
    total_stack = np.sum(ordered_volumes[:stack_counter])
    for sc2 in range(0, stack_counter):
      total_stack -= ordered_volumes[sc2]
      rights_distance[sc2] = total_stack * (-1.0)
    total_stack = 0.0
    for sc3 in range(stack_counter, len(ordered_ids)):
      rights_distance[sc3] = total_stack * 1.0
      total_stack += ordered_volumes[sc3]
    rights_stack[index][index_rs] = rights_distance
    print(index, end = " ")
    print(index_rs, end = " ")
    print(rights_stack[index][index_rs])
    #for index_us, row_us in rights_structure.iterrows():
      #ds_demand = 0.0
      #for index_ds, row_ds in downstream_rights.iterrows():
          #if float(row_us['priority']) > float(row_ds['priority']):
            #for month_no in range(1, 13):            
              #rights_list_sorted.loc[index_us, 'downstream_senior_demand ' + str(month_no)] += float(row_ds['demand ' + str(month_no)]) * 1.0
          #else:
            #for month_no in range(1, 13):
              #rights_list_sorted.loc[index_ds, 'upstream_senior_demand ' + str(month_no)] += float(row_us['demand ' + str(month_no)]) * 1.0 


for month_no in range(1, 13):
  rights_list_sorted['prob ' + str(month_no)] = np.zeros(len(rights_list_sorted.index))

deliveries_df = pd.read_csv('delivery_by_struct.csv')

counter = 0
all_deliveries = {}
for timeseries_name in deliveries_df:
  struct_id = timeseries_name[:-4]
  month_name = timeseries_name[-3:]
  rights_structure = rights_list_sorted[rights_list_sorted['structure'] == struct_id]
  if len(rights_structure) > 0:
    skip = 0  
    if month_name == 'JAN':
      month_no = 1
    elif month_name == 'FEB':
      month_no = 2
    elif month_name == 'MAR':
      month_no = 3
    elif month_name == 'APR':
      month_no = 4
    elif month_name == 'MAY':
      month_no = 5
    elif month_name == 'JUN':
      month_no = 6
    elif month_name == 'JUL':
      month_no = 7
    elif month_name == 'AUG':
      month_no = 8
    elif month_name == 'SEP':
      month_no = 9
    elif month_name == 'OCT':
      month_no = 10
    elif month_name == 'NOV':
      month_no = 11
    elif month_name == 'DEC':
      month_no = 12
    else:
      skip = 1
    if skip == 0:
      this_struct_values = np.asarray(deliveries_df[struct_id + '_' + month_name])
      for x in range(0, len(deliveries_df[struct_id + '_' + month_name])):
        remaining_demand = this_struct_values[x] * 1.0
        for index_rs, row_rs in rights_structure.iterrows():
          right_demand = min(max(remaining_demand, 0.0), rights_list_sorted.loc[index_rs, 'demand ' + str(month_no)])
          if right_demand >= 0.001:
            rights_list_sorted.loc[index_rs, 'prob ' + str(month_no)] += 1.0/float(len(this_struct_values))
            if index_rs in all_deliveries:
              if month_name in all_deliveries[index_rs]:
                all_deliveries[index_rs][month_name][x] = right_demand / rights_list_sorted.loc[index_rs, 'demand ' + str(month_no)]
              else:
                for all_m in ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']:
                  all_deliveries[index_rs][all_m] = np.zeros(len(deliveries_df[struct_id + '_' + month_name]))
                all_deliveries[index_rs][month_name][x] = right_demand / rights_list_sorted.loc[index_rs, 'demand ' + str(month_no)]
            else:
              all_deliveries[index_rs] = {}
              for all_m in ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']:
                all_deliveries[index_rs][all_m] = np.zeros(len(deliveries_df[struct_id + '_' + month_name]))
              all_deliveries[index_rs][month_name][x] = right_demand / rights_list_sorted.loc[index_rs, 'demand ' + str(month_no)]
          else:
            if index_rs in all_deliveries:
              if month_name in all_deliveries[index_rs]:
                all_deliveries[index_rs][month_name][x] = 0.0
              else:
                for all_m in ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']:
                  all_deliveries[index_rs][all_m] = np.zeros(len(deliveries_df[struct_id + '_' + month_name]))
                all_deliveries[index_rs][month_name][x] = 0.0
            else:
              all_deliveries[index_rs] = {}
              for all_m in ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']:
                all_deliveries[index_rs][all_m] = np.zeros(len(deliveries_df[struct_id + '_' + month_name]))
              all_deliveries[index_rs][month_name][x] = 0.0
          remaining_demand -= right_demand * 1.0

          
  counter += 1
  print(counter)
  if counter > 500:
    break
  
snowpack_values = {}
for watershed_name in huc_8_watersheds:
  snowpack_values[watershed_name] = pd.read_csv('UCRB_analysis-master/Sobol_sample/Snow_Data/Basin_' + watershed_name + '/basinwide_snowpack.csv')
 
month_snow = 'jun'
month_calls = 'JUN'
month_delivery = 'JUL'
month_cycle = ['OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP']
month_cycle_snow = ['none' , 'none', 'none', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jun', 'jun', 'jun']
start_year = 1908
call_structs_df = pd.read_csv('UCRB_analysis-master/Sobol_sample/Experiment_files/call_times.csv')
calls_month = call_structs_df[call_structs_df['month'] == month_calls]
counter = 0
for x in rights_stack:
  print(counter, end = " ")
  print(x, end = " ")
  print(type(x))
for index_rs, row_rs in rights_list_sorted[400:].iterrows():
  use_line = True
  index_rs = 1194
  try:
    delivery_values = all_deliveries[index_rs][month_delivery]
    downstream_stations = rights_stack[row_rs['structure']]
    distance_from_call = downstream_stations.loc[downstream_stations.index[-1], index_rs]
  except:
    use_line = False
  print(row_rs['watershed'])
  if int(row_rs['watershed']) < 5 and use_line:

    downstream_stations = rights_stack[row_rs['structure']]
    print(downstream_stations)
    counter = 0
    for watershed_name in huc_8_watersheds:
      if int(row_rs['watershed']) == counter:
        watershed_snow = snowpack_values[watershed_name]
      counter += 1
    this_right_values = {}
    rights_structure = rights_list_sorted[rights_list_sorted['structure'] == row_rs['structure']]
    for month_name in month_cycle:
      this_struct_values = np.asarray(deliveries_df[row_rs['structure'] + '_' + month_name])
      print(deliveries_df[row_rs['structure'] + '_' + month_name])
      print(this_struct_values)
      this_right_values[month_name] = np.zeros(len(this_struct_values))
      this_right_values[month_name + '_demand'] = np.zeros(len(this_struct_values))
      for x in range(0, len(this_struct_values)):
        remaining_demand = this_struct_values[x] * 1.0
        for index_rs2, row_rs2 in rights_structure.iterrows():
          right_demand = min(max(remaining_demand, 0.0), rights_list_sorted.loc[index_rs2, 'demand ' + str(month_no)])
          remaining_demand -= right_demand * 1.0          
          print(month_name, end = " ")
          print(x, end = " ")
          print(index_rs2, end = " ")
          print(index_rs, end = " ")
          print(right_demand, end = " ")
          print(remaining_demand, end = " ")
          print(rights_list_sorted.loc[index_rs, 'demand ' + str(month_no)])
          if index_rs2 == index_rs:
            this_right_values[month_name][x] = right_demand * 1.0
            this_right_values[month_name + '_demand'][x] = rights_list_sorted.loc[index_rs, 'demand ' + str(month_no)]
          
    snowpack_start_toggle = 0
    fig, ax = plt.subplots()
    season_colors = sns.color_palette('RdYlBu', 12).as_hex()

    for hist_year in range(1908, 2013):
      monthcounter = 0
      print(hist_year)
      for snowmonth, callmonth in zip(month_cycle_snow, month_cycle):
        toggle_month = 0
        total_delivery = 0.0
        total_demand = 0.0
        for monthsum_name in month_cycle:
          if monthsum_name == callmonth:
            toggle_month = 1
          if toggle_month == 1:
            total_delivery += this_right_values[monthsum_name][hist_year - 1908]
            total_demand += this_right_values[monthsum_name + '_demand'][hist_year - 1908]
            
        if snowmonth == 'none':
          if watershed_snow.loc[hist_year - 1908, 'jan'] > -990.0:
            monthly_snow = 0.0
          else:
            monthly_snow = -999.9          
        else:
          monthly_snow = watershed_snow.loc[hist_year - 1908, snowmonth]

        if monthly_snow > -990.0 and snowpack_start_toggle == 0:
          snowpack_start_toggle = 1
          snowpack_start_year = hist_year * 1
          monthly_call_distance = np.zeros((2014-snowpack_start_year, 12))
          monthly_snowpack = np.zeros((2014-snowpack_start_year, 12))
          monthly_remaining_fraction = np.zeros((2014-snowpack_start_year, 12))
        if snowpack_start_toggle == 1:
        
          this_month_calls = np.logical_and(call_structs_df['month'] == callmonth, call_structs_df['year'] == hist_year)
          calls_month = call_structs_df[this_month_calls]

          distance_from_call = downstream_stations.loc[downstream_stations.index[-1], index_rs]

          for index_cm, row_cm in calls_month.iterrows():
            right_called = row_cm['right'] * 1.98 * 30.0
            structure_name = row_cm['structure']
            this_structure_rights = rights_list_sorted[rights_list_sorted['structure'] == structure_name]
            calling_right = this_structure_rights[this_structure_rights['cfs'] == right_called]
            if len(calling_right) == 1:            
              try:
                distance_from_call = min(distance_from_call, downstream_stations.loc[calling_right.index[0], index_rs])
              except:
                e = 1
          if total_demand > 0.0:
            monthly_remaining_fraction[hist_year - snowpack_start_year, monthcounter] = total_delivery/total_demand
          monthly_call_distance[hist_year - snowpack_start_year, monthcounter] = distance_from_call * 1.0
          monthly_snowpack[hist_year - snowpack_start_year, monthcounter] = monthly_snow * 1
          if monthcounter > 0:
            ax.plot([monthly_call_distance[hist_year - snowpack_start_year, monthcounter - 1], monthly_call_distance[hist_year - snowpack_start_year, monthcounter]], [monthly_snowpack[hist_year - snowpack_start_year, monthcounter - 1], monthly_snowpack[hist_year - snowpack_start_year, monthcounter]], color = season_colors[int(monthly_remaining_fraction[hist_year - snowpack_start_year, monthcounter] * 100.0)], linewidth = 3.0)
            
        monthcounter += 1
    plt.show()
    plt.close() 
    
for index_rs, row_rs in rights_list_sorted.iterrows():
  use_line = True
  try:
    delivery_values = all_deliveries[index_rs][month_delivery]
  except:
    use_line = False
  if int(row_rs['watershed']) < 5 and use_line:
    
    structure_rights = rights_list_sorted[rights_list_sorted['structure'] == structure_name]
    season_colors = sns.color_palette('rocket', 100).as_hex()
    snowpack_1 = []
    snowpack_2 = []
    delivery_1 = []
    call_distance = []
    for xx in range(0, len(delivery_values)):
      if monthly_average[xx + 1] > 0:
        snowpack_1.append(monthly_average[xx + 1])
        delivery_1.append(delivery_values[xx])
        snowpack_2.append(weighted_average[xx + 1])
    
    snowpack_1 = np.asarray(snowpack_1)
    snowpack_2 = np.asarray(snowpack_2)
    delivery = np.asarray(delivery_1)
    delivery_ones = delivery < 1.0
    delivery_zeros = delivery > 0.0
    if len(snowpack_1[delivery_ones]) > 0:
      sn_max_1 = max(snowpack_1[delivery_ones])
      sn_max_2 = max(snowpack_2[delivery_ones])
    else:
      sn_max_1 = 0.0
      sn_max_2 = 0.0
    if len(snowpack_2[delivery_zeros]) > 0:
      sn_min_1 = min(snowpack_1[delivery_zeros])
      sn_min_2 = min(snowpack_2[delivery_zeros])
    else:
      sn_min_1 = max(snowpack_1)
      sn_min_2 = max(snowpack_2)
    
    deliveries_both = np.logical_and(np.logical_and(snowpack_1 < sn_max_1, snowpack_1 > sn_min_1), np.logical_and(snowpack_2 < sn_max_2, snowpack_2 > sn_min_2))
    if len(snowpack_1[deliveries_both]) > 60:
      bins_left_1 = np.asarray(pd.qcut(pd.Series(snowpack_1[deliveries_both]), q = 4).cat.categories.left)
      bins_right_1 = np.asarray(pd.qcut(pd.Series(snowpack_1[deliveries_both]), q = 4).cat.categories.right)
      bins_left_1[0] = sn_min_1 + .00001
      bins_right_1[-1] = sn_max_1 - .00001
    elif len(snowpack_1[deliveries_both]) > 30:
      bins_left_1 = np.asarray(pd.qcut(pd.Series(snowpack_1[deliveries_both]), q = 3).cat.categories.left)
      bins_right_1 = np.asarray(pd.qcut(pd.Series(snowpack_1[deliveries_both]), q = 3).cat.categories.right)
      bins_left_1[0] = sn_min_1 + .00001
      bins_right_1[-1] = sn_max_1 - .00001
    elif len(snowpack_1[deliveries_both]) > 10:
      bins_left_1 = np.asarray(pd.qcut(pd.Series(snowpack_1[deliveries_both]), q = 2).cat.categories.left)
      bins_right_1 = np.asarray(pd.qcut(pd.Series(snowpack_1[deliveries_both]), q = 2).cat.categories.right)
      bins_left_1[0] = sn_min_1 + .00001
      bins_right_1[-1] = sn_max_1 - .00001
    else:
      bins_left_1[0] = sn_min_1 + .00001
      bins_right_1[-1] = sn_max_1 - .00001
    
    if len(snowpack_1[deliveries_both]) > 0:
      for left1, right1 in zip(bins_left_1, bins_right_1):
        this_bin = np.logical_and(snowpack_1 < right1, snowpack_1 >= left1)
        if len(snowpack_2[this_bin]) > 20:
          bins_left_2 = np.asarray(pd.qcut(pd.Series(snowpack_2[this_bin]), q = 4).cat.categories.left)
          bins_right_2 = np.asarray(pd.qcut(pd.Series(snowpack_2[this_bin]), q = 4).cat.categories.right)
          bins_left_2[0] = sn_min_2 + .0001
          bins_right_2[-1] = sn_max_2 - .0001
        if len(snowpack_2[this_bin]) > 15:
          bins_left_2 = np.asarray(pd.qcut(pd.Series(snowpack_2[this_bin]), q = 3).cat.categories.left)
          bins_right_2 = np.asarray(pd.qcut(pd.Series(snowpack_2[this_bin]), q = 3).cat.categories.right)
          bins_left_2[0] = sn_min_2 + .0001
          bins_right_2[-1] = sn_max_2 - .0001
        elif len(snowpack_2[this_bin]) > 10:
          bins_left_2 = np.asarray(pd.qcut(pd.Series(snowpack_2[this_bin]), q = 2).cat.categories.left)
          bins_right_2 = np.asarray(pd.qcut(pd.Series(snowpack_2[this_bin]), q = 2).cat.categories.right)
          bins_left_2[0] = sn_min_2 + .0001
          bins_right_2[-1] = sn_max_2 - .0001
        else:
          bins_left_2 = np.asarray([sn_min_2 + .0001,])
          bins_right_2 = np.asarray([sn_max_2 - .0001,])
        
        for left2, right2 in zip(bins_left_2, bins_right_2):
          deliveries_bin = np.logical_and(np.logical_and(snowpack_1 >= left1, snowpack_1 < right1), np.logical_and(snowpack_2 >= left2, snowpack_2 < right2))
          if len(delivery[deliveries_bin]) > 0:
            section_average = np.mean(delivery[deliveries_bin])
            color_number = int(section_average * 100)
            if color_number == 100:
              color_number -= 1
            ax.fill_between([left1, right1], [left2, left2], [right2, right2], facecolor = season_colors[color_number], edgecolor = 'black', alpha = 1.0)
          else:
            ax.fill_between([left1, right1], [left2, left2], [right2, right2], facecolor = 'lightslategray', edgecolor = 'black', alpha = 1.0)
            print()
    ax.fill_between([0.0, sn_min_1], [0.0, 0.0], [max(snowpack_2), max(snowpack_2)], facecolor = season_colors[0], edgecolor = 'black', alpha = 1.0)
    ax.fill_between([sn_max_1, max(snowpack_1)], [0.0, 0.0], [max(snowpack_2), max(snowpack_2)], facecolor = season_colors[99], edgecolor = 'black', alpha = 1.0)
    ax.fill_between([0.0, max(snowpack_1)], [0.0, 0.0], [sn_min_2, sn_min_2], facecolor = season_colors[0], edgecolor = 'black', alpha = 1.0)
    ax.fill_between([0.0, max(snowpack_1)], [sn_max_2,sn_max_2],[max(snowpack_2), max(snowpack_2)], facecolor = season_colors[99], edgecolor = 'black', alpha = 1.0)
    ax.set_xlabel('% of average snowpack, HUC8 basin')
    ax.set_ylabel('area -weighted snowpack, UCRC')
    ax.set_xlim([min(snowpack_1), max(snowpack_1)])
    ax.set_ylim([min(snowpack_2), max(snowpack_2)])
    
    plt.show()
    plt.close()
    fig, ax = plt.subplots(2)
    ax[0].scatter(snowpack_1, delivery)
    ax[1].scatter(snowpack_2, delivery)
    ax[1].set_xlabel('area -weighted snowpack, UCRC')
    ax[0].set_xlabel('% of average snowpack, HUC8 basin')
    ax[1].set_ylabel('delivery (% demand)')
    ax[0].set_ylabel('delivery (% demand)')
    plt.show()
    plt.close()
 
cumulative_decree = {}
cumulative_prob = {}
len_years = len(rights_list_sorted['demand 1'])
huc_8_watersheds.append('special')
for watershed_name in huc_8_watersheds:
  cumulative_decree[watershed_name] = np.zeros((12,11))
  cumulative_prob[watershed_name] = {}
  for month_no in range(1,13):
    cumulative_prob[watershed_name]['probs ' + str(month_no)] = np.zeros(len_years)
    cumulative_prob[watershed_name]['cumdemand ' + str(month_no)] = np.zeros(len_years)
    cumulative_prob[watershed_name]['demand ' + str(month_no)] = np.zeros(len_years)
    cumulative_prob[watershed_name]['senior_demand_ratio ' + str(month_no)] = np.zeros(len_years)
  

  
current_limit = 0.0
decree_counter = 0
last_value = 0
running_rights = np.zeros((len(huc_8_watersheds) + 1, 12))
for month_no in range(0, 12):
  rights_demand = np.asarray(rights_list_sorted['demand ' + str(month_no + 1)])
  rights_prob = np.asarray(rights_list_sorted['prob ' + str(month_no + 1)])

  running_rights[int(rights_watershed[0]), month_no] += rights_demand[0]
  running_rights[len(huc_8_watersheds), 0] += rights_demand[0]
  counter = 0
  for watershed_name in huc_8_watersheds:
    if int(rights_watershed[0]) == counter:
      cumulative_prob[watershed_name]['cumdemand '+ str(month_no + 1)][0] += rights_demand[0]
      cumulative_prob[watershed_name]['probs ' + str(month_no + 1)][0] += rights_prob[0] * rights_demand[0]
      cumulative_prob[watershed_name]['demand ' + str(month_no + 1)][0] += rights_demand[0]
    counter += 1
    
for x in range(1, len(rights_list_sorted['structure'])):
  if running_rights[len(huc_8_watersheds), 0] > current_limit + 350000.0 and decree_counter < 10:
    counter = 0
    for watershed_name in huc_8_watersheds:
      for month_no in range(0, 12):
        cumulative_decree[watershed_name][month_no, decree_counter] = running_rights[counter, month_no]
      counter += 1

    decree_counter+= 1
    current_limit += 350000.0
  elif x == len(rights_list_sorted['structure']) - 1:
    counter = 0
    for watershed_name in huc_8_watersheds:
      for month_no in range(0, 12):
        cumulative_decree[watershed_name][month_no, decree_counter] = running_rights[counter, month_no]
      counter += 1
    
  for month_no in range(1, 13):
    rights_demand = np.asarray(rights_list_sorted['demand ' + str(month_no)])
    rights_prob = np.asarray(rights_list_sorted['prob ' + str(month_no)])
    running_rights[len(huc_8_watersheds), 0] += rights_demand[x]
    counter = 0
    for watershed_name in huc_8_watersheds:
      if int(rights_watershed[x]) == counter:
        running_rights[counter, month_no - 1] += rights_demand[x]
        cumulative_prob[watershed_name]['probs ' + str(month_no)][x] += rights_prob[x] * rights_demand[x]
        cumulative_prob[watershed_name]['demand ' + str(month_no)][x] += rights_demand[x]

      counter += 1

  counter = 0
  for watershed_name in huc_8_watersheds:
    for month_no in range(0, 12):
      upstream_demand = np.asarray(rights_list_sorted['upstream_senior_demand ' + str(month_no + 1)])
      downstream_demand = np.asarray(rights_list_sorted['downstream_senior_demand ' + str(month_no + 1)])
      cumulative_prob[watershed_name]['cumdemand ' + str(month_no + 1)][x] += running_rights[counter, month_no]
      if (downstream_demand[x] + upstream_demand[x]) > 0.0:
        cumulative_prob[watershed_name]['senior_demand_ratio ' + str(month_no + 1)][x] = upstream_demand[x] /(downstream_demand[x] + upstream_demand[x])
    counter += 1
                  
counter = 0
season_colors = sns.color_palette('rocket', 11).as_hex()
fig, ax = plt.subplots(len(huc_8_watersheds))
for watershed_name in huc_8_watersheds:
  bottom_layer = np.zeros(12)
  for decree_counter in range(0, 11):
    top_layer = copy(cumulative_decree[watershed_name][:,decree_counter])
    ax[counter].fill_between(np.arange(12), bottom_layer, top_layer, color = season_colors[decree_counter])
    bottom_layer = copy(cumulative_decree[watershed_name][:,decree_counter])
    
  if counter == len(huc_8_watersheds) - 1:
    ax[counter].set_xticks([0, 4, 7, 11])
    ax[counter].set_xticklabels(['Jan', 'May', 'August', 'December'])
    legend_location = 'upper left'
    legend_element = []
    legend_element.append(Patch(facecolor=season_colors[0], edgecolor='black', label='Most Senior', alpha = 0.7))
    legend_element.append(Patch(facecolor=season_colors[-1], edgecolor='black', label='Least Senior', alpha = 0.7))

    legend_properties = {'family':'Gill Sans MT','size':8}
    ax[counter].legend(handles=legend_element, loc=legend_location, prop=legend_properties) #legend 'Title' fontsize

  else:
    ax[counter].set_xticks([])
    ax[counter].set_xticklabels('')
  ax[counter].set_xlim([0, 11])
  counter +=1 
  
plt.show()

season_colors = sns.color_palette('rocket', len(cumulative_prob[watershed_name]['senior_demand_ratio ' + str(month_no + 1)])).as_hex()
for month_no in range(1, 13):
  counter = 0
  counter2 = 0
  fig, ax = plt.subplots(3, 2)
  for watershed_name in huc_8_watersheds:
    for x in range(0, len(cumulative_prob[watershed_name]['probs ' + str(month_no)])):
      if x == 0:
        xvals = [0.0, cumulative_prob[watershed_name]['demand ' + str(month_no)][x]]
      else:
        xvals = [cumulative_prob[watershed_name]['cumdemand ' + str(month_no)][x-1], cumulative_prob[watershed_name]['cumdemand ' + str(month_no)][x]]
      print(watershed_name, end = " ")
      print(x, end = " ")
      print(xvals)      
      if cumulative_prob[watershed_name]['demand ' + str(month_no)][x] > 0.0:
        y_height = cumulative_prob[watershed_name]['probs ' + str(month_no)][x]/cumulative_prob[watershed_name]['demand ' + str(month_no)][x]
        yvals = [y_height, y_height]
        color_index = int(len(cumulative_prob[watershed_name]['senior_demand_ratio ' + str(month_no + 1)]) * cumulative_prob[watershed_name]['senior_demand_ratio ' + str(month_no + 1)][x])
        if color_index == len(cumulative_prob[watershed_name]['senior_demand_ratio ' + str(month_no + 1)]):
          color_index -= 1
        ax[counter][counter2].fill_between(xvals, [0.0, 0.0], yvals, color = season_colors[color_index])
    
    ax[counter][counter2].set_xlim([0.0, cumulative_prob[watershed_name]['cumdemand ' + str(month_no)][-1]])
    ax[counter][counter2].set_ylim([0.0, 1.0])
    counter +=1 
    if counter == 3:
      counter = 0
      counter2 += 1
  
  plt.tight_layout()
  plt.show()
  plt.close()


data_figure = Mapper()
projection_string = 'EPSG:3857'#project raster to data projection

##raster file names
raster_name_pt1 = 'LC08_L1TP_'
raster_name_pt2 = '_02_T1'
raster_band_list = ['_B4', '_B3', '_B2']
raster_id_list = {}
raster_id_list['034033'] = ['20200702_20200913',]
raster_id_list['034032'] = ['20200702_20200913',]
raster_id_list['036032'] = ['20200817_20200920',]
raster_id_list['036033'] = ['20200817_20200920',]
raster_id_list['035032'] = ['20200709_20200912',]
raster_id_list['035033'] = ['20200709_20200912',]
snow_stations = pd.read_csv('BraysBayou/colorado_snow_stations.csv')
geometry = [Point(xy) for xy in zip(snow_stations['Lon'], snow_stations['Lat'])]
crs = 'EPSG:4326'
snow_stations_gdf = gpd.GeoDataFrame(snow_stations['Site_Name'], crs = crs, geometry = geometry)
snow_stations_gdf = snow_stations_gdf.to_crs(epsg = 3857)
with open('BraysBayou/Shapefiles_UCRB/transboundary_diversions.json') as json_data:
  transboundary_stations = json.load(json_data)
for x in transboundary_stations:
  print(x)
  for y in transboundary_stations[x]:
    print(y)
lats = []
longs = []
names = []
for x in transboundary_stations['features']:
  lats.append(x['properties']['Latitude'])
  longs.append(x['properties']['Longitude'])
  names.append(x['properties']['TransbasinDiversionName'])
geometry = [Point(xy) for xy in zip(longs, lats)]
transboundary_stations_gdf = gpd.GeoDataFrame(names, crs = crs, geometry = geometry)
transboundary_stations_gdf = transboundary_stations_gdf.to_crs(epsg = 3857)
print(transboundary_stations_gdf)


data_figure.load_batch_raster('BraysBayou/stitched_satellite/', raster_id_list, raster_name_pt1, raster_name_pt2, raster_band_list, projection_string)

area_ucrb = gpd.read_file('BraysBayou/Shapefiles_UCRB/DIV3CO.shp')
area_ucrb = area_ucrb[area_ucrb['DIV'] == 5]
districts_ucrb = gpd.read_file('BraysBayou/Shapefiles_UCRB/Water_Districts.shp')
ditches_ucrb = gpd.read_file('BraysBayou/Shapefiles_UCRB/Div5_Irrigated_Lands_2015/Div5_2015_Ditches.shp')
irrigation_ucrb = gpd.read_file('BraysBayou/Shapefiles_UCRB/Div5_Irrigated_Lands_2015/Div5_Irrig_2015.shp')
streams_ucrb = gpd.read_file('BraysBayou/Shapefiles_UCRB/UCRBstreams.shp')
structures_ucrb = gpd.read_file('BraysBayou/Shapefiles_UCRB/Div_5_structures.shp')
flowlines_ucrb = gpd.read_file('BraysBayou/Shapefiles_UCRB/flowline.shp')
area_ucrb = area_ucrb.to_crs(epsg = 3857)
districts_ucrb = districts_ucrb.to_crs(epsg = 3857)
ditches_ucrb = ditches_ucrb.to_crs(epsg = 3857)
irrigation_ucrb = irrigation_ucrb.to_crs(epsg = 3857)
streams_ucrb = streams_ucrb.to_crs(epsg = 3857)
structures_ucrb = structures_ucrb.to_crs(epsg = 3857)
flowlines_ucrb = flowlines_ucrb.to_crs(epsg = 3857)
print(list(set(structures_ucrb['StructType'])))
print(list(set(structures_ucrb['FeatureTyp'])))
print(list(set(structures_ucrb['CurrInUse'])))
for x in irrigation_ucrb:
  print(x)
  print(irrigation_ucrb[x])
crop_list = list(set(irrigation_ucrb['CROP_TYPE']))
print(crop_list)
plot_type = 'fig1'
perennial_crops = irrigation_ucrb[irrigation_ucrb['PERENNIAL'] == 'YES']
streams_ucrb = gpd.sjoin(streams_ucrb, area_ucrb, how = 'inner', op = 'intersects')
data_figure.plot_scale(area_ucrb, 'depth', type = 'polygons', solid_color = 'none', solid_alpha = 0.5, linewidth_size = 2.0, outline_color = 'black')
data_figure.plot_scale(streams_ucrb, 'depth', type = 'polygons', solid_color = 'steelblue', solid_alpha = 1.0, linewidth_size = 1.0, outline_color = 'steelblue')
data_figure.plot_scale(irrigation_ucrb, 'depth', type = 'polygons', solid_color = 'forestgreen', solid_alpha = 1.0, linewidth_size = 0.2, outline_color = 'forestgreen')
data_figure.plot_scale(ditches_ucrb, 'depth', type = 'points', solid_color = 'black', solid_alpha = 0.5, linewidth_size = 0.0, outline_color = 'black', markersize = 2)
if plot_type == 'fig1':
  data_figure.plot_scale(transboundary_stations_gdf, 'depth', type = 'points', solid_color = 'indianred', solid_alpha = 1.0, linewidth_size = 0.0, outline_color = 'black', markersize = 15)
  data_figure.plot_scale(snow_stations_gdf, 'depth', type = 'points', solid_color = 'navy', solid_alpha = 1.0, linewidth_size = 0.0, outline_color = 'black', markersize = 15)
  data_figure.plot_scale(perennial_crops, 'depth', type = 'polygons', solid_color = 'goldenrod', solid_alpha = 1.0, linewidth_size = 0.2, outline_color = 'goldenrod')
if plot_type == 'fig2':
  data_figure.plot_scale(snow_stations_gdf, 'depth', type = 'points', solid_color = 'navy', solid_alpha = 1.0, linewidth_size = 0.0, outline_color = 'black', markersize = 15)
  data_figure.plot_scale(perennial_crops, 'depth', type = 'polygons', solid_color = 'goldenrod', solid_alpha = 1.0, linewidth_size = 0.2, outline_color = 'goldenrod')
if plot_type == 'fig3':
  data_figure.plot_scale(snow_stations_gdf, 'depth', type = 'points', solid_color = 'navy', solid_alpha = 1.0, linewidth_size = 0.0, outline_color = 'black', markersize = 15)

#plot colorbar w/perimeter
xrange = area_ucrb.total_bounds[2] - area_ucrb.total_bounds[0]
yrange = area_ucrb.total_bounds[3] - area_ucrb.total_bounds[1] 
data_figure.format_plot(xlim = (area_ucrb.total_bounds[0] - xrange*0.025, area_ucrb.total_bounds[2] + xrange*0.025), ylim = (area_ucrb.total_bounds[1]  -yrange*0.025, area_ucrb.total_bounds[3] + yrange*0.025))
legend_location = 'upper left'
if plot_type == 'fig1':
  legend_element = [Patch(facecolor='forestgreen', edgecolor='black', label='Field Crops'), 
                  Patch(facecolor='goldenrod', edgecolor='black', label='Perennial Crops'),
                  Line2D([0], [0], markerfacecolor='steelblue', markeredgecolor='black',  lw = 2, label='River'),
                  Line2D([0], [0], markerfacecolor='black', markeredgecolor='black',  lw = 2, marker = 'o', markersize = 10, label='Irrigation Diversion'),
                  Line2D([0], [0], markerfacecolor='navy', markeredgecolor='black',  lw = 2, marker = 'o', markersize = 10, label='Snowpack Station'),
                  Line2D([0], [0], markerfacecolor='indianred', markeredgecolor='black',  lw = 2, marker = 'o', markersize = 10, label='Transboundary Diversion')]
if plot_type == 'fig2':
  legend_element = [Patch(facecolor='forestgreen', edgecolor='black', label='Field Crops'), 
                  Patch(facecolor='goldenrod', edgecolor='black', label='Perennial Crops'),
                  Line2D([0], [0], markerfacecolor='steelblue', markeredgecolor='black', lw = 2, label='River'),
                  Line2D([0], [0], markerfacecolor='black', markeredgecolor='black', lw = 2, marker = 'o', markersize = 10, label='Irrigation Diversion'),
                  Line2D([0], [0], markerfacecolor='navy', markeredgecolor='black', lw = 2, marker = 'o', markersize = 10, label='Snowpack Station')]
if plot_type == 'fig3':
  legend_element = [Patch(facecolor='forestgreen', edgecolor='black', label='Irrigation'),
                  Line2D([0], [0], markerfacecolor='steelblue', markeredgecolor='black', lw = 2, label='River'),
                  Line2D([0], [0], markerfacecolor='black', markeredgecolor='black', lw = 2, marker = 'o', markersize = 10, label='Irrigation Diversion'),
                  Line2D([0], [0], markerfacecolor='navy', markeredgecolor='black', lw = 2, marker = 'o', markersize = 10, label='Snowpack Station')]
if plot_type == 'fig4':
  legend_element = [Patch(facecolor='forestgreen', edgecolor='black', label='Irrigation'),
                  Line2D([0], [0], markerfacecolor='steelblue', markeredgecolor='black', lw = 2, label='River'),
                  Line2D([0], [0], markerfacecolor='black', markeredgecolor='black', lw = 2, marker = 'o', markersize = 10, label='Irrigation Diversion')]
                  
legend_properties = {'family':'Gill Sans MT','weight':'bold','size':8}
data_figure.add_legend(legend_location, legend_element, legend_properties)
plt.savefig('BraysBayou/Shapefiles_UCRB/' + plot_type + '.png', dpi = 300, bbox_inches = 'tight', pad_inches = 0.0)

  #plot figure title
 # p2 = Polygon([(xl + (xr - xl) * 0.3, by + (uy - by) * 0.885), (xl + (xr - xl) * 0.7, by + (uy - by) * 0.885), (xl + (xr - xl) * 0.7, by + (uy - by) * 0.985), (xl + (xr - xl) * 0.3, by + (uy - by) * 0.985)])
  #df1 = gpd.GeoDataFrame({'geometry': p2, 'df1':[1,1]})
  #df1.crs = {'init' :'epsg:3857'}
  #data_figure.plot_scale(df1, 'none', type = 'polygons', solid_color = 'beige', solid_alpha = 0.4, outline_color = 'goldenrod', linewidth_size = 2)
  #data_figure.plot_scale(school_districts_use, 'none', type = 'polygons', solid_color = 'none', outline_color = 'beige', linewidth_size = 0.5)

      #label_title
  #data_figure.ax.text(xl + (xr - xl)*0.5, by + (uy - by) * 0.9, 'Harris County\n ' + str(end_time), horizontalalignment='center', fontsize = 10, weight = 'bold', fontname = 'Gill Sans MT')
plt.show()
