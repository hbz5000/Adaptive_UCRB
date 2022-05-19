import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from osgeo import gdal
import rasterio
from shapely.geometry import Point, Polygon, LineString
import geopandas as gpd
import fiona
from matplotlib.colors import ListedColormap
import matplotlib.pylab as pl
from skimage import exposure
import seaborn as sns
import sys
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


class Mapper():

  def __init__(self, nr = 1, nc = 0):
    self.sub_rows = nr
    self.sub_cols = nc
    if self.sub_cols == 0:
      self.fig, self.ax = plt.subplots(self.sub_rows)
      if self.sub_rows == 1:
        self.type = 'single'
        self.ax.grid(False)
      else:
        self.type = '1d'
    else:
      self.fig, self.ax = plt.subplots(self.sub_rows, self.sub_cols, figsize = (15,12.5))
      self.type = '2d'
    self.band_min = np.ones(3)*(-1.0)
    self.band_max = np.ones(3)*(-1.0)
    self.brightness_adj = 2.5
    plt.tight_layout()

  def load_sat(self, raster_name, pixel_bounds, projection_string, nr = 1, nc = 0):
    ds = gdal.Open(raster_name + '.tif')
    #geotransform - tells you where pixel 0 (x,y) is in the coordinate system and the scale of pixels to coordinates
    geotransform = ds.GetGeoTransform()
    ##clip the raster - note: this is done in UTM coordinates before projection
    ##so that you don't need to re-project entire contiguous US
    clip_name = raster_name + '_clipped.tif'
    ds = gdal.Translate(clip_name, ds, projWin = [geotransform[0] + 400*geotransform[1], geotransform[3] + 200*geotransform[5], geotransform[0] + 2500*geotransform[1], geotransform[3] +1200*geotransform[5]])
    output_name = raster_name + '_projected.tif'
    ##re-project raster from UTM to LAT/LONG
    gdal.Warp(output_name, ds, dstSRS = 'EPSG:4326')
    raster = rasterio.open(output_name)
    ###read RGB bands of raster
    b = raster.read()
    ##put bands into RGB order
    image = rasterio.plot.reshape_as_image(b)
    ##give coordinates of image bands
    spatial_extent = rasterio.plot.plotting_extent(raster)
    ##plot image
    if self.type == 'single':
      self.ax.imshow(image, extent = spatial_extent)
    elif self.type == '1d':
      self.ax[nr].imshow(image, extent = spatial_extent)
    elif self.type == '2d':
      self.ax[nr][nc].imshow(image, extent = spatial_extent)
  def add_legend(self, legend_location, legend_handle, legend_properties, nr = 0, nc = 0):
    if self.type == 'single':
      self.ax.legend(handles=legend_handle, loc=legend_location, prop=legend_properties)
    elif self.type == '2d':
      self.ax[nr][nc].legend(handles=legend_handle, loc=legend_location, prop=legend_properties)
  
  def load_shapefile_solid(self, shapefile_name, alpha_shp, color_shp, clip_list, nr = 1, nc = 0):
    map_shape = gpd.read_file(shapefile_name)
    map_shape['geometry'] = map_shape['geometry'].to_crs(epsg = 4326)
    if clip_list != 'None':
      map_shape = map_shape[map_shape[clip_list[0]] == clip_list[1]]	  
    
    if self.type == 'single':
      map_shape.plot(ax=self.ax, color = color_shp, alpha = alpha_shp)
    elif self.type == '1d':
      map_shape.plot(ax=self.ax[nr], color = color_shp, alpha = alpha_shp)
    elif self.type == '2d':
      map_shape.plot(ax=self.ax[nr][nc], color = color_shp, alpha = alpha_shp)
  def plot_scale(self, values, key, type = 'points', value_lim = 'None', solid_color = 'scaled', outline_color = 'black', colorscale = 'none', solid_alpha = 0.8, linewidth_size = 0.2, nr = 0, nc = 0, log_toggle = 0, use_outline = True, markersize = 25, zorder = 2):
    if solid_color == 'none':
      if type == 'points':
        if self.type == 'single':
          values.plot(ax = self.ax, facecolor = 'none', edgecolor = outline_color, linewidth = linewidth_size, zorder = zorder)
        elif self.type == '1d':      
          values.plot(ax = self.ax[nr], facecolor = 'none', edgecolor = outline_color, linewidth = linewidth_size, zorder = zorder)
        elif self.type == '2d':      
          values.plot(ax = self.ax[nr][nc], facecolor = 'none', edgecolor = outline_color, linewidth = linewidth_size, zorder = zorder)
      elif type == 'polygons':
        if self.type == 'single':
          values.plot(ax = self.ax, facecolor = 'none', edgecolor = outline_color, linewidth = linewidth_size, zorder = zorder)
        elif self.type == '1d':      
          values.plot(ax = self.ax[nr], facecolor = 'none', edgecolor = outline_color, linewidth = linewidth_size, zorder = zorder)
        elif self.type == '2d':      
          values.plot(ax = self.ax[nr][nc], facecolor = 'none', edgecolor = outline_color, linewidth = linewidth_size, zorder = zorder)
    elif solid_color == 'scaled':
      float_num = values[key].notnull()
      all_val = values[key]
      if value_lim == 'None':
        value_lim = (min(values[key]), max(values[key]))
      
      if np.min(all_val[float_num]) < 0.0 and np.max(all_val[float_num]) > 0.0:
        if log_toggle == 1:
          cmap = pl.cm.Reds
        else:
          cmap = pl.cm.spring
      else:
        cmap = pl.cm.RdBu
      my_cmap = cmap(np.arange(cmap.N))
      my_cmap[:,-1] = np.linspace(0.9, 1, cmap.N)
      if colorscale == 'none':
        my_cmap = ListedColormap(my_cmap)
      else:
        my_cmap = ListedColormap(sns.color_palette(colorscale).as_hex())


      if type == 'points':
        if self.type == 'single':
          if use_outline:
            values.plot(ax = self.ax, column = key, cmap = my_cmap, edgecolor = 'black', linewidth = 0.25, marker = 'o', markersize = markersize, alpha = solid_alpha, vmin = value_lim[0], vmax = value_lim[1], zorder = zorder)
          else:
            values.plot(ax = self.ax, column = key, cmap = my_cmap, linewidth = 0.25, marker = 'o', markersize = markersize, alpha = solid_alpha, vmin = value_lim[0], vmax = value_lim[1], zorder = zorder)
        elif self.type == '1d':      
          values.plot(ax = self.ax[nr], column = key, cmap = my_cmap, edgecolor = 'black', linewidth = 0.25, marker = 'o', markersize = 2.5, alpha = solid_alpha, vmin = value_lim[0], vmax = value_lim[1])
        elif self.type == '2d':      
          values.plot(ax = self.ax[nr][nc], column = key, cmap = my_cmap, edgecolor = 'black', linewidth = 0.25, markersize = markersize, alpha = solid_alpha, vmin = value_lim[0], vmax = value_lim[1])
      elif type == 'polygons':
        if self.type == 'single':
          values.plot(ax = self.ax, column = key, cmap = my_cmap, vmin = value_lim[0], vmax = value_lim[1], edgecolor = outline_color, linewidth = linewidth_size, zorder = zorder)
        elif self.type == '1d':      
          values.plot(ax = self.ax[nr], column = key, cmap = my_cmap, vmin = value_lim[0], vmax = value_lim[1], edgecolor = outline_color, linewidth = linewidth_size, zorder = zorder)
        elif self.type == '2d':      
          values.plot(ax = self.ax[nr][nc], column = key, cmap = my_cmap, vmin = value_lim[0], vmax = value_lim[1], edgecolor = outline_color, linewidth = linewidth_size, zorder = zorder)
    else:
      if type == 'points':
        if self.type == 'single':
          values.plot(ax = self.ax, color = solid_color, edgecolor = 'black', linewidth = 0.5, alpha = solid_alpha, marker = 'o', markersize = markersize, zorder = zorder)
        elif self.type == '1d':      
          values.plot(ax = self.ax[nr], color = solid_color, edgecolor = 'black', linewidth = 0.5, alpha = solid_alpha, marker = 'o', markersize = markersize, zorder = zorder)
        elif self.type == '2d':      
          values.plot(ax = self.ax[nr][nc], color = solid_color, edgecolor = 'black', linewidth = 0.5, alpha = solid_alpha, markersize = markersize, zorder = zorder)
      elif type == 'polygons':
        if self.type == 'single':
          if solid_color == 'default':
            values.plot(ax = self.ax)
          else:
            values.plot(ax = self.ax, color = solid_color, alpha = solid_alpha, edgecolor = outline_color, linewidth = linewidth_size, zorder = zorder)
        elif self.type == '1d':      
          values.plot(ax = self.ax[nr], color = solid_color, alpha = solid_alpha, zorder = zorder)
        elif self.type == '2d':      
          values.plot(ax = self.ax[nr][nc], color = solid_color, alpha = solid_alpha, zorder = zorder)

  def add_colorbar(self, coords, label_loc, label_key, title_name, colorscale = 'none', nr = 0, nc = 0):
    if self.type == 'single':   
      fig = self.ax.get_figure()
    elif self.type == '2d':      
      fig = self.ax[nr][nc].get_figure()
    cax = fig.add_axes([coords[0], coords[1], coords[2], coords[3]])
    cmap = pl.cm.RdBu
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:,-1] = np.linspace(0.9, 1, cmap.N)
    if colorscale == 'none':
      my_cmap = ListedColormap(my_cmap)
    else:
      my_cmap = ListedColormap(sns.color_palette(colorscale).as_hex())
    sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=plt.Normalize(vmin=label_loc[0], vmax=label_loc[-1]))
    # fake up the array of the scalar mappable. Urgh...
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax)
    cax.set_title(title_name, fontsize = 12, weight = 'bold', fontname = 'Gill Sans MT')
    cbar.set_ticks(label_loc)
    cbar.ax.set_yticklabels(label_key, fontsize = 12, weight = 'bold', fontname = 'Gill Sans MT')
    

  def format_plot(self, xlim = 'None', ylim = 'None', title = '', title_loc = 'top', legend_properties = {'family':'Gill Sans MT','weight':'bold','size':14}):
    if xlim == 'None' and ylim == 'None':
      do_nothing = True
    elif ylim == 'None':
      if self.type == 'single':
        self.ax.set_xlim(xlim)
        self.ax.set_xticklabels('')
        self.ax.set_yticklabels('')
        self.ax.set_title(title)    
      elif self.type == '1d':
        for nr in range(0, self.sub_rows):
          self.ax[nr].set_xlim(xlim)
          self.ax[nr].set_xticklabels('')
          self.ax[nr].set_yticklabels('')
          self.ax[nr].set_title(title)    
      elif self.type == '2d':
        for nr in range(0, self.sub_rows):
          for nc in range(0, self.sub_cols):
            self.ax[nr][nc].set_xlim(xlim)
            self.ax[nr][nc].set_xticklabels('')
            self.ax[nr][nc].set_yticklabels('')
            self.ax[nr][nc].set_title(title, fontproperties = legend_properties)			
    else:
      if self.type == 'single':
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_xticklabels('')
        self.ax.set_yticklabels('')
        self.ax.set_title(title)
      elif self.type == '1d':
        for nr in range(0, self.sub_rows):
          self.ax[nr].set_xlim(xlim)
          self.ax[nr].set_ylim(ylim)
          self.ax[nr].set_xticklabels('')
          self.ax[nr].set_yticklabels('')
          self.ax[nr].set_title(title)        
      elif self.type == '2d':
        title_counter = 0
        for nr in range(0, self.sub_rows):
          for nc in range(0, self.sub_cols):
            self.ax[nr][nc].set_xlim(xlim)
            self.ax[nr][nc].set_ylim(ylim)
            self.ax[nr][nc].set_xticklabels('')
            self.ax[nr][nc].set_yticklabels('')
            if title_loc == 'top':
              self.ax[nr][nc].set_title(title[title_counter], fontproperties = legend_properties)
            else:
              self.ax[nr][nc].set_xlabel(title[title_counter], fontproperties = legend_properties)
            
            title_counter += 1			

  def load_batch_raster(self, project_folder, raster_id_list, raster_name_pt1, raster_name_pt2, raster_band_list, projection_string, max_bright = (100.0, 1500.0), use_gamma = 0.4, contrast_range = (0.5, 99.5), upscale_factor = 'none'):
    for grid_cell in raster_id_list:
      for date_range in raster_id_list[grid_cell]:
        file_folder = project_folder + raster_name_pt1 + grid_cell + '_' + date_range + raster_name_pt2 + '/'
        rgb_list = []
        for band_name in raster_band_list:
          rgb_list.append(raster_name_pt1 + grid_cell + '_' + date_range + raster_name_pt2 + band_name)

        self.load_sat_bands(file_folder, rgb_list, projection_string, max_bright = max_bright, use_gamma =use_gamma, contrast_range = contrast_range, upscale_factor = upscale_factor)
  
  def find_plotting_data(self, plot_name):
    vmin = 0
    vmax = 0
    markersize = 2.5
    use_log = False
    if plot_name == 'land use':
      folder_name = '/180115_NeuseHydrology_Draft2/Shapefiles/'
      data_shapefile = 'Soil_Landuse_raw.shp'
      data_layer = 'Land_Use'
      data_type = 'polygons'
      data_labels = 'features'
      xkcd_color_list = ['xkcd:beige', 'xkcd:goldenrod', 'xkcd:moss green', 'xkcd:scarlet', 'xkcd:rose pink','xkcd:reddish pink', 'xkcd:pale rose', 'xkcd:navy green', 'xkcd:light green', 'xkcd:nice blue', 'xkcd:jungle green', 'xkcd:deep sky blue', 'xkcd:canary yellow', 'xkcd:pale', 'xkcd:ocean'] 
	  
    elif plot_name == 'road closure incident type':
      folder_name = '/NCState/20180412_RoadImpacts_Infrastructure_data/20180412_RoadImpacts_Infrastructure_data/Road_Impacts/'
      data_shapefile = 'NCDOT_Incidents_Counties.shp'
      data_layer = 'Incident_T'
      data_type = 'points'
      data_labels = 'features'
      xkcd_color_list = ['xkcd:beige', 'xkcd:goldenrod', 'xkcd:moss green', 'xkcd:scarlet', 'xkcd:rose pink','xkcd:reddish pink', 'xkcd:pale rose', 'xkcd:navy green', 'xkcd:light green', 'xkcd:nice blue', 'xkcd:jungle green', 'xkcd:deep sky blue', 'xkcd:canary yellow', 'xkcd:pale', 'xkcd:ocean'] 

    elif plot_name == 'road closure condition':
      folder_name = '/NCState/20180412_RoadImpacts_Infrastructure_data/20180412_RoadImpacts_Infrastructure_data/Road_Impacts/'
      data_shapefile = 'NCDOT_Incidents_Counties.shp'
      data_layer = 'ConditionN'
      data_type = 'points'
      data_labels = 'features'
      xkcd_color_list = ['white', 'xkcd:pale rose', 'xkcd:pale rose', 'indianred', 'indianred', 'indianred'] 

    elif plot_name == 'road closure duration':
      folder_name = '/NCState/20180412_RoadImpacts_Infrastructure_data/20180412_RoadImpacts_Infrastructure_data/Road_Impacts/'
      data_shapefile = 'NCDOT_Incidents_Counties.shp'
      data_layer = 'Duration'
      data_type = 'points'
      data_labels = 'scaled'
      xkcd_color_list = ['white', 'xkcd:pale rose', 'xkcd:pale rose', 'indianred', 'indianred', 'indianred'] 
      use_log = False
      markersize = 25
      vmin = 0
      vmax = 30
	  
    elif plot_name == 'dam status':
      folder_name = '/NCState/20180412_RoadImpacts_Infrastructure_data/20180412_RoadImpacts_Infrastructure_data/Infrastructure/'
      data_shapefile = 'NCDAMS_20130923.shp'
      data_layer = 'DAM_STATUS'
      data_type = 'points'
      data_labels = 'features'
      xkcd_color_list = ['indianred', 'indianred', 'xkcd:goldenrod', 'xkcd:beige', 'xkcd:beige','xkcd:beige', 'xkcd:beige', 'xkcd:beige', 'xkcd:beige', 'xkcd:beige', 'steelblue', 'xkcd:canary yellow', 'xkcd:rose pink'] 

    elif plot_name == 'dam hazard':
      folder_name = '/NCState/20180412_RoadImpacts_Infrastructure_data/20180412_RoadImpacts_Infrastructure_data/Infrastructure/'
      data_shapefile = 'NCDAMS_20130923.shp'
      data_layer = 'DAM_HAZARD'
      data_type = 'points'
      data_labels = 'features'
      xkcd_color_list = ['indianred', 'xkcd:goldenrod', 'xkcd:beige'] 
      markersize = 25
	  
    elif plot_name == 'dam condition':
      folder_name = '/NCState/20180412_RoadImpacts_Infrastructure_data/20180412_RoadImpacts_Infrastructure_data/Infrastructure/'
      data_shapefile = 'NCDAMS_20130923.shp'
      data_layer = 'CONDITION'
      data_type = 'points'
      data_labels = 'features'
      xkcd_color_list = ['xkcd:beige', 'xkcd:goldenrod', 'xkcd:moss green', 'xkcd:scarlet', 'xkcd:rose pink','xkcd:reddish pink', 'xkcd:pale rose', 'xkcd:navy green', 'xkcd:light green', 'xkcd:nice blue', 'xkcd:jungle green', 'xkcd:deep sky blue', 'xkcd:canary yellow', 'xkcd:pale', 'xkcd:ocean'] 

    elif plot_name == 'dam impoundment':
      folder_name = '/NCState/20180412_RoadImpacts_Infrastructure_data/20180412_RoadImpacts_Infrastructure_data/Infrastructure/'
      data_shapefile = 'NCDAMS_20130923.shp'
      data_layer = 'MAX_IMPOUN'
      data_type = 'points'
      data_labels = 'scaled'
      xkcd_color_list = ['xkcd:beige', 'xkcd:goldenrod', 'xkcd:moss green', 'xkcd:scarlet', 'xkcd:rose pink','xkcd:reddish pink', 'xkcd:pale rose', 'xkcd:navy green', 'xkcd:light green', 'xkcd:nice blue', 'xkcd:jungle green', 'xkcd:deep sky blue', 'xkcd:canary yellow', 'xkcd:pale', 'xkcd:ocean'] 
      vmin = 0
      vmax = 6
      use_log = True
      markersize = 25
    elif plot_name == 'dam discharge':
      folder_name = '/NCState/20180412_RoadImpacts_Infrastructure_data/20180412_RoadImpacts_Infrastructure_data/Infrastructure/'
      data_shapefile = 'NCDAMS_20130923.shp'
      data_layer = 'MAX_DISCHA'
      data_type = 'points'
      data_labels = 'scaled'
      xkcd_color_list = ['xkcd:beige', 'xkcd:goldenrod', 'xkcd:moss green', 'xkcd:scarlet', 'xkcd:rose pink','xkcd:reddish pink', 'xkcd:pale rose', 'xkcd:navy green', 'xkcd:light green', 'xkcd:nice blue', 'xkcd:jungle green', 'xkcd:deep sky blue', 'xkcd:canary yellow', 'xkcd:pale', 'xkcd:ocean'] 
      vmin = 0
      vmax = 5
      use_log = True
      markersize = 25
    
    return folder_name, data_shapefile, data_layer, data_type, data_labels, xkcd_color_list, vmin, vmax, use_log, markersize
	
  def load_sat_bands(self, file_folder, rgb_list, projection_string, max_bright = (100.0, 1500.0), use_gamma = 0.4, contrast_range = (0.5, 99.5), upscale_factor = 'none'):
    
    counter = 0
    for raster_name in rgb_list:
      ds = gdal.Open(file_folder + raster_name + '.tif')
      #geotransform - tells you where pixel 0 (x,y) is in the coordinate system and the scale of pixels to coordinates
      geotransform = ds.GetGeoTransform()
      ##clip the raster - note: this is done in UTM coordinates before projection
      ##so that you don't need to re-project entire contiguous US
      clip_name = raster_name + '_clipped.tif'
      #ds = gdal.Translate(clip_name, ds, projWin = [geotransform[0] + x_bound[0]*geotransform[1], geotransform[3] + y_bound[0]*geotransform[5], geotransform[0] + x_bound[1]*geotransform[1], geotransform[3] +y_bound[1]*geotransform[5]])
      output_name = file_folder + raster_name + '_projected.tif'
      ##re-project raster from UTM to LAT/LONG
      gdal.Warp(output_name, ds, dstSRS = projection_string)
      raster = rasterio.open(output_name)
      if upscale_factor == 'none':
        ind_rgb = raster.read()
      else:
        ind_rgb = raster.read(out_shape=(raster.count, int(raster.height * upscale_factor), int(raster.width * upscale_factor)), resampling=rasterio.enums.Resampling.bilinear)
      zero_mask = np.zeros(ind_rgb.shape)
      ones_mask = np.ones(ind_rgb.shape)
#      ind_rgb[ind_rgb <= 1.0] = ones_mask[ind_rgb <= 1.0]
      real_values = ind_rgb[~np.isnan(ind_rgb)]
      for contrast_range in [(0.2, 99.8) ,(0.5, 99.5), (2.0, 98.0)]:
        pLow, pHigh = np.percentile(real_values[real_values > 0.0], contrast_range)
      pLow = max_bright[0] 
      pHigh = max_bright[1] 
      ind_rgb = exposure.rescale_intensity(ind_rgb, in_range=(pLow,pHigh))
      if self.band_min[counter] == -1.0:
        self.band_min[counter], self.band_max[counter] = np.min(ind_rgb), np.max(ind_rgb)
      ind_rgb = (ind_rgb - self.band_min[counter])/(self.band_max[counter] - self.band_min[counter])
      ind_rgb = exposure.adjust_gamma(ind_rgb, gamma = use_gamma, gain = 1)
      if counter == 0:
        rgb_bands = np.zeros((4, ind_rgb.shape[1], ind_rgb.shape[2]))
      rgb_bands[counter,:,:] = ind_rgb
      
      counter = counter + 1
    
    true_value_mask = np.ones((1,ind_rgb.shape[1], ind_rgb.shape[2]))
    false_value_overlay = np.zeros((1, ind_rgb.shape[1], ind_rgb.shape[2]))
    false_value_mask = ind_rgb == 0.0
    true_value_mask[false_value_mask] = false_value_overlay[false_value_mask]
    rgb_bands[3,:,:] = true_value_mask
    image = rasterio.plot.reshape_as_image(rgb_bands)
   
    del rgb_bands
    del ind_rgb
    del true_value_mask
    del false_value_overlay
    del false_value_mask

    ##put bands into RGB order
    #image = rasterio.plot.reshape_as_image(b)
    ##give coordinates of image bands
    spatial_extent = rasterio.plot.plotting_extent(raster)
    ##plot image
    if self.type == 'single':
      self.ax.imshow(image, extent = spatial_extent)
    elif self.type == '1d':
      self.ax[nr].imshow(image, extent = spatial_extent)
    elif self.type == '2d':
      for nr in range(0, self.sub_rows):
        for nc in range(0, self.sub_cols):

          self.ax[nr][nc].imshow(image, extent = spatial_extent)

  def sizeof_fmt(self, num, suffix='B'):
      ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
      for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
          if abs(num) < 1024.0:
              return "%3.1f %s%s" % (num, unit, suffix)
          num /= 1024.0
      return "%.1f %s%s" % (num, 'Yi', suffix)

  def plot_scalar_raster(self, projection_string, raster_filename, cmap):
    ds = gdal.Open(raster_filename + '.tif')
    #geotransform - tells you where pixel 0 (x,y) is in the coordinate system and the scale of pixels to coordinates
    geotransform = ds.GetGeoTransform()
    ##clip the raster - note: this is done in UTM coordinates before projection
    ##so that you don't need to re-project entire contiguous US
    clip_name = raster_filename + '_clipped.tif'
    #ds = gdal.Translate(clip_name, ds, projWin = [geotransform[0] + x_bound[0]*geotransform[1], geotransform[3] + y_bound[0]*geotransform[5], geotransform[0] + x_bound[1]*geotransform[1], geotransform[3] +y_bound[1]*geotransform[5]])
    output_name = raster_filename + '_projected.tif'
    ##re-project raster from UTM to LAT/LONG
    gdal.Warp(output_name, ds, dstSRS = projection_string)
    raster = rasterio.open(output_name)
    ind_rgb = raster.read()
    band_min, band_max = np.min(ind_rgb), np.max(ind_rgb)
    ind_rgb = (ind_rgb * 0.8 + (band_max - band_min) * 0.2)      
    image = rasterio.plot.reshape_as_image(ind_rgb)
    spatial_extent = rasterio.plot.plotting_extent(raster)
    self.ax.imshow(image, extent = spatial_extent, cmap=cmap, vmin = band_max*(-0.33), vmax = band_max)

  def add_colorbar_offmap(self, colorbar_title, colorbar_labels):
    self.fig.subplots_adjust(right=0.9)
    cbar_ax = self.fig.add_axes([0.9187, 0.15, 0.025, 0.7])
    sm = plt.cm.ScalarMappable(cmap=pl.cm.gnuplot, norm=plt.Normalize(vmin=0, vmax=100))
    clb1 = plt.colorbar(sm, cax = cbar_ax, ticks=[0, 50, 100])
    clb1.ax.set_yticklabels(['0', '1', '2']) 
    clb1.ax.invert_yaxis()
    clb1.ax.tick_params(labelsize=20)
    clb1.ax.set_ylabel(colorbar_title, rotation=90, fontsize = 14, fontname = 'Gill Sans MT', fontweight = 'bold')
    for item in clb1.ax.yaxis.get_ticklabels():
      item.set_fontname('Gill Sans MT')  
      item.set_fontsize(12)


  def add_inset_figure(self, map_shape_state, box_lim_x, box_lim_y, inset_lim_x, inset_lim_y, epsg_num, shapefile2 = 'none', use_ocean = False, inset_location_number = 3):

    p2 = Polygon([(box_lim_x[0], box_lim_y[0]), (box_lim_x[1], box_lim_y[0]), (box_lim_x[1], box_lim_y[1]), (box_lim_x[0], box_lim_y[1])])
    df1 = gpd.GeoDataFrame({'geometry': p2, 'df1':[1,1]})
    df1.crs = {'init' :'epsg:'+ str(epsg_num)}
    axins = inset_axes(self.ax, width = '25%', height = '25%', loc = inset_location_number, bbox_to_anchor=(0,0,1,1), bbox_transform=self.ax.transAxes)
    map_shape_state.plot(ax = axins, facecolor = 'beige', edgecolor = 'black', linewidth = 1.0, alpha = 0.6, zorder = 5)
    if shapefile2 == 'none':
      p3 = Polygon([(inset_lim_x[0], inset_lim_y[0]), (inset_lim_x[1], inset_lim_y[0]), (inset_lim_x[1], inset_lim_y[1]), (inset_lim_x[0], inset_lim_y[1])])
      df2 = gpd.GeoDataFrame({'geometry': p3, 'df2':[1,1]})
      if use_ocean:
        df2.plot(ax = axins, facecolor = 'steelblue', edgecolor = 'steelblue', linewidth = 1.0, alpha = 1.0, zorder = 1)
      
    df1.plot(ax = axins, facecolor = 'indianred', alpha = 0.6,zorder = 25)
    #extent = map_shape_state.bounds
    extent = df2.bounds
    extent = pd.DataFrame(extent)
    extent = extent.reset_index()
    index = 0
    x_extent = np.max(extent.loc[:, 'maxx']) - np.min(extent.loc[:, 'minx'])
    y_extent = np.max(extent.loc[:, 'maxy']) - np.min(extent.loc[:, 'miny'])
    xl = np.min(extent.loc[:, 'minx']) - x_extent/8.0
    xr = np.max(extent.loc[:, 'maxx']) + x_extent/8.0
    by = np.min(extent.loc[:, 'miny']) - y_extent/8.0
    uy = np.max(extent.loc[:, 'maxy']) + y_extent/8.0
    axins.set_xlim(xl, xr)
    axins.set_ylim(by, uy)
    axins.set_xticklabels('')
    axins.set_yticklabels('')      



def add_ocean(self, filename, x_left):
  coastline = gpd.read_file(project_folder + '/CALFEWS_shapes/coastline/hj484bt5758.shp')
  geo_list = []
  for x in coastline['geometry']:
    newcords = list(x.coords)
    counter = 0
    polygon_list = []
    start_loc = newcords[0]
    polygon_list.append((x_left, start_loc[1]))
    for y in newcords:
      polygon_list.append((y[0], y[1]))
    polygon_list.append((x_left, y[1]))
    polygon_list.append((x_left, start_loc[1]))
    p2 = Polygon(polygon_list)
    geo_list.append(p2)
  values = range(len(geo_list))
  df1 = gpd.GeoDataFrame(values, geometry = geo_list)
  df1.crs = {'init' :'epsg:4326'}
  df1.plot(ax = self.ax)

