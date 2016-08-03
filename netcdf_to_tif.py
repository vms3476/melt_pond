import gdal
import os

ncFilename = '/Users/vscholl/Documents/melt_pond/data/seadas/path80row8/LC80800082015172LGN00.L2OC.nc'

info = os.system('gdalinfo ' + ncFilename)

im = gdal.Open('NETCDF:' + ncFilename +':') #+ ':geophysical_data')
print im