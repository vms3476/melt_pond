# PYTHON CODE 

import matplotlib.pyplot 
import matplotlib.image
import numpy
import spectral.io.envi as envi
import PIL
import gdal
import matplotlib.pyplot
import time


# read image using matplotlib
im = matplotlib.image.imread('/Users/vscholl/Downloads/BlueMarble_2005_globe.tif')
implot = matplotlib.pyplot.imshow(im)
matplotlib.pyplot.show()
rows,cols,bands = im.shape

# display a plot using matplotlib. More: http://matplotlib.org/users/pyplot_tutorial.html
plot = matplotlib.pyplot.plot([0,1,2,3])
matplotlib.pyplot.ylabel('vertical')
matplotlib.pyplot.xlabel('horizontal')
matplotlib.pyplot.title('image')
matplotlib.pyplot.axis([0, 10, 0, 100]) # set axes ranges
matplotlib.pyplot.show()

# scale for natural color display
rows, cols, bands = im.shape
print 'im shape: ', str(im.shape)
rgb = im[:, :, 2:5]
rgbScaled = (rgb * 255.0).astype('uint8')
view = matplotlib.pyplot.imshow(rgbScaled.astype('uint8'))
matplotlib.pyplot.show()

# PIL image module
from PIL import Image

# spectral python
import spectral.io.envi as envi
im = envi.open('/Users/vscholl/Downloads/sr_stacked_scaled/LC80840092015152LGN00/LC80840092015152LGN00_sr_stackedScaled.tif.hdr','/Users/vscholl/Downloads/sr_stacked_scaled/LC80840092015152LGN00/LC80840092015152LGN00_sr_stackedScaled.tif')
rgb = im[:,:,2:5]

# loop through bands
for i in range(0, bands):
    print i

# set a variable for each landsat band used in classification, for readability
coastal = im[:, :, 0].reshape([rows,cols])
blue = im[:, :, 1].reshape([rows,cols])
green = im[:, :, 2].reshape([rows,cols])
red = im[:, :, 3].reshape([rows,cols])
nir = im[:, :, 4].reshape([rows,cols])
swir1 = im[:, :, 5].reshape([rows,cols])
swir2 = im[:, :, 6].reshape([rows,cols])

# get image dimensions
rows, cols, bands = im.shape
print 'im shape: ', str( im.shape )

# set a variable for each landsat band used during the classification
# coastal = im[:, :, 0].reshape([rows, cols])
blue = im[:, :, 1].reshape([rows, cols])
green = im[:, :, 2].reshape([rows, cols])
red = im[:, :, 3].reshape([rows, cols])
nir = im[:, :, 4].reshape([rows, cols])





# LAT/LONG

### Use the affine transform parameters with GDAL to compute lat,long for any pixel
def pixel2coord(col, row):
    """Returns global coordinates to pixel center using base-0 raster index"""
    xp = a * col + b * row + a * 0.5 + b * 0.5 + c
    yp = d * col + e * row + d * 0.5 + e * 0.5 + f
    return(xp, yp)

from osgeo import gdal
tifName = '/Users/vscholl/Downloads/LC80840092015152-SC20160615172456/LC80840092015152LGN00_sr_band1.tif'
im = gdal.Open(tifName)
# unravel GDAL affine transform parameters
c, a, b, f, d, e = im.GetGeoTransform()
easting, northing = pixel2coord(10, 22)

# get image dimensions
rows = im.RasterXSize
cols = im.RasterYSize
bands =im.RasterCount
band1 = im.GetRasterBand(1)
bandType = gdal.GetDataTypeName(band1.DataType)

# use gdalinfo to get coordinate info for a tif
import os
tifName = '/Users/vscholl/Downloads/LC80840092015152-SC20160615172456/LC80840092015152LGN00_sr_band1.tif'
tifName = '/Users/vscholl/Documents/melt_pond/data/sr_stacked_scaled/LC80840092015152LGN00/LC80840092015152LGN00_sr_stackedScaled.tif'
tifName = '/Users/vscholl/Documents/melt_pond/data/sr/LC80840092015152-SC20160615172456/LC80840092015152LGN00_sr_band1.tif'
info = os.system('gdalinfo ' + tifName)

import utm
utm.to_latlon(easting, northing, 4, 'N')

# lat/long from geotiff file tutorial
tifName = '/Users/vscholl/Documents/melt_pond/data/sr/LC80840092015152-SC20160615172456/LC80840092015152LGN00_sr_band1.tif'
ds = gdal.Open(tifName)
old_cs= osr.SpatialReference()
old_cs.ImportFromWkt(ds.GetProjectionRef())
# create the new coordinate system
wgs84_wkt = """GEOGCS["WGS 84", DATUM["WGS_1984", SPHEROID["WGS 84",6378137,298.257223563, AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]]"""
new_cs = osr.SpatialReference()
new_cs .ImportFromWkt(wgs84_wkt)
# create a transform object to convert between coordinate systems
transform = osr.CoordinateTransformation(old_cs,new_cs)
width = ds.RasterXSize
height = ds.RasterYSize
gt = ds.GetGeoTransform()
minx = gt[0]
miny = gt[3] + width*gt[4] + height*gt[5]
#get the coordinates in lat long
latlong = transform.TransformPoint(10,22)
#latlong = transform.TransformPoint(x,y)


# another attempt:
import osr

def transform_utm_to_wgs84(easting, northing, zone):
    utm_coordinate_system = osr.SpatialReference()
    utm_coordinate_system.SetWellKnownGeogCS("WGS84") # Set geographic coordinate system to handle lat/lon
    is_northern = northing > 0
    utm_coordinate_system.SetUTM(zone, is_northern)

    wgs84_coordinate_system = utm_coordinate_system.CloneGeogCS() # Clone ONLY the geographic coordinate system

    # create transform component
    utm_to_wgs84_transform = osr.CoordinateTransformation(utm_coordinate_system, wgs84_coordinate_system) # (<from>, <to>)
    return utm_to_wgs84_transform.TransformPoint(easting, northing, 0) # returns lon, lat, altitude


# read MS stacked, scaled tif
# tifName = '/Users/vscholl/Downloads/sr_stacked_scaled/LC80840092015152LGN00/LC80840092015152LGN00_sr_stackedScaled.tif'
# hdrName = tifName + '.hdr'
# im = envi.open(hdrName, tifName)



# 152 file path
fileDir = '/Users/vscholl/Documents/melt_pond/data/sr/LC80840092015152-SC20160615172456/'

# write basic stat info to file
f = open(outDir + statFilename, 'w')  # write basic stat info to file
f.write('pond fraction \t' + str(pondFraction) + '\n')
f.write('unclass count \t' + str(unclassCount) + '\n')
f.write('ice count \t' + str(iceCount) + '\n')
f.write('water count \t' + str(waterCount) + '\n')
f.write('pond count \t' + str(pondCount) + '\n')
f.close()

# read basic stat info file
f = open(outDir + statFilename, 'r')
lines = f.read().splitlines()
labels = []
stats = []
for line in lines:
    label, stat = line.strip().split('\t')
    labels.append(label)
    stats.append(stat)
print labels
print stats

cv2.imshow('class image', classImScaled)
cv2.waitKey(0)



# parsing XML
from xml.dom.minidom import parse
xmlFile = '/Users/vscholl/Documents/melt_pond/data/sr/LC80800082015172-SC20160617110020/LC80800082015172LGN00.xml'
dom = parse(xmlFile)
ids  = dom.getElementsByTagName('bounding_coordinates')

import xml.etree.ElementTree
root = xml.etree.ElementTree.parse(xmlFile).getroot()
for child in root:
        print child.tag, child.attrib
        tag = child.tag
        list = tag.getchildren()
        print list

for attributes in root.iter(child.tag):
    attribute_list = attributes.getchildren()
print attribute_list


import xml.etree.ElementTree
root = xml.etree.ElementTree.parse(xmlFile).getroot()
subroot = root.getchildren()
global_metadata = subroot[0]
elements = global_metadata.getchildren()
cornerUL = elements[11].attrib
cornerLR = elements[12].attrib

latUL = float(cornerUL['latitude'])
lonUL = float(cornerUL['longitude'])
latLR = float(cornerLR['latitude'])
lonLR = float(cornerLR['longitude'])
print 'Upper Left lat, lon = ', latUL, lonUL
print 'Lower Right lat, lon = ', latLR, lonLR

latLL = latLR
lonLL = lonUL
latUR = latUL
lonUR = lonLR
print 'Lower Left lat, lon = ', latLL, lonLL
print 'Upper Right lat, lon = ', latUR, lonUR


import numpy as np
import matplotlib.pyplot as plt

n = 100000
x = np.random.standard_normal(n)
y = 2.0 + 3.0 * x + 4.0 * np.random.standard_normal(n)
plt.hexbin(x,y)

plt.show()



gdalinfo '/Users/vscholl/Documents/melt_pond/data/seadas/path80row8/LC80800082015172LGN00.L2OC.nc'

thresholds = numpy.array(([0.3, 0.3, -0.1,  # L5     # iteration 1
                           0.2, 0.25, -0.1,  # L7
                           0.2, 0.25, -0.1],  # L8

                          [0.32, 0.3, -0.1,  # L5     # iteration 2
                           0.2, 0.25, -0.1,  # L7
                           0.2, 0.25, -0.1],  # L8

                          [0.34, 0.3, -0.1,  # L5     # iteration 3
                           0.2, 0.25, -0.1,  # L7
                           0.2, 0.25, -0.1]))  # L8
for i in range(0, thresholds.shape[0]):  # row dimension in threshold array

    # Decision tree: assign thresholds based on input
    if landsat == 5:  # water is [abnormally] brighter in L5 imagery
        l = 0  # column dimension within threshold array

    elif landsat == 7:
        l = 1

    else:  # water is darker in L8 and L7 imagery
        l = 2
# netcdf to tif
ncFilename = '/Users/vscholl/Documents/melt_pond/data/seadas/path80row8/LC80800082015172LGN00.L2OC.nc'

info = os.system('gdalinfo ' + ncFilename)

im = gdal.Open('NETCDF:' + ncFilename +':') #+ ':geophysical_data')
print im

