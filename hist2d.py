import cv2
import numpy
import gdal
import utm
import os
import osr
from xml.dom.minidom import parse


def hist2d(classIm):

    rows,cols = classIm.shape

    histIm = numpy.zeros([rows,cols])

    return histIm


def pixel2coord(col, row):
    """Returns global coordinates to pixel center using base-0 raster index"""
    xp = a * col + b * row + a * 0.5 + b * 0.5 + c
    yp = d * col + e * row + d * 0.5 + e * 0.5 + f
    return(xp, yp)


if __name__ == '__main__':

    classImFile = '/Users/vscholl/Documents/melt_pond/data/classification/LC80800082015172LGN00/LC80800082015172LGN00decision_tree_classification_test.img'
    tifFile = '/Users/vscholl/Documents/melt_pond/data/sr/LC80800082015172-SC20160617110020/LC80800082015172LGN00_sr_band1.tif'
    xmlFile = '/Users/vscholl/Documents/melt_pond/data/sr/LC80800082015172-SC20160617110020/LC80800082015172LGN00.xml'

    im = gdal.Open(classImFile)
    classIm = numpy.asarray(im.GetRasterBand(1).ReadAsArray())
    rows, cols = classIm.shape
    pondIm = (classIm == 3.0).astype('uint8')
    print 'total # of pond pixels: ', numpy.sum(pondIm)
    print 'number of pond pixels in upper right quadrant: ', numpy.sum(pondIm[0:rows / 2, 0:cols / 2])

    print 'class image dimensions: ', rows, cols


    """
    # display class image
    classImScaled = (classIm / classIm.max() * 255).astype(numpy.uint8)
    classImResized = cv2.resize(classImScaled,(rows/10, cols/10))
    cv2.namedWindow('class image', cv2.WINDOW_NORMAL)
    cv2.imshow('class image', classImResized)
    cv2.waitKey(0)
    """
    """
    # display binary image with pond pixels = 1
    classImScaled = (classIm / classIm.max() * 255).astype(numpy.uint8)
    classImResized = cv2.resize(classImScaled,(rows/10, cols/10))
    cv2.namedWindow('class image', cv2.WINDOW_NORMAL)
    cv2.imshow('class image', classImResized)
    cv2.waitKey(0)
    """

    """
    # read tif file for coordinate information
    geoTif = gdal.Open(tifFile)
    print 'class image dimensions: ', geoTif.RasterXSize, geoTif.RasterYSize
    c, a, b, f, d, e = geoTif.GetGeoTransform()

    ulE, ulN = pixel2coord(0,0)
    print 'coordinate of upper left: ', ulE, ulN
    uRE, uRN = pixel2coord(0, cols-1)
    print 'coordinate of upper right: ', uRE, uRN
    llE, llN = pixel2coord(rows-1, 0)
    print 'coordinate of lower left: ', llE, llN
    lrE, lrN = pixel2coord(rows-1, cols-1)
    print 'coordinate of lower right: ', lrE, lrN

    info = os.system('gdalinfo ' + tifFile) # determine UTM zone

    import subprocess
    direct_output = subprocess.check_output('gdalinfo ' + tifFile, shell=True)
    print direct_output[20]

    prj = geoTif.GetProjection()
    print 'wkt: ', wkt

    srs = osr.SpatialReference(wkt=prj)
    if srs.IsProjected:
        print srs.GetAttrValue('projcs')
    print srs.GetAttrValue('geogcs')

    old_cs = osr.SpatialReference()
    old_cs.ImportFromWkt(geoTif.GetProjectionRef())
    print ' old cs: ', old_cs


    #ulLat, ulLon = utm.to_latlon(easting, northing, 4, 'N')
    """

    # parse the Landsat xml file to determine lat long for pixels
    import xml.etree.ElementTree
    root = xml.etree.ElementTree.parse(xmlFile).getroot()
    subroot = root.getchildren()
    global_metadata = subroot[0]
    elements = global_metadata.getchildren()
    cornerUL = elements[11].attrib
    cornerLR = elements[12].attrib
    latUL = float(cornerUL['latitude']) # record the corner lat,long coordinates
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
    latRange = latUL - latLL
    lonRange = lonLR - lonLL



    # create 2d histogram image showing where the pond pixels occur

    testIm = pondIm[0:-1,0:-1]

    gridRows = testIm.shape[0] / 10
    gridCols = testIm.shape[1] / 10

    rowInc = rows * 1.0 / gridRows # row increment, number of image rows per grid pixel row
    colInc = cols * 1.0 / gridCols # col increment, number of image columns per grid pixel column

    gridIm = numpy.zeros((gridRows,gridCols))

    print 'gridIm dimensions: ', gridIm.shape

    i = 0
    j = 0
    for r in range(0,gridRows):
        #print 'current row of grid image: ', r
        for c in range(0,gridCols):
            #print 'current col of grid image: ', c
            gridIm[r,c] = numpy.sum(pondIm[i:i+rowInc,j:j+colInc])
            #print 'sum of pond pixels in this region: ', numpy.sum(pondIm[i:i+rowInc,j:j+colInc])

            j += colInc
        i += rowInc

    print numpy.max(gridIm)

    print 'value of grid im pixel 0,0 = ', gridIm[0,0]


    # scale the grid image for display
    gridImScaled = (gridIm / gridIm.max() * 255).astype(numpy.uint8)
    gridImResized = cv2.resize(gridImScaled, (rows / 10, cols / 10))
    cv2.namedWindow('grid image', cv2.WINDOW_NORMAL)
    #cv2.imshow('grid image', gridImResized)
    #cv2.waitKey(0)


