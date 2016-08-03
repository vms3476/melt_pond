import cv2
import numpy
import gdal
import utm
import os
import osr
from xml.dom.minidom import parse
from matplotlib import pyplot as plt


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
    classIm = numpy.asarray(im.GetRasterBand(1).ReadAsArray()).astype('int')
    rows, cols = classIm.shape
    pondIm = (classIm == 3)
    waterIm = (classIm == 2)
    iceIm = (classIm == 1)
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

    #"""
    # display binary image with pond pixels = 1
    classIm = (classIm==3)
    classImScaled = (classIm / classIm.max() * 255).astype(numpy.uint8)
    classImResized = cv2.resize(classImScaled,(rows/10, cols/10))
    pondImScaled = (pondIm / pondIm.max() * 255).astype(numpy.uint8)
    pondImResized = cv2.resize(pondImScaled, (rows / 10, cols / 10))
    #cv2.namedWindow('class image', cv2.WINDOW_NORMAL)
    #cv2.imshow('class image', classImResized)
    #cv2.waitKey(0)
    #"""



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
    print ' dimensions of pond im: ', pondIm.shape

    gridRows = 20
    gridCols = 20

    rowInc = numpy.floor(rows * 1.0 / gridRows).astype('int') # row increment, number of image rows per grid pixel row
    colInc = numpy.floor(cols * 1.0 / gridCols).astype('int') # col increment, number of image columns per grid pixel column

    print 'row increment: ', rowInc
    print 'col increment: ', colInc

    gridIm = numpy.zeros((gridRows,gridCols)) # keeps track of # of pond class pixels in area
    fractionIm = numpy.zeros((gridRows,gridCols)) #keeps track of average pond fraction in area

    print 'gridIm dimensions: ', gridIm.shape

    i = 0
    j = 0
    for r in range(0,gridRows):
        #print 'current row of grid image: ', r
        for c in range(0,gridCols):

            #print 'current i value: ', i       # debugging
            #print 'current j value: ', j
            #print 'current r value: ', r
            #print 'current c value: ', c
            #print 'i + rowInc: ', str(i + rowInc)
            #print 'j + colInc: ', str(j + colInc)
            #print 'sum of pond pixels in this region: ', numpy.sum(pondIm[i:i + rowInc, j:j + colInc])
            #print '--------'


            # check for the last row or column
            if r == (gridRows-1):
                gridIm[r, c] = numpy.sum(pondIm[i:, j:j + colInc])
                pondCount = pondIm[i:, j:j + colInc]
                iceCount = iceIm[i:, j:j + colInc]
                fraction = (pondCount * 1.0 / ( pondCount + iceCount))
                fractionIm[r,c] = numpy.mean(numpy.mean(fraction))

            elif c == (gridCols-1):
                gridIm[r, c] = numpy.sum(pondIm[i:i + rowInc, j:])
                pondCount = pondIm[i:, j:j + colInc]
                iceCount = iceIm[i:, j:j + colInc]
                fraction = (pondCount * 1.0 / ( pondCount + iceCount))
                fractionIm[r,c] = numpy.mean(numpy.mean(fraction))

            else:
                gridIm[r, c] = numpy.sum(pondIm[i:i + rowInc, j:j + colInc])
                pondCount = numpy.sum(pondIm[i:i + rowInc, j:j + colInc])
                print 'pond count at current region: ', pondCount
                iceCount = numpy.sum(iceIm[i:i + rowInc, j:j + colInc])
                print 'ice count at current region: ', iceCount
                fraction = (pondCount * 1.0 / ( pondCount + iceCount))
                print 'fraction at current region: ', fraction
                fractionIm[r,c] = fraction
                print ' --------- '

            j += colInc
        i += rowInc
        j = 0

    #print fractionIm

    # scale the grid image for display
    gridImScaled = (gridIm / gridIm.max() * 255).astype(numpy.uint8)
    gridImResized = cv2.resize(gridImScaled, (rows / 10, cols / 10))
    #cv2.namedWindow('grid image', cv2.WINDOW_NORMAL)
    #cv2.imshow('grid image', gridImResized)
    #cv2.waitKey(0)

    fig= plt.figure()
    latTicks = numpy.linspace(latLL, latUL, 9)
    lonTicks = numpy.linspace(lonLL, lonLR, 9)

    print 'lats: ', latTicks
    print 'lons: ', lonTicks

    plt.xticks(lonTicks)
    plt.yticks(latTicks)
    plt.title('Binary Class Image, 1 = Melt Pond')
    plt.xlabel('Longitude [decimal degrees] W')
    plt.ylabel('Latitude [decimal degrees] N')
    plt.imshow(classImResized, cmap='gray') #, extent=[lonLL,lonLR,latLL,latUL],) # show original class map highlighting ponds

    gridFig = plt.figure()
    plt.imshow(gridIm,interpolation='none')         # show the binned class map
    plt.title('Number of Pond Class Pixels per Gridded Region')
    plt.colorbar()

    fractionFig = plt.figure()
    plt.imshow(fractionIm,interpolation='none')         # show the binned class map
    plt.colorbar()

    plt.show()