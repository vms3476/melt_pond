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


# def pixel2coord(col, row):
#     """Returns global coordinates to pixel center using base-0 raster index"""
#     xp = a * col + b * row + a * 0.5 + b * 0.5 + c
#     yp = d * col + e * row + d * 0.5 + e * 0.5 + f
#     return(xp, yp)


if __name__ == '__main__':

    
    # L5
    classImFile = '/Users/vscholl/Documents/melt_pond/data/classification/threshold_testing_4class/LT50750092009168GLC00/LT50750092009168GLC00_decision_tree_class_image_blue03_gradient-01_nirIce03_nirShallowPond0065.img'
    xmlFile = '/Users/vscholl/Documents/melt_pond/data/sr/threshold_testing_4class/LT50750092009168-SC20160714131203/LT50750092009168GLC00.xml'

    # L7 
    classImFile = '/Users/vscholl/Documents/melt_pond/data/classification/L7test/LE70660092000169PAC00/LE70660092000169PAC00_decision_tree_class_image_blue02_gradient-01_nirIce025_nirShallowPond007.img'
    xmlFile = '/Users/vscholl/Documents/melt_pond/data/sr/L7test/LE70660092000169-SC20160715104425/LE70660092000169PAC00.xml'

    classImFile = '/Users/vscholl/Documents/melt_pond/data/classification/path80row8_4class/LE70810082001180AGS00/LE70810082001180AGS00_decision_tree_classification_bThreshPoint32.img'
    xmlFile = '/Users/vscholl/Documents/melt_pond/data/sr/path80row8/LE70810082001180-SC20160712153336/LE70810082001180AGS00.xml'

    # L8
    #classImFile = '/Users/vscholl/Documents/melt_pond/data/classification/threshold_testing_4class/LC80800082015172LGN00/LC80800082015172LGN00_decision_tree_class_image_blue02_gradient-01_nirIce025_nirShallowPond0025.img'
    #xmlFile = '/Users/vscholl/Documents/melt_pond/data/sr/threshold_testing_4class/LC80800082015172-SC20160712150214/LC80800082015172LGN00.xml'

    im = gdal.Open(classImFile)
    classIm = numpy.asarray(im.GetRasterBand(1).ReadAsArray()).astype('int')
    rows, cols = classIm.shape
    pondIm = (classIm == 3) + (classIm == 4)    # 3 = shallow melt pond, 4 = deep melt pond
    waterIm = (classIm == 2)
    iceIm = (classIm == 1)
    print '--------------------------------------'
    print 'class image dimensions: ', rows, cols
    print 'total # pond pixels: ', numpy.sum(pondIm)
    print 'total # ice pixels: ', numpy.sum(iceIm)

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

    # parse the Landsat xml file to determine lat long for UL and LR pixels
    import xml.etree.ElementTree
    root = xml.etree.ElementTree.parse(xmlFile).getroot()
    subroot = root.getchildren()
    global_metadata = subroot[0]
    elements = global_metadata.getchildren()
    for i, j in enumerate(elements): 
        if 'location' in j.attrib: 
            if j.attrib['location'] == 'UL': 
                latUL = float(j.attrib['latitude'])
                lonUL = float(j.attrib['longitude'])
            else: 
                latLR = float(j.attrib['latitude'])
                lonLR = float(j.attrib['longitude'])
    
    # determine lat,lon range and values for UR,LL corners
    latLL = latLR
    lonLL = lonUL
    latUR = latUL
    lonUR = lonLR
    latRange = latUL - latLL
    lonRange = lonLR - lonLL

    print 'Upper Left lat, lon = ', latUL, lonUL
    print 'Lower Right lat, lon = ', latLR, lonLR
    print 'Lower Left lat, lon = ', latLL, lonLL
    print 'Upper Right lat, lon = ', latUR, lonUR
    print 'Lat range: ', str(min(latUL, latLL, latUR, latLR)), ' to ', str(max(latUL, latLL, latUR, latLR))
    print 'Lon range: ', str(min(lonUL, lonLL, lonUR, lonLR)), ' to ', str(max(lonUL, lonLL, lonUR, lonLR))

    # determine pond fraction as a function of latitude
    #latIncrement = float(raw_input('Enter latitude increment: '))
    latIncrement = 1.0

    print 'Computing pond fraction over the following latitude increments: '
    latStart = numpy.round(min(latUL, latLL, latUR, latLR),1)
    print 'Minimum latitude: ', latStart
    latEnd = numpy.round(max(latUL, latLL, latUR, latLR),1)
    print 'Maximum latitude: ', latEnd
    print 'Lat Range: ', latEnd - latStart
    latIncrements = numpy.arange(latStart, latEnd + latIncrement, latIncrement)
    print 'increments: ', latIncrements

    print 'row increment:  ', rows / len(latIncrements)
    rowIncrement = rows / len(latIncrements)

    pondCounts = []
    iceCounts = []
    for i in range(0,len(latIncrements)-1): 
        print 'row range: ', str(i*rowIncrement), ' - ', str((i+1) * rowIncrement)
        print 'latitude range: ', str(latIncrements[i]), ' - ', str(latIncrements[i+1])
        pondCounts.append(numpy.sum(pondIm[i*rowIncrement : (i+1) * rowIncrement, :]))
        iceCounts.append(numpy.sum(iceIm[i*rowIncrement : (i+1) * rowIncrement, :]))
        print 'pondCounts: ', pondCounts
        print 'iceCounts: ', iceCounts


    # for the final increment of image rows, sum all the way to the end of the image. 
    pondCounts[-1] = numpy.sum(pondIm[i*rowIncrement : rows, :])
    iceCounts[-1] = numpy.sum(iceIm[i*rowIncrement : rows, :])

    # convert lists to arrays for pond fraction calculation
    pondCounts = numpy.asarray(pondCounts)
    iceCounts = numpy.asarray(iceCounts)
    pondFractions = (pondCounts * 1.0 / ( pondCounts + iceCounts)) * 100

    # reverse the order of pondFractions since the lowest index starts at the top of 
    # the image but the lowest latitude value starts at the bottom of the image 
    pondFractions = pondFractions[::-1]

    print 'total # of pond pixels according to the pondCounts list: ', sum(pondCounts)
    print 'total # of ice pixels according to the iceCounts list: ', sum(iceCounts)
    print 'pond fractions: ', pondFractions

    print '---------------'
    print 'range 1: ', latIncrements[0], ' - ', latIncrements[1]
    print pondFractions[0]
    print 'range 2: ', latIncrements[1], ' - ', latIncrements[2]
    print pondFractions[1]
    print 'range 3: ', latIncrements[2], ' - ', latIncrements[3]
    print pondFractions[2]
    print '---------------------------------------------------------'


    # write to text file in same directory as classification stats 
    

    latRange = numpy.asarray([min(latUL, latLL, latUR, latLR),max(latUL, latLL, latUR, latLR)])
    latMax = 76
    latMin = 69
    latIncrements = numpy.arange(latMin, latMax + latIncrement, latIncrement)
    print 'lat increments defined by user: ', latIncrements
    print 'lat range of current image: ', latRange




    rEnd = 0 # counter for starting row index
    data = False # boolean variable indicates whether image data exists in latitude increment

    pondCounts = []
    iceCounts = []

    for i in range(0, len(latIncrements)-1): 
        print 'i = ', i
        lat1 = latIncrements[i]     # low end of current latitude increment
        lat2 = latIncrements[i+1]   # high end of current latitude increment
        
        # determine first latitude increment range where there is image data
        if (latRange[0] >= lat1) and (latRange[0] < lat2): 
            print 'image data starts to exist between ', lat1, ' and ', lat2

            # interpolate to find the number of rows corresponding to the current latitude increment
            rStart = int(numpy.round(numpy.interp(lat2,latRange,numpy.asarray([0,rows-1]))))
            pondCounts.append(numpy.sum(pondIm[(rows - rStart) : (rows - rEnd), :]))
            iceCounts.append(numpy.sum(iceIm[(rows - rStart) : (rows - rEnd), :]))

            print 'pond count from ', (rows - rStart), ' to ', rows - rEnd, ' = ', pondCounts[i]
            rEnd = rStart
            data = True
        
        # check to see if it's the last latitude increment where image data exists
        elif (latRange[1] >= lat1) and (latRange[1] < lat2):
            
            print 'last increment that image data exists between: ', lat1, ' and ', lat2
            #rStart = int(numpy.round(numpy.interp(lat2,latRange,numpy.asarray([0,rows-1]))))
            pondCounts.append(numpy.sum(pondIm[0 : (rows - rEnd), :]))
            iceCounts.append(numpy.sum(iceIm[0 : (rows - rEnd), :]))
            print 'pond count from ', 0, ' to ', rows-rEnd, ' = ', pondCounts[i]
            break
            # enter in None for any following latitude columns for this scene


        elif data == True: 
            print 'current increment: ', lat1, ' and ', lat2
            rStart = int(numpy.round(numpy.interp(lat2,latRange,numpy.asarray([0,rows-1]))))
            pondCounts.append(numpy.sum(pondIm[(rows - rStart) : (rows - rEnd), :]))
            iceCounts.append(numpy.sum(iceIm[(rows - rStart) : (rows - rEnd), :]))
            print 'pond count from ', (rows - rStart), ' to ', (rows-rEnd), ' = ', pondCounts[i]
            rEnd = rStart

        else: 
            print 'No image data in current latitude increment: ', lat1, ' and ', lat2
            pondCounts.append(0)
            iceCounts.append(0)

    print 'pondCounts: ', pondCounts
    print 'iceCounts: ', iceCounts

    # convert lists to arrays for pond fraction calculation
    pondCounts = numpy.asarray(pondCounts)
    iceCounts = numpy.asarray(iceCounts)
    pondFractions = (pondCounts * 1.0 / ( pondCounts + iceCounts)) * 100
    print 'pondFractions: ', pondFractions


    # # create 2d histogram image showing where the pond pixels occur
    # print ' dimensions of pond im: ', pondIm.shape
    # gridRows = 3
    # gridCols = 4
    # rowInc = numpy.floor(rows * 1.0 / gridRows).astype('int') # row increment, number of image rows per grid pixel row
    # colInc = numpy.floor(cols * 1.0 / gridCols).astype('int') # col increment, number of image columns per grid pixel column

    # print 'row increment: ', rowInc
    # print 'col increment: ', colInc

    # gridIm = numpy.zeros((gridRows,gridCols)) # keeps track of # of pond class pixels in area
    # fractionIm = numpy.zeros((gridRows,gridCols)) #keeps track of average pond fraction in area

    # print 'gridIm dimensions: ', gridIm.shape

    # i = 0
    # j = 0
    # for r in range(0,gridRows):
    #     #print 'current row of grid image: ', r
    #     for c in range(0,gridCols):

    #         # check for the last row or column
    #         if r == (gridRows-1):
    #             gridIm[r, c] = numpy.sum(pondIm[i:, j:j + colInc])
    #             pondCount = pondIm[i:, j:j + colInc]
    #             iceCount = iceIm[i:, j:j + colInc]
    #             fraction = (pondCount * 1.0 / ( pondCount + iceCount))
    #             fractionIm[r,c] = numpy.mean(numpy.mean(fraction))

    #         elif c == (gridCols-1):
    #             gridIm[r, c] = numpy.sum(pondIm[i:i + rowInc, j:])
    #             pondCount = pondIm[i:, j:j + colInc]
    #             iceCount = iceIm[i:, j:j + colInc]
    #             fraction = (pondCount * 1.0 / ( pondCount + iceCount))
    #             fractionIm[r,c] = numpy.mean(numpy.mean(fraction))

    #         else:
    #             gridIm[r, c] = numpy.sum(pondIm[i:i + rowInc, j:j + colInc])
    #             pondCount = numpy.sum(pondIm[i:i + rowInc, j:j + colInc])
    #             #print 'pond count at current region: ', pondCount
    #             iceCount = numpy.sum(iceIm[i:i + rowInc, j:j + colInc])
    #             #print 'ice count at current region: ', iceCount
    #             fraction = (pondCount * 1.0 / ( pondCount + iceCount))
    #             #print 'fraction at current region: ', fraction
    #             fractionIm[r,c] = fraction
    #             #print ' --------- '

    #         j += colInc
    #     i += rowInc
    #     j = 0

    # #print fractionIm

    # # scale the grid image for display
    # gridImScaled = (gridIm / gridIm.max() * 255).astype(numpy.uint8)
    # gridImResized = cv2.resize(gridImScaled, (rows / 10, cols / 10))
    # #cv2.namedWindow('grid image', cv2.WINDOW_NORMAL)
    # #cv2.imshow('grid image', gridImResized)
    # #cv2.waitKey(0)

    # fig= plt.figure()

    # plt.title('Binary Class Image, 1 = Melt Pond')
    # plt.xlabel('Longitude [decimal degrees] W')
    # plt.ylabel('Latitude [decimal degrees] N')
    # plt.imshow(classImResized, cmap='gray') #, extent=[lonLL,lonLR,latLL,latUL],) # show original class map highlighting ponds

    # gridFig = plt.figure()
    # plt.imshow(gridIm,interpolation='none')         # show the binned class map
    # plt.title('Number of Pond Class Pixels per Gridded Region')
    # plt.colorbar()

    # fractionFig = plt.figure()
    # plt.imshow(fractionIm,interpolation='none')         # show the binned class map
    # plt.colorbar()

    # plt.show()