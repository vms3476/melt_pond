import numpy
from spectral import *
import time
import glob
import os
import gdal
from osgeo import gdal
from PIL import Image
import spectral.io.envi as envi


def decision_tree(sr,
                  bThresh,
                  gradientThresh,
                  nirThresh,
                  landsat='8'):

    rows, cols, bands = sr.shape
    print 'dimensions of surface reflectance image cube: ', rows, cols, bands


    if landsat == 8:        # OLI
        blue = sr[:,:,1]
        green = sr[:,:,2]
        red = sr[:,:,3]
        nir = sr[:,:,4]

    else:  #landsat == 5 or 7 :      # TM or EMT+
        blue = sr[:,:,0]
        green = sr[:,:,1]
        red = sr[:,:,2]
        nir = sr[:, :, 3]




    classIm = numpy.zeros([rows, cols])
    unclass = numpy.ones([rows, cols], dtype=bool)

    # Decision 1: if b2, blue < bThresh, then it's water.
    classIm[ blue < bThresh ] = 2
    unclass [ blue < bThresh ] = False

    # Decision 2: check red-green gradient, b4-b3.
    # If > gradientThresh, then it's ice.
    gradient = red - green
    classIm[ unclass & (gradient > gradientThresh)] = 1
    unclass[ gradient > gradientThresh ] = False

    # Decision 3: check for bright NIR reflectance.
    # If b5, NIR  above nirThresh, it's ice. If below, it's pond.
    classIm[ nir > nirThresh] = 1
    unclass[ nir > nirThresh] = False

    # Remaining pixels are classified as ponds
    classIm [ unclass ] = 3

    return classIm


def stack_scale_mask(fileDir, scale=0.0001):

    # reads all 7 OIL surface reflectance files in fileDir, scales to units of
    # reflectance, outputs a single numpy array. Also outputs masks for cloud
    # and Landsat fill and a base filename for naming subsequent outputs

    os.chdir(fileDir)
    for i, file in enumerate(glob.glob('*band*.tif')):
        print 'current file: ', file
        currentBand = gdal.Open(file)
        currentBandArray = numpy.asarray(currentBand.GetRasterBand(1).ReadAsArray())
        if i==0:
            rows, cols = currentBandArray.shape
            bands = len(glob.glob('*band*.tif'))
            srCube = numpy.zeros([rows, cols, bands])
        srCube[:,:,i] = currentBandArray * scale

    cfmaskFile = gdal.Open(file[:-12] + 'cfmask.tif')
    cfmaskArray = numpy.array(cfmaskFile.GetRasterBand(1).ReadAsArray())
    cfmask = (cfmaskArray == 2) + (cfmaskArray == 4)
    cfmask = numpy.invert(cfmask)
    # Black fill pixels with value -9999 in sr image and 255 in cfmaskFile
    bmask = (cfmaskArray != 255)

    baseFilename = file[:-13]

    landsatNumber = int(baseFilename[2])

    return srCube, cfmask, bmask, baseFilename, landsatNumber


def timer(start,end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print '%d:%02d:%02d' % (hours, minutes, seconds)


def pixel2coord(col, row):
    """Returns global coordinates to pixel center using base-0 raster index"""
    xp = a * col + b * row + a * 0.5 + b * 0.5 + c
    yp = d * col + e * row + d * 0.5 + e * 0.5 + f
    return(xp, yp)







if __name__ == '__main__':

    ## Read imagery, convert to proper form
    startTime = time.time()

    fileDir = '/Users/vscholl/Documents/melt_pond/data/sr/path80row8/LC80820082015186-SC20160712151000'

    scale = 0.0001

    srCube, cfmask, bmask, baseFilename, landsatNumber = stack_scale_mask(fileDir, scale)
    print 'landsat number: ', landsatNumber
    endTime = time.time()
    print 'Time elapsed to stack and scale SR and create masks:' # ~3min
    timer(startTime, endTime)


    ## Classification

    # Decision tree: define thresholds
    bThresh = 0.2
    gradientThresh = -0.1
    nirThresh = 0.25
    print 'decision thresholds: blue = ', str(bThresh), ', ', \
          'gradient =  ', str(gradientThresh), ', ', \
          'nir = ', str(nirThresh)

    print 'Performing decision tree classification...'
    treeStartTime = time.time()
    classIm = decision_tree(srCube, bThresh, gradientThresh, nirThresh, landsatNumber)
    treeEndTime = time.time()
    print 'Time elapsed during decision tree:' # ~4min
    timer(treeStartTime, treeEndTime)

    # Apply cloud and landsat fill masks to class image
    classImMasked = bmask * cfmask * classIm
    rows, cols = classImMasked.shape
    print 'dimensions of classImMasked: ', classImMasked.shape








    ## Statistics

    # Determine number of pixels assigned as each material
    unclassCount = numpy.sum(classImMasked == 0)
    iceCount = numpy.sum(classImMasked == 1)
    waterCount = numpy.sum(classImMasked == 2)
    pondCount = numpy.sum(classImMasked == 3)

    print '# unclassified pixels = ', unclassCount
    print '# ice class pixels = ', iceCount
    print '# water class pixels = ', waterCount
    print '# pond class pixels = ', pondCount

    # Compute pond fraction (defined as the percent of total ice area)
    pondFraction = (pondCount * 1.0 / ( pondCount + iceCount)) * 100
    print 'Pond Fraction using decision tree for current image: %.4f' % pondFraction
    totalEndTime = time.time()
    print 'Total elapsed time: '
    timer(startTime, totalEndTime)

    # Save stat info anf classification image
    outDir = '/Users/vscholl/Documents/melt_pond/data/classification/' + baseFilename + '/'
    hdrFilename = baseFilename + 'decision_tree_classification_test.hdr'
    statFilename = baseFilename + '_stats.txt'

    if not os.path.exists(outDir):
        os.makedirs(outDir)

    f = open(outDir + statFilename, 'w') # write basic stat info to file
    f.write('pond fraction \t' + str(pondFraction) + '\n')
    f.write('unclass count \t' + str(unclassCount) + '\n')
    f.write('ice count \t' + str(iceCount) + '\n')
    f.write('water count \t' + str(waterCount) + '\n')
    f.write('pond count \t' + str(pondCount) + '\n')
    f.close()

    ## to read the stat text file:
    #f = open(outDir + statFilename, 'r')
    #lines = f.read().splitlines()
    #labels = []
    #stats = []
    #for line in lines:
    #    label, stat = line.strip().split('\t')
    #    labels.append(label)
    #    stats.append(stat)
    #print labels
    #print stats

    # define metadata parameters for classification header file
    metadata = {'lines': cols,
                'samples': rows,
                'bands': 1,
                'data type': 1,
                'classes': 4,
                'byte order': 0,
                'header offset': 0,
                'description': 'Decision Tree Initial Classification Result',
                'file type': 'ENVI Classification'}

    # set class names and colors for display in ENVI
    classNames =  ['Unclassified', 'Ice', 'Water', 'Pond']
    unclassColor = [0, 0, 0]
    iceColor = [255, 255, 255]
    waterColor = [0, 0, 190]
    pondColor = [35, 178, 139]
    classColors = unclassColor + iceColor + waterColor + pondColor

    envi.save_classification(outDir + hdrFilename,
                             classImMasked,
                             metadata=metadata,
                             class_names=classNames,
                             class_colors=classColors,
                             force=True)


    # Create 2d histogram image based on lat/long


    # convert pixel coordinate to lat/long






