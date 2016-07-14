import numpy
from spectral import *
import time
import glob
import os
import gdal
from PIL import Image
import spectral.io.envi as envi


def decision_tree(sr,
                  bThresh,
                  gradientThresh,
                  nirThresh):

    rows, cols, bands = sr.shape
    print 'dimensions of surface reflectance image cube: ', rows, cols, bands

    blue = sr[:,:,1]
    green = sr[:,:,2]
    red = sr[:,:,3]
    nir = sr[:,:,4]

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


def stack_scale(fileDir, scale=0.0001):

    # reads all 7 OIL surface reflectance files in fileDir, scales to units of
    # reflectance, outputs a single numpy array

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

    return srCube


def timer(start,end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print '%d:%02d:%02d' % (hours, minutes, seconds)

if __name__ == '__main__':


    ## Read imagery, convert to proper form
    startTime = time.time()
    fileDir = '/Users/vscholl/Documents/melt_pond/data/sr/LC80860102015166-SC20160615172321'
    scale = 0.0001

    srCube = stack_scale(fileDir, scale)
    endTime = time.time()
    print 'Time elapsed to stack and scale SR:'   # ~0:02:50
    timer(startTime, endTime)


    # os.chdir(fileDir)
    #
    # # Read all seven SR bands and combine into single image cube
    # scale = 0.0001 # scale factor to convert 16bit DN to refl
    # for i, file in enumerate(glob.glob('*band*.tif')):
    #     print 'current file: ', file
    #     currentBand = gdal.Open(file)
    #     currentBandArray = numpy.asarray(currentBand.GetRasterBand(1).ReadAsArray())
    #     if i==0:
    #         rows, cols = currentBandArray.shape
    #         bands = len(glob.glob('*band*.tif'))
    #         srCube = numpy.zeros([rows, cols, bands])
    #     srCube[:,:,i] = currentBandArray * scale



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
    classIm = decision_tree(srCube, bThresh, gradientThresh, nirThresh)
    treeEndTime = time.time()
    print 'Time elapsed during decision tree:'
    timer(treeStartTime, treeEndTime)


    ## Create and apply masks

    print 'Creating and applying masks for cloud and Landsat fill...'
    maskStartTime = time.time()

    # CF mask for cloud masking in L8 scenes
    # 2 = cloud shadow, 4 = cloud, assign as False.
    cfmaskFile = gdal.Open(file[:-12] + 'cfmask.tif')
    cfmaskArray = numpy.array(cfmaskFile.GetRasterBand(1).ReadAsArray())
    cfmask = (cfmaskArray == 2) + (cfmaskArray == 4)
    cfmask = numpy.invert(cfmask)

    # Black fill pixels with value -9999 in sr image and 255 in cfmaskFile
    bmask = (cfmaskArray != 255)

    # Apply cloud and landsat fill masks to class image
    classImMasked = bmask * cfmask * classIm

    print 'dimensions of classImMasked: ', classImMasked.shape
    print 'Time elapsed during mask processes:'
    maskEndTime = time.time()
    timer(maskStartTime, maskEndTime)

    ## Statistics

    # Validate classification compared to known values
    # 152
    #print 'class at point (6001,4065) should be 1 = ', classImMasked[6001,4065]
    #print 'class at point (5997,3974) should be 2= ', classImMasked[5997,3974]
    #print 'class at point (6181,3842) should be 3= ', classImMasked[6181,3842]

    # Determine number of pixels assigned as each material
    iceCount = numpy.sum(classImMasked == 1)
    waterCount = numpy.sum(classImMasked == 2)
    pondCount = numpy.sum(classImMasked == 3)
    unclassCount = numpy.sum(classImMasked == 0)

    print '# ice class pixels = ', iceCount
    print '# water class pixels = ', waterCount
    print '# pond class pixels = ', pondCount
    print '# unclassified pixels = ', unclassCount

    # Compute pond fraction (defined as the percent of total ice area)
    pondFraction = (pondCount * 1.0 / ( pondCount + iceCount)) * 100
    print 'Pond Fraction using decision tree for current image: %.4f' % pondFraction


    print 'Total elapsed time: '
    totalEndTime = time.time()
    timer(startTime, totalEndTime)


    # Create 2d histogram image based on lat/long


    ## Write results to file

    # Save as classification image
    outDir = '/Users/vscholl/Documents/melt_pond/data/classification/' + file[:-13] + '/'
    hdrFilename = file[:-13] + 'decision_tree_classification_test.hdr'

    if not os.path.exists(outDir):
        os.makedirs(outDir)

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










