import numpy
from spectral import *
import time
import glob
import os
import gdal
from osgeo import gdal
import spectral.io.envi as envi


def decision_tree(sr,
                  bThresh,
                  gradientThresh,
                  nirThresh,
                  landsat='8'):

    """Performs decision tree classification.

    Assigns input surface reflectance pixels to one of four possible class
    types (ice, water, pond, or unclassified) based on the following logic:
    1. Checks if the blue band value is less than the blue threshold value.
        If so, this pixel is classified as water.
    2. For the remaining pixels, checks if the gradient value
        (red band - green band) is greater than the gradient threshold value.
        If so, this pixel is classified as ice.
    3. For the remaining pixels, checks if the near infrared (NIR) band value is
        greater than the NIR threshold. If so, this pixel is classified as ice.
        If not, this pixel is classified as melt pond.

    Args:
        sr: surface reflectance numpy array
        bThresh:
        gradientThresh:
        nirThresh:
        landsat: number indicating landsat mission used to generate the
            input surface reflectance data, of type <str>. default value is '8'.
            this is used

    Returns:
        classIm: numpy array containing integer values representing the
            assigned class of each surface reflectance pixel:
                1 = ice
                2 = water
                3 = pond
                0 = unclassified

    """

    rows, cols, bands = sr.shape
    print 'dimensions of surface reflectance image cube: ', rows, cols, bands

    # assign each band to a variable.
    # different index for L8 vs. L5/7 due to coastal band
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

    classIm = numpy.zeros([rows, cols], dtype=numpy.uint8)
    unclass = numpy.ones([rows, cols], dtype=bool)

    # Decision 1: check blue band; determine water pixels
    classIm[ blue < bThresh ] = 2
    unclass [ blue < bThresh ] = False

    # Decision 2: check red-green gradient; determine ice pixels
    gradient = red - green
    classIm[ unclass & (gradient > gradientThresh)] = 1
    unclass[ gradient > gradientThresh ] = False

    # Decision 3: check nir band; determine ice pixels
    classIm[ nir > nirThresh] = 1
    unclass[ nir > nirThresh] = False

    # Remaining pixels are classified as ponds
    classIm [ unclass ] = 3

    return classIm


def stack_scale_mask(fileDir, scale=0.0001):

    """Stacks L# bands into single array, scales to units of reflectance

    Args:
        fileDir: directory containing surface reflectance images (band*.tif)
        scale: factor to multiply SR data by to yield units of decimal refl [0,1]

    Returns:
        srCube: numpy array containing scaled surface reflectance image data
        cfmask: binary mask created from the cfmask file, also included within the
            landsat surface reflectance data product. It is used to
            mask clouds during classification. Values of 2 (cloud shadow) and
            4 (cloud) are assigned a value of 1. Landsat fill pixels have a value of
            255 in this product, which is used to create bmask.
        bmask = binary mask indicating Landsat fill pixels.
        baseFilename = landsat ID used for naming files
        landsat = integer value (of type <str>) indicating Landsat mission # (5,7,8)
    """

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
    landsat = int(baseFilename[2])

    return srCube, cfmask, bmask, baseFilename, landsat


def timer(start,end):
    """Returns the time elapsed in hh:mm:ss"""
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print '%d:%02d:%02d' % (hours, minutes, seconds)


def pixel2coord(col, row):
    """Returns global coordinates to pixel center using base-0 raster index"""
    xp = a * col + b * row + a * 0.5 + b * 0.5 + c
    yp = d * col + e * row + d * 0.5 + e * 0.5 + f
    return(xp, yp)

# def write_stats_file(classImMasked):
#
#     # Determine number of pixels assigned as each material
#     unclassCount = numpy.sum(classImMasked == 0)
#     iceCount = numpy.sum(classImMasked == 1)
#     waterCount = numpy.sum(classImMasked == 2)
#     pondCount = numpy.sum(classImMasked == 3)
#
#     print '# unclassified pixels = ', unclassCount
#     print '# ice class pixels = ', iceCount
#     print '# water class pixels = ', waterCount
#     print '# pond class pixels = ', pondCount
#
#     # Compute pond fraction (defined as the percent of total ice area)
#     pondFraction = (pondCount * 1.0 / ( pondCount + iceCount)) * 100
#     print 'Pond Fraction using decision tree for current image: %.4f' % pondFraction
#
#     # Save stat info and classification image
#     hdrFilename = baseFilename + classImageFilename
#
#     if not os.path.exists(classificationDir + baseFilename):
#         os.makedirs(classificationDir + baseFilename)
#
#     if counter == 0: # for first iteration, write column titles and threshold values
#         f = open(classificationDir + statFilename, 'w')
#         f.write('Landsat ID \t' +
#                 'Blue threshold: \t' +
#                 'NIR threshold: \t' +
#                 'Gradient threshold: \t' +
#                 'Class 0: Unclassified # Pixels \t' +
#                 'Class 1: Ice/Snow # Pixels \t' +
#                 'Class 2: Water # Pixels \t' +
#                 'Class 3: Melt Pond \t' +
#                 'Pond Fraction per Scene (Pond / (Pond + Ice)) \n')
#
#     f.write(baseFilename + '\t' +
#             str(bThresh) + '\t' +
#             str(nirThresh) + '\t' +
#             str(gradientThresh) + '\t' +
#             str(unclassCount) + '\t' +
#             str(iceCount) + '\t' +
#             str(waterCount) + '\t' +
#             str(pondCount) + '\t' +
#             str(pondFraction) + '\n')
#
#     ## to read the stat text file:
#     #f = open(outDir + statFilename, 'r')
#     #lines = f.read().splitlines()
#     #labels = []
#     #stats = []
#     #for line in lines:
#     #    label, stat = line.strip().split('\t')
#     #    labels.append(label)
#     #    stats.append(stat)
#     #print labels
#     #print stats
#
#     # define metadata parameters for classification header file
#     metadata = {'lines': cols,
#                 'samples': rows,
#                 'bands': 1,
#                 'data type': 1,
#                 'classes': 4,
#                 'byte order': 0,
#                 'header offset': 0,
#                 'description': 'Decision Tree Initial Classification Result',
#                 'file type': 'ENVI Classification'}
#
#     # set class names and colors for display in ENVI
#     classNames =  ['Unclassified', 'Ice', 'Water', 'Pond']
#     unclassColor = [0, 0, 0]
#     iceColor = [255, 255, 255]
#     waterColor = [0, 0, 190]
#     pondColor = [35, 178, 139]
#     classColors = unclassColor + iceColor + waterColor + pondColor
#
#     envi.save_classification(classificationDir + baseFilename + '/' + hdrFilename,
#                             classImMasked,
#                             metadata=metadata,
#                             class_names=classNames,
#                             class_colors=classColors,
#                             force=True)
#
#     endTime = time.time()
#     print 'Time elapsed for decision tree & class image generation:' # ~4min
#     timer(startTime, endTime)
#
#     os.chdir(mainDir + processingDir)
#     counter += 1 # advance to next row in output text file
#
#     f.close()





if __name__ == '__main__':




    ### USER-DEFINED INPUT ###
    mainDir = '/Users/vscholl/Documents/melt_pond/data/sr/'
    processingDir = 'threshold_testing/'
    classificationDir = '/Users/vscholl/Documents/melt_pond/data/classification/' + processingDir
    classImageFilename = '_decision_tree_classification_image.hdr'
    statFilename = 'pond_stats.txt' # filename for text file with pond stats info
    ###########################




    # Begin processing Landsat surface reflectance data in specified directory
    startTime = time.time()
    os.chdir(mainDir + processingDir)
    dirList = glob.glob('L*')
    counter = 0
    for fileDir in dirList:
        print 'currently processing ', fileDir
        ## Read imagery, convert to proper form
        scale = 0.0001
        srCube, cfmask, bmask, baseFilename, landsat = stack_scale_mask(fileDir, scale)
        print 'landsat number: ', landsat
        endTime = time.time()
        print 'Time elapsed to stack and scale SR and create masks:' # ~3min
        timer(startTime, endTime)



        ## Classification

        # Decision tree: define thresholds
        if landsat == 5:  # water is [abnormally] brighter in L5 imagery
            bThresh = 0.32
            nirThresh = 0.3

        else:  # water is darker in L8 and L7 imagery
            bThresh = 0.2
            nirThresh = 0.25

        gradientThresh = -0.1

        print 'decision thresholds: blue = ', str(bThresh), ', ', \
              'gradient =  ', str(gradientThresh), ', ', \
              'nir = ', str(nirThresh)

        print 'Performing decision tree classification...'
        treeStartTime = time.time()
        classIm = decision_tree(srCube, bThresh, gradientThresh, nirThresh, landsat)
        treeEndTime = time.time()
        print 'Time elapsed during decision tree:'
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

        # Save stat info and classification image
        hdrFilename = baseFilename + classImageFilename

        if not os.path.exists(classificationDir + baseFilename):
            os.makedirs(classificationDir + baseFilename)

        if counter == 0: # for first iteration, write column titles and threshold values
            f = open(classificationDir + statFilename, 'w')
            f.write('Landsat ID \t' +
                'Blue threshold: \t' +
                'NIR threshold: \t' +
                'Gradient threshold: \t' +
                'Class 0: Unclassified # Pixels \t' +
                'Class 1: Ice/Snow # Pixels \t' +
                'Class 2: Water # Pixels \t' +
                'Class 3: Melt Pond \t' +
                'Pond Fraction per Scene (Pond / (Pond + Ice)) \n')

        f.write(baseFilename + '\t' +
                str(bThresh) + '\t' +
                str(nirThresh) + '\t' +
                str(gradientThresh) + '\t' +
                str(unclassCount) + '\t' +
                str(iceCount) + '\t' +
                str(waterCount) + '\t' +
                str(pondCount) + '\t' +
                str(pondFraction) + '\n')

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

        envi.save_classification(classificationDir + baseFilename + '/' + hdrFilename,
                                 classImMasked,
                                 metadata=metadata,
                                 class_names=classNames,
                                 class_colors=classColors,
                                 force=True)
        endTime = time.time()
        print 'Time elapsed for decision tree & class image generation:' # ~4min
        timer(startTime, endTime)

        os.chdir(mainDir + processingDir)
        counter += 1 # advance to next row in output text file

    f.close()


