import numpy
from spectral import *
import time
import glob
import os
import gdal
from osgeo import gdal
import spectral.io.envi as envi
import xml.etree.ElementTree


def decision_tree(sr,
                  bThresh,
                  gradientThresh,
                  nirIceThresh,
                  nirShallowPondThresh,
                  landsat):

    """Performs decision tree classification.

    Assigns input surface reflectance pixels to one of five possible class
    types (ice, water, shallow melt pond, deep melt pond, or unclassified) 
    based on the following logic:
    1. Checks if the blue band value is less than the blue threshold value.
        If so, this pixel is classified as water.
    2. For the remaining pixels, checks if the gradient value
        (red band - green band) is greater than the gradient threshold value.
        If so, this pixel is classified as ice.
    3. For the remaining pixels, checks if the near infrared (NIR) band value is
        greater than the NIR threshold. If so, this pixel is classified as ice.
    4. For the remaining pixels, checks if the NIR band value is less than the
        NIR shallow pond threshold. If so, this pixel is classified as a deep
        pond. If not, this pixel is classified as a shallow melt pond. 

    Args:
        sr: surface reflectance numpy array
        bThresh: decimal floating point number [0, 1.0]
        nirIceThresh: decimal floating point number [0, 1.0]
        gradientThresh: decimal floating point number [0, 1.0]
        nirShallowPondThresh: decimal floating point number [0, 1.0]
        landsat: number indicating landsat mission used to generate the
            input surface reflectance data, of type <str>. default value is '8'.
            this is used

    Returns:
        classIm: numpy array containing integer values representing the
            assigned class of each surface reflectance pixel:
                1 = ice
                2 = water
                3 = shallow melt pond
                4 = deep melt pond
                0 = unclassified

    """

    rows, cols, bands = sr.shape

    # assign each band to a variable.
    # different index for L8 vs. L5/7 due to coastal band
    if landsat == 8:                 # OLI
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
    classIm[ unclass & (nir > nirIceThresh)] = 1
    unclass[ nir > nirIceThresh] = False

    # Decision 4: check nir band; determine deep melt ponds
    classIm[ unclass & (nir < nirShallowPondThresh)] = 4
    unclass[ nir < nirShallowPondThresh] = False

    # Remaining pixels are classified as shallow ponds
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

def pond_fraction_per_latitude(xmlFile, classImMasked, latIncrements): 

    pondIm = (classIm == 3) + (classIm == 4)
    iceIm = (classIm == 1)

    # parse the Landsat xml file to determine lat long for UL and LR pixels
    root = xml.etree.ElementTree.parse(xmlFile).getroot()
    subroot = root.getchildren()
    global_metadata = subroot[0]
    elements = global_metadata.getchildren()
    for i, j in enumerate(elements): 
        if 'location' in j.attrib: 
            if j.attrib['location'] == 'UL': 
                latUL = float(j.attrib['latitude'])
                #lonUL = float(j.attrib['longitude'])
            else: 
                latLR = float(j.attrib['latitude'])
                #lonLR = float(j.attrib['longitude'])

    # determine lat range and values for UR,LL corners
    latLL = latLR
    latUR = latUL
    latRange = numpy.asarray([min(latUL, latLL, latUR, latLR),max(latUL, latLL, latUR, latLR)])
    latIncrements = numpy.arange(latMin, latMax + latIncrement, latIncrement)
    print 'lat increments defined by user: ', latIncrements
    print 'lat range of current image: ', latRange


    rEnd = 0 # counter for starting row index
    data = False # boolean variable indicates whether image data exists in latitude increment

    pondCounts = numpy.zeros((len(latIncrements)-1))
    iceCounts = numpy.zeros((len(latIncrements)-1))

    for i in range(0, len(latIncrements)-1): 
        print 'i = ', i
        lat1 = latIncrements[i]     # low end of current latitude increment
        lat2 = latIncrements[i+1]   # high end of current latitude increment
        
        # determine first latitude increment range where there is image data
        if (latRange[0] >= lat1) and (latRange[0] < lat2): 
            print 'image data starts to exist between ', lat1, ' and ', lat2

            # interpolate to find the number of rows corresponding to the current latitude increment
            rStart = int(numpy.round(numpy.interp(lat2,latRange,numpy.asarray([0,rows-1]))))
            pondCounts[i] = (numpy.sum(pondIm[(rows - rStart) : (rows - rEnd), :]))
            iceCounts[i] = (numpy.sum(iceIm[(rows - rStart) : (rows - rEnd), :]))
            print 'pond count from ', (rows - rStart), ' to ', rows - rEnd, ' = ', pondCounts[i]
            rEnd = rStart
            data = True
        
        # check to see if it's the last latitude increment where image data exists
        elif (latRange[1] >= lat1) and (latRange[1] < lat2):
            
            print 'last increment that image data exists between: ', lat1, ' and ', lat2
            #rStart = int(numpy.round(numpy.interp(lat2,latRange,numpy.asarray([0,rows-1]))))
            pondCounts[i] = (numpy.sum(pondIm[0 : (rows - rEnd), :]))
            iceCounts[i] = (numpy.sum(iceIm[0 : (rows - rEnd), :]))
            print 'pond count from ', 0, ' to ', rows-rEnd, ' = ', pondCounts[i]
            break
            # enter in None for any following latitude columns for this scene


        elif data == True: 
            print 'current increment: ', lat1, ' and ', lat2
            rStart = int(numpy.round(numpy.interp(lat2,latRange,numpy.asarray([0,rows-1]))))
            pondCounts[i] = (numpy.sum(pondIm[(rows - rStart) : (rows - rEnd), :]))
            iceCounts[i] = (numpy.sum(iceIm[(rows - rStart) : (rows - rEnd), :]))
            print 'pond count from ', (rows - rStart), ' to ', (rows-rEnd), ' = ', pondCounts[i]
            rEnd = rStart

        else: 
            print 'No image data in current latitude increment: ', lat1, ' and ', lat2

    print 'pondCounts: ', pondCounts
    print 'iceCounts: ', iceCounts

    # convert lists to arrays for pond fraction calculation
    pondCounts = numpy.asarray(pondCounts)
    iceCounts = numpy.asarray(iceCounts)
    pondFractions = (pondCounts * 1.0 / ( pondCounts + iceCounts)) * 100
    print 'pondFractions: ', pondFractions

    return pondFractions






if __name__ == '__main__':


    ### USER-DEFINED INPUT #############################################################################################
    
    # surface reflectance main directory
    mainDir = '/Users/vscholl/Documents/melt_pond/data/sr/'
    
    # subdirectory containing the desired images to process
    processingDir = 'path80row8/'
    
    # directory to place classification output files
    classificationDir = '/Users/vscholl/Documents/melt_pond/data/classification/' + processingDir
    
    # filename for text file with classification statistics / information
    statFilename = 'decision_tree_classification_stats.txt' 


    # define thresholds as an array where each row contains the decision thresholds 
    # blue, gradient, NIR ice, and NIR shallow pond
    # for L5, then L7, and finally L8 (in that order)

                               #b,      #gradient,  #nirIce, #nirShallow
    thresholds = numpy.array(([0.32,     -0.1,       0.3,        0.065,   # L5     
                               0.2,     -0.1,       0.25,       0.070,   # L7
                               0.2,     -0.1,       0.25,       0.025])) # L8

                              # [0.3,     -0.1,       0.3,        0.065,   # L5     
                              #  0.2,     -0.1,       0.25,       0.070,   # L7
                              #  0.2,     -0.1,       0.25,       0.025])) # L8

    # when a classification stats file already exists in the output directory, 
    # this boolean variable is used to either create a new text file (True) or append
    # lines to the existing text file with the filename specified above (False)
    createNewStatFile = False


    # specify desired latitude min, max, and increment 
    latMin = 70
    latMax = 75 
    latIncrement = 1.0

    ####################################################################################################################




    # Begin processing Landsat surface reflectance data in specified directory
    startTime = time.time()
    os.chdir(mainDir + processingDir)
    dirList = glob.glob('L*')
    counter = 0 # used to determine the first iteration
    for fileDir in dirList:
        print 'currently processing ', fileDir
        
        ## Read imagery, convert to units of reflectance in a single array
        scale = 0.0001
        srCube, cfmask, bmask, baseFilename, landsat = stack_scale_mask(fileDir, scale)
        xmlFile = mainDir + processingDir + fileDir + '/' + baseFilename+ '.xml'
        print 'xml file: ', xmlFile
        root = xml.etree.ElementTree.parse(xmlFile).getroot()
        print 'ROOT: ', root

        print 'landsat number: ', landsat
        endTime = time.time()
        print 'Time elapsed to stack and scale SR and create masks:' # ~3min
        timer(startTime, endTime)


        if not os.path.exists(classificationDir + baseFilename):
            os.makedirs(classificationDir + baseFilename)
            createNewStatFile = True

        # for first iteration, if createNewStatFile has been assigned to True,
        # create a new text file and write the column titles and threshold values.
        # otherwise, append to the existing file.
        if counter == 0 and createNewStatFile:
            f = open(classificationDir + statFilename, 'w')
            f.write('Landsat ID \t' +
                    'Blue threshold: \t' +
                    'Gradient threshold: \t' +
                    'NIR Ice threshold: \t' +
                    'NIR Shallow Pond threshold: \t' +
                    'Class 0: Unclassified # Pixels \t' +
                    'Class 1: Ice/Snow # Pixels \t' +
                    'Class 2: Water # Pixels \t' +
                    'Class 3: Shallow Melt Pond \t' +
                    'Class 4: Deep Melt Pond \t' +
                    'Shallow Pond Fraction per Scene ' + 
                    '(Shallow Pond / (Total Pond + Ice)) \t' + 
                    'Deep Pond Fraction per Scene ' + 
                    '(Deep Pond / (Total Pond + Ice)) \t' +
                    'Total Pond Fraction per Scene (Pond / (Pond + Ice)) \t')
            
            
            latIncrements = numpy.arange(latMin, latMax + latIncrement, latIncrement)
            print 'lat increments: ', latIncrements
            # add columns to text file for each latitude range
            for i in range(0, len(latIncrements)-1): 
                f.write(str(latIncrements[i]) + ' - ' + str(latIncrements[i+1]) + ' N \t')

            f.write('\n') 
            f.close()



        ## Classification

        # for a single combination of thresholds, reshape the array for proper indexing
        if len(thresholds.shape)== 1: 
            thresholds = thresholds.reshape((1,thresholds.shape[0]))
            print 'shape of thresholds array: ', thresholds.shape

        for i in range(0,thresholds.shape[0]): # row dimension in threshold array

            # Decision tree: assign thresholds based on input
            if landsat == 5:  # water is [abnormally] brighter in L5 imagery
                l = 0   # column dimension within threshold array

            elif landsat == 7:
                l = 1

            else:  # water is darker in L8 and L7 imagery
                l = 2

            bThresh = thresholds[i, 0 + 4 * l]
            gradientThresh = thresholds[i, 1 + 4 * l]
            nirIceThresh = thresholds[i, 2 + 4 * l]
            nirShallowPondThresh = thresholds[i, 3 + 4 * l]

            print 'current decision thresholds: blue = ', str(bThresh), ', ', \
                  'gradient =  ', str(gradientThresh), ', ', \
                  'nir ice = ', str(nirIceThresh), ', nir shallow pond = ', \
                   str(nirShallowPondThresh)

            # check to see if this image and threshold combination already exists
            line = baseFilename +'\t'+ str(bThresh) +'\t'+ str(gradientThresh)+'\t'+ \
                   str(nirIceThresh)+'\t'+ str(nirShallowPondThresh)

            if line in open(classificationDir + statFilename).read():
                print 'This image has already been processed with the current decision thresholds.'

            else:
                print 'Performing decision tree classification...'
                treeStartTime = time.time()
                classIm = decision_tree(srCube, bThresh, gradientThresh, nirIceThresh, nirShallowPondThresh, landsat)
                treeEndTime = time.time()
                print 'Time elapsed during decision tree:'
                timer(treeStartTime, treeEndTime)

                # Apply cloud and landsat fill masks to class image
                classImMasked = bmask * cfmask * classIm
                rows, cols = classImMasked.shape



                ## Statistics

                # Determine number of pixels assigned as each material
                unclassCount = numpy.sum(classImMasked == 0)
                iceCount = numpy.sum(classImMasked == 1)
                waterCount = numpy.sum(classImMasked == 2)
                shallowPondCount = numpy.sum(classImMasked == 3)
                deepPondCount = numpy.sum(classImMasked == 4)
                totalPondCount = shallowPondCount + deepPondCount

                # Compute total pond fraction (defined as the percent of total ice area)
                totalPondFraction = (totalPondCount * 1.0 / ( totalPondCount + iceCount)) * 100
                shallowPondFraction = (shallowPondCount * 1.0 / ( totalPondCount + iceCount)) * 100
                deepPondFraction = (deepPondCount * 1.0 / ( totalPondCount + iceCount)) * 100

                # Save stat info and classification image
                # create a string to name class images with specificity
                threshStr = ('blue' + str(bThresh) + '_gradient' +  str(gradientThresh) + '_nirIce' + str(nirIceThresh) + '_nirShallowPond' + str(nirShallowPondThresh)).replace('.','')
                classImageFilename = '_decision_tree_class_image_' + threshStr + '.hdr'
                hdrFilename = baseFilename + classImageFilename


                # calculate pond fraction as a function of latitude
                pondFractions = pond_fraction_per_latitude(xmlFile, classImMasked, latIncrements)


                f = open(classificationDir + statFilename, 'a') # open file to append lines

                f.write(baseFilename + '\t' +
                    str(bThresh) + '\t' +
                    str(gradientThresh) + '\t' +
                    str(nirIceThresh) + '\t' +
                    str(nirShallowPondThresh) + '\t' +
                    str(unclassCount) + '\t' +
                    str(iceCount) + '\t' +
                    str(waterCount) + '\t' +
                    str(shallowPondCount) + '\t' +
                    str(deepPondCount) + '\t' +
                    str(shallowPondFraction) + '\t' +
                    str(deepPondFraction) + '\t' +
                    str(totalPondFraction) + '\t')

                for i in range(0, len(latIncrements)-1): 
                    f.write(str(pondFractions[i]) + ' \t')
                f.write('\n') 

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
                classNames =  ['Unclassified', 'Ice', 'Water', 'Shallow Pond', 'Deep Pond']
                unclassColor = [128, 128, 128]  # gray
                iceColor = [255, 255, 255]      # white
                waterColor = [0, 0, 0]          # black
                shallowPondColor = [0,190,190]  # light blue-green
                deepPondColor = [54, 76, 189]   # dark blue
                classColors = unclassColor + iceColor + waterColor + shallowPondColor + deepPondColor

                envi.save_classification(classificationDir + baseFilename + '/' + hdrFilename,
                                     classImMasked,
                                     metadata=metadata,
                                     class_names=classNames,
                                     class_colors=classColors,
                                     force=True)
                
                endTime = time.time()
                print 'Time elapsed for decision tree & class image generation:' # ~4min
                timer(startTime, endTime)
                f.close()

            os.chdir(mainDir + processingDir)
            counter += 1 # advance to next row in output text file




