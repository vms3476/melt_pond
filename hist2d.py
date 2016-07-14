import cv2
import numpy
import gdal


def hist2d(classIm):

    rows,cols = classIm.shape

    histIm = numpy.zeros([rows,cols])

    return histIm

if __name__ == '__main__':

    classImFile = '/Users/vscholl/Documents/melt_pond/data/classification/LC80800082015172LGN00/LC80800082015172LGN00decision_tree_classification_test.img'
    im = gdal.Open(classImFile)
    classIm = numpy.asarray(im.GetRasterBand(1).ReadAsArray())
    rows, cols = classIm.shape

    print 'image dimensions: ' rows, cols

    classImScaled = (classIm / classIm.max() * 255).astype(numpy.uint8)
    classImResized = cv2.resize(classImScaled,(rows/10, cols/10))
    cv2.namedWindow('class image', cv2.WINDOW_NORMAL)
    cv2.imshow('class image', classImResized)
    cv2.waitKey(0)