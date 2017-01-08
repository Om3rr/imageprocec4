import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread as imread
from skimage import color

#constants
COLOR = 2
GRAY = 1
PNG = 4
RGB = 3
ROW = 0
COL = 1
PIXEL = 2
PALLETE = 255
BINS = 256
JUMP_FOR_Y = 3
MIN_ITERS = 1
MIN_COLOR = 1

grayCmap = plt.get_cmap("gray")
png2jpgMatrix = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
rgb2yiqMatrix = np.array([[0.299     ,  0.587     ,  0.114     ],
                         [0.59590059, -0.27455667, -0.32134392],
                         [0.21153661, -0.52273617,  0.31119955]])
rgb2yiqMatrixPNG = np.array([[0.299, 0.587, 0.114],
                          [0.59590059, -0.27455667, -0.32134392],
                          [0.21153661, -0.52273617, 0.31119955]])
yiq2rgbMatrix = np.linalg.inv(rgb2yiqMatrix)

pictures = ["12.jpg","5.jpg","jerusalem.jpg", "Low Contrast.jpg", "monkey.jpg"]
# pictures = ["jerusalem.jpg"]
'''
3.1 answers.
'''
'''
image read, get a filename - path to an image and representation
and returns an np.array - as a 3dimensional array while a-col is the
picture`s rows, b-col is the picture`s cols and c-col is the picture`s pixel
representation
representation = 1 will return array represent grayscale picture
representation = 2 will return array represent rgb picture
'''
def read_image(filename, representation):
    if representation not in [1,2]:
        print("The number should be 1 or 2 please fix it :)")
        return
    try:
        im = imread(filename).astype(np.float32)
    except Exception:
        print("Cant find the specified picture")
        return

    im /= PALLETE #from 0 to 1
    if(im.shape[PIXEL] is PNG): # handle png
        im = png2jpg(im)

    #handle grayscale
    if representation is GRAY :
        return makeItGray(im)


    return im
'''
inside function that dot-product every vector in pic 3rd column in
YIQ matrix first line this will represent the grayscale convertion
of a picture.'''
def makeItGray(pic):
    return color.rgb2gray(pic)


'''
simple function that convert png matrixes (row,col,4)
to rgb matirxes (row,col,3) by multiply by 3X4 matrix.
'''
def png2jpg(im):
    return np.dot(im,png2jpgMatrix.transpose())

'''
3.2
'''
'''
this function will open a new figure and display the given picture
representation = 1 will show grayscale picture
representation = 2 will show rgb picture
'''
def imdisplay(filename, representation):
    img = read_image(filename, representation)
    arrayDisplay(img, representation)

'''
sub function that gets img as array and representation and will
display creates new subplot and display the picture on this plt.'''
def arrayDisplay(img, repr):
    fig, ax = plt.subplots()
    if(repr is COLOR):
        ax.imshow(img)
    else:
        ax.imshow(img, cmap=grayCmap)
# 3.3

'''
converts rgb image (represented by array
to yiq image
'''
def rgb2yiq(imRGB):
    try:
        if(imRGB.ndim is not RGB):
            print("cannot convert non RGB picture to YIQ")
            return

    except:
        raise ValueError("imRGB should be an numpy object")
    r,c,s = imRGB.shape
    newimRGB = np.dot(imRGB.reshape(r*c,s), rgb2yiqMatrix.transpose())
    return newimRGB.reshape((r,c,s))

'''
converts yiq image (represented by array
to rgb image
'''
def yiq2rgb(imYIQ):
    try:
        if (imYIQ.ndim is not RGB):
            print("cannot convert non YIQ picture to RGB")
            return
    except:
        raise ValueError("imRGB should be an numpy object")
    r, c, s = imYIQ.shape
    imYIQ = imYIQ.reshape(r * c, s)
    newimRGB = np.dot(imYIQ, yiq2rgbMatrix.transpose())
    return newimRGB.reshape((r, c, s))

# 3.4
'''
histogram equalize. this function get and image-as-array (dettype = float32)
and perform equalize base on the pictures histogram.
this function will return a list of
newPic (pic as array), OldPicHistogram, NewPicHistogram'''
def histogram_equalize(im_orig):
    isColored = im_orig.ndim is RGB
    newArray, origHist, newHist = histogramForGray(makeItGray(im_orig))
    if(isColored): # Handle colored picture
        #convert into YIQ
        origYIQ = rgb2yiq(im_orig.copy())
        origYIQ = origYIQ.reshape(-1)
        # change the Y value
        origYIQ[::JUMP_FOR_Y] = newArray.reshape(-1)
        origYIQ = origYIQ.reshape(im_orig.shape)

        #convert back into RGB and clipping the pic :)
        newArray = np.clip(yiq2rgb(origYIQ), 0, 1)

    return [newArray, origHist, newHist]

'''
calculating histogram for grayscale picture
'''
def histogramForGray(im_orig):
    origHist, array = np.histogram(im_orig * PALLETE, bins=BINS, range=(0, PALLETE))
    cmap, toMap = histogramToCMAP(origHist, im_orig.size)
    newArray = im_orig.copy() * PALLETE
    newArray = np.interp(np.floor(newArray), toMap, cmap)
    newHist, fArray = np.histogram(newArray, BINS, (0, PALLETE))
    newArray = newArray / PALLETE
    return [newArray, origHist, newHist]


def histogramToCMAP(histogram, size):
    cumsum = np.cumsum(histogram)
    cumsum = stretch((cumsum / size) * PALLETE)
    toMap = np.array(range(0, BINS))
    return cumsum, toMap

'''
this function gets an array and stretch linearly his values
from 0 to 255.'''
def stretch(array):
    delta = array[-1] - array[0]
    t = lambda x:((x-array[0])/delta)*PALLETE
    return np.floor(np.vectorize(t)(array))


# 3.5

'''
quantize algorithm that gets image, n, iterations
and return new image that quantized into n colors
 and the error for each iteration
'''
def quantize(im_orig, n_quant, n_iter):
    ## run the algorithm for gray image
    img, error = quantizeGray(makeItGray(im_orig.copy()), n_quant, n_iter)

    if(im_orig.shape[-1] is RGB): #if the image is RGB than do it`s thing
        # some conversions
        newImg = rgb2yiq(im_orig.copy()).reshape(-1)
        newImg[::JUMP_FOR_Y] = img.reshape(-1)
        img = yiq2rgb(newImg.reshape(im_orig.shape))


    return img, error


'''
this function will quantize Gray image (assuming that im is in shape[row,col]
or just in shape :)
returns the image after conversion and the error array.
'''
def quantizeGray(im, n_quant, n_iter):
    #histogram calculation
    histogram, array = np.histogram(im,bins=BINS,range=(0,1))
    n_quant -= 1

    #generate initial Z
    z = np.append([0],
        (np.cumsum(np.bincount(
        np.floor(
            np.cumsum(histogram) * (n_quant/im.size)).astype(np.uint8))))
                  -1)

    #run the algorithm (get the c_map)
    c_map, error = maxLoydAlgorithm(histogram, z, n_iter)

    #change the picture according to the cmap
    items = np.array(range(BINS))
    im = im * PALLETE
    newImgGray = np.interp(im, items, c_map) / PALLETE

    #end!
    return newImgGray , error

'''
this function represent the core of the algorithm
it will calculate the best Q and Z for this picture
for the given iterations.'''
def maxLoydAlgorithm(histogram, z, iters):
    # create and empty error array
    z = z.astype(np.float64)
    error = np.empty((iters,))
    q = np.array((z.size-1,))
    for i in range(iters):
        #perform the two
        q = calculateQ(histogram, z)
        oldz = z.copy()
        z = calculateZ(q.copy(),z)
        #calculate error
        error[i] = calculateError(histogram, q, z)
        #check for converges
        if(np.array_equal(np.around(z),np.around(oldz))):
            error[i:] = None
            return qandzToCmap(q,z), np.floor(error)
    return qandzToCmap(q,z), np.floor(error)

'''
This function generates a color map keyLocation = key, LocationVal = value
that should interp as old color -> new color map'''
def qandzToCmap(q,z):

    #creating the map
    cmap = np.empty((BINS,))
    z = np.around(z).astype(np.uint8)
    for i in range(q.shape[0]):
        cmap[np.around(z[i]):np.around(z[i + 1])+1] = q[i]
    return cmap


'''
This function will calculate Q`s for specific
histogram and z array.'''
def calculateQ(histogram, z):
    # creating variables
    z = np.around(z).astype(np.uint8)
    arrayLen = z.shape[0]
    q = np.empty((arrayLen-1,)).astype(np.float64)
    # iterating for each color choice
    for i in range(arrayLen-1):
        ar = np.array(range(z[i], z[i+1]))

        # handling the case that two bins are in the same borders.
        if(ar.size is 0):
            q[i] = q[i-1]
        currHist = histogram[z[i]:z[i+1]]
        #calculate variables
        upper = np.dot(ar,currHist.transpose())
        lower = np.sum(currHist)

        # set specific Q
        q[i] = upper/lower
    return q

'''
this function will return function that returns
'''
def getHistoFunc(histogram):
    def getSpecificVal(val):
        if(0 > val) or (val > PALLETE):
            raise ValueError
        return histogram[val]*val
    return getSpecificVal

'''
This function calculate array of Z`s for a given
Q array.
'''
def calculateZ(q, z):
    arrayLen = q.shape[0]
    q = np.around(q)
    f = lambda x,y:(x+y)/2.0
    z[0] = 0 # first val
    for i in range(arrayLen-1):
        #other vals :)
        z[i+1] = f(q[i],q[i+1])
    z[arrayLen] = PALLETE # last val
    return z


'''
this function calculate the Q,Z error
according to the picture histogram
'''
def calculateError(histogram, q, z):
    error = 0
    z = np.around(z).astype(np.uint8)
    for i in range(z.shape[0]-1):
        ar = np.array(range(z[i], z[i+1]))
        if(ar.size is 0):
            continue
        tempFunc = np.vectorize(lambda j:((q[i]-j)**2)*histogram[j])
        error += np.sum(tempFunc(ar))
    return error



