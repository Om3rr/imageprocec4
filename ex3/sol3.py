from scipy.misc import imread
import os
from scipy.ndimage.filters import convolve
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.color import rgb2gray
DEFAULT_FILTER = np.array([0.25,0.5,0.25]).astype(np.float32)


##### 3.1 #####
'''
dumb function that getting an array size and calculating the required array
'''
def build_filter(size):
    assert(size%2==1)
    currFilter = DEFAULT_FILTER.copy()
    for i in range(int((size/2)-1)):
        currFilter = np.convolve(currFilter, DEFAULT_FILTER)
    return currFilter, currFilter.reshape((size,1)), currFilter.reshape((1,size))

'''
function that getting filter size and return
a nice blur function and the filter itself.

the blur function have an extended version of blur!
it can also blur according the expand idea that
the filter array doesnt trivial (the sum = 2)
'''
def blurFunction(filter_size):
    fil, hfil, vfil = build_filter(filter_size)
    def blur(im,expand=False):
        if(expand):
            newIm = convolve(im.copy(),hfil*2)
        else:
            newIm = convolve(im.copy(), hfil)
        if(expand):
            newIm = convolve(newIm,vfil * 2)
        else:
            newIm = convolve(newIm, vfil)
        return newIm
    return blur, fil

'''
dumb function that getting a picture and return
an expand of this picture into bigger one!'''
def expandPic(im):
    newim = np.ndarray((im.shape[0]*2,im.shape[1]*2)).astype(np.float32)
    newim.fill(0)
    newim[::2,::2] = im
    return newim


'''
this function getting a picture and reduce it
by factor of 2!
'''
def reducePic(im):
    return im[::2,::2].copy()

'''
this function getting a filter size and return a brand new
laplacian function that getting an image and returns its laplacian
image.
this sub-function reduce the picture, expand it and sub it from the
original pic
'''
def laplacianFunction(filter_size):
    blur, fi = blurFunction(filter_size)
    def expand_and_sub(im):
        im = im.copy()
        # reduce
        reduced = reducePic(blur(im))
        #expand
        expanded = blur(expandPic(reduced.copy()),True)
        r,c = im.shape
        ### special case handling when one of the sizes is odd.. ####
        im = im - expanded[0:r,0:c]
        ## returns the image to this step and the reduced to the next step.
        return im, reduced
    return expand_and_sub, fi

'''
this function designed as described in ex3
'''
def build_gaussian_pyramid(im, max_levels, filter_size):
    ## set max_levels according to the minimal of log2(rows), log2(cols), max_levels
    ## take the lower limit (cause if we reach log2(rows\cols) we cant iterate
    ##                       ( if we reach max_levels we dont need more iterations)
    max_levels = min(max_levels, int(min(math.log2(im.shape[0]), math.log2(im.shape[1])))-3)
    # getting the blur function + the filter (to return it..)
    blur, fil = blurFunction(filter_size)
    im = im.astype(np.float32)
    pyramid = [im]
    for i in range(max_levels-1): ## reduce 1 from max levels cause we already inserted im to pyramid
        #iterating over the last picture in the list..
        # for each picture we are bluring and reducing it..
        pyramid.append(reducePic(blur(pyramid[-1])))
    return pyramid, fil.copy().reshape((1,filter_size))

'''
this function designed as described in ex3
'''
def build_laplacian_pyramid(im,max_levels,filter_size):
    ## calculate the max_levels
    im = im.astype(np.float32)
    max_levels = min(max_levels, int(min(math.log2(im.shape[0]), math.log2(im.shape[1]))) - 3)
    ##
    expand_and_sub, fi = laplacianFunction(filter_size)
    lapPyramid = []
    for i in range(max_levels-1):
        subbed, im = expand_and_sub(im)
        lapPyramid.append(subbed)
    lapPyramid.append(im)
    return lapPyramid, fi.copy().reshape((1,filter_size))




#### 3.2 ####
'''
function that create a function that blur the picture
with the given filter vector
'''
def getBlur(filter_vec):
    r = filter_vec.shape[len(filter_vec.shape)-1]
    hfil = filter_vec.reshape((1,r))
    vfil = filter_vec.reshape((r,1))

    def blur(im, expand=False):
        if (expand):
            newIm = convolve(im.copy(), hfil * 2)
        else:
            newIm = convolve(im.copy(), hfil)
        if (expand):
            newIm = convolve(newIm, vfil * 2)
        else:
            newIm = convolve(newIm, vfil)
        return newIm
    return blur

'''
assigment fucntion that getting laplacians list and build
an image..
'''
def laplacian_to_image(lpyr, filter_vec, coeff):
    lpyr = lpyr[::-1]
    currImg = lpyr[0]
    blur = getBlur(filter_vec)
    for idx,elem in enumerate(lpyr[1:]):
        expanded = blur(expandPic(currImg),True)
        r,c = elem.shape
        ### special case handler :\ ###
        currImg = expanded[0:r,0:c] + (elem*coeff[idx])
    return currImg


### 3.3 ###
def stretchPic(im):
    imMin = np.min(im)
    imMax = np.max(im)
    stretch = lambda x:((x-imMin)/(imMax-imMin))
    return stretch(im)

def render_pyramid(pyr,levels):
    totalLength = sum([pyr[i].shape[1] for i in range(levels)])
    totalHeight = max([elem.shape[0] for elem in pyr])
    canvas = np.ndarray((totalHeight,totalLength))
    ldx = 0
    for i in range(levels):
        elem = pyr[i]
        canvas[0:elem.shape[0],ldx:ldx+elem.shape[1]] = stretchPic(elem)
        ldx += elem.shape[1]
    return canvas

def display_pyramid(pyr, levels):
    res = render_pyramid(pyr,levels)
    plt.figure()
    plt.imshow(res,cmap=plt.gray())



### 4 (pam pam pam) ###
def generate_blended_pyramid(l1,l2,gm):
    iterations = min(len(l1),len(l2),len(gm))
    py = []
    cal = lambda x,y,z:(x*z)+((1-z)*y)
    for i in range(iterations):
        py.append(cal(l1[i],l2[i],gm[i]))
    return py


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    L1, fil = build_laplacian_pyramid(im1, max_levels,filter_size_im)
    L2, fil = build_laplacian_pyramid(im2, max_levels,filter_size_im)
    Gm, fil = build_gaussian_pyramid(mask.astype(np.float32),max_levels,filter_size_mask)
    vec = np.ndarray((max_levels))
    vec.fill(1)
    return laplacian_to_image(generate_blended_pyramid(L1,L2,Gm),fil,vec)

#### sheker ####
def relpath(fn):
    return os.path.join(os.path.dirname(__file__),fn)
def blending_example1():
    mask = rgb2gray(imread(relpath('try1/mask.jpg'))).astype(np.float32)
    mask[mask>0.5] = 1
    mask[mask<=0.5] = 0
    mask = mask.astype(np.bool)
    im1 = (imread(relpath('try1/1.jpg')) / 255).astype(np.float32)
    im2 = (imread(relpath('try1/2.jpg')) / 255).astype(np.float32)
    r = pyramid_blending(im1[:,:,0],im2[:,:,0],mask,5,5,5)
    g = pyramid_blending(im1[:,:,1],im2[:,:,1],mask,5,5,5)
    b = pyramid_blending(im1[:,:,2],im2[:,:,2],mask,5,5,5)
    im = np.clip(np.dstack([r,g,b]),0,1)
    return im1, im2, mask, im


def blending_example2():
    mask = rgb2gray(imread(relpath('try2/mask.jpg'))).astype(np.float32)
    mask[mask>0.5] = 1
    mask[mask<=0.5] = 0
    mask = mask.astype(np.bool)
    im1 = (imread(relpath('try2/1.jpg'))/255).astype(np.float32)
    im2 = (imread(relpath('try2/2.jpg'))/255).astype(np.float32)
    r = pyramid_blending(im1[:,:,0],im2[:,:,0],mask,5,5,5)
    g = pyramid_blending(im1[:,:,1],im2[:,:,1],mask,5,5,5)
    b = pyramid_blending(im1[:,:,2],im2[:,:,2],mask,5,5,5)
    im = np.clip(np.dstack([r,g,b]),0,1)
    return im1, im2, mask, im
