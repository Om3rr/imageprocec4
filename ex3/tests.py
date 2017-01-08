from sol3 import *
from skimage.color import rgb2gray

def buildPyramid(pyramid):
    canvas = np.ndarray((pyramid[0].shape[0],pyramid[0].shape[1]*2+1,))
    xloc = 0
    for i in pyramid:
        canvas[0:i.shape[0],xloc:xloc+i.shape[1]] = i
        xloc += i.shape[1]
    return canvas

def laplacianBuilder():
    im = rgb2gray(imread('temp/dog.jpg')).astype(np.float32)
    py, fil = build_laplacian_pyramid(im.copy(), 3, 3)
    vec = np.array([1, 1, 1, 1, 1])
    imnew = laplacian_to_image(py, fil, vec)
    ca = render_pyramid(py,len(py))
    plt.figure()
    plt.imshow(ca, cmap=plt.gray())
    plt.figure()
    plt.imshow(im, cmap=plt.gray())
    plt.figure()
    plt.imshow(imnew, cmap=plt.gray())
    print(np.abs(np.max(im - imnew)))
    plt.show()

def gaussianBuilder():
    im = rgb2gray(imread('temp/dog.jpg')).astype(np.float32)
    py, fil = build_gaussian_pyramid(im.copy(), 5, 5)
    ca = render_pyramid(py, len(py))
    plt.figure()
    plt.imshow(ca, cmap=plt.gray())
    plt.figure()
    plt.imshow(im, cmap=plt.gray())
    plt.show()

def combine():
    mask = rgb2gray(imread('try1/mask.jpg')).astype(np.float32)
    mask[mask>0.5] = 1
    mask[mask<=0.5] = 0
    im1 = imread('try1/1.jpg')/255
    im2 = imread('try1/2.jpg')/255
    r = pyramid_blending(im1[:,:,0],im2[:,:,0],mask,5,5,5)
    g = pyramid_blending(im1[:,:,1],im2[:,:,1],mask,5,5,5)
    b = pyramid_blending(im1[:,:,2],im2[:,:,2],mask,5,5,5)
    im = np.clip(np.dstack([r,g,b]),0,1)
    plt.imshow(im)
    plt.show()


def combine2():
    mask = rgb2gray(imread('temp/try2/mask.jpg')).astype(np.float32)
    mask[mask>0.5] = 1
    mask[mask<=0.5] = 0
    im1 = imread('try2/1.jpg')/255
    im2 = imread('try2/2.jpg')/255
    r = pyramid_blending(im1[:,:,0],im2[:,:,0],mask,5,5,5)
    g = pyramid_blending(im1[:,:,1],im2[:,:,1],mask,5,5,5)
    b = pyramid_blending(im1[:,:,2],im2[:,:,2],mask,5,5,5)
    im = np.clip(np.dstack([r,g,b]),0,1)
combine()
