from sol1 import *
import skimage
testImages = ["jerusalem.jpg", "monkey.jpg", "Low Contrast.jpg"]
def pixelCheck(name):
    c2 = read_image("jerusalem.jpg", 2)
    pixel = c2[43, 46]
    print(pixel * 255)
    newImg, origHist, newHist, cumsum = histogram_equalize(c2)
    pixelYIG = np.dot(rgb2yiqMatrix,   pixel)
    print(pixelYIG)
    print(pixelYIG * 255)
    pixelYIG[0] = cumsum[int(pixelYIG[0] * 255)] / 255
    print(pixelYIG)
    print(cumsum[222])
    pixelRGB = np.dot(np.around(yiq2rgbMatrix, 3), pixelYIG)
    print(pixelRGB)

def testImg(name):
    c2 = read_image(name, 2)
    c2Gray = makeItGray(c2)
    newImg, origHist, newHist = histogram_equalize(c2)
    newImgGray, origHistGray, newHistGray = histogram_equalize(c2Gray)
    plt.subplot(241)
    plt.imshow(c2)
    plt.title('original image')
    plt.subplot(242)
    plt.imshow(newImg)
    plt.title('hist. equalized image')
    plt.subplot(245)
    plt.plot(origHist,'b',newHist,'r')
    plt.title('equalization process rgb')
    plt.subplot(243)
    plt.imshow(c2Gray, cmap=grayCmap)
    plt.title('image in gray before')
    plt.subplot(244)
    plt.imshow(newImgGray, cmap=grayCmap)
    plt.title('image in gray after')
    plt.subplot(247)
    plt.plot(origHistGray, 'b', newHistGray, 'r')
    plt.title('equalization process gray')
    plt.show()

def quantush(name,colors, tries):
    c2 = read_image(name,2)
    pic, error = quantize(c2,colors,tries)
    plt.subplot(211)
    plt.plot(error, 'b')
    plt.title('errors during iteration')
    plt.subplot(212)
    plt.imshow(pic, cmap=grayCmap)
    plt.title('image in gray before')

def test_quantization(name,colors):
    c2 = read_image(name,2)
    c2Gray = makeItGray(c2)
    pic, error = quantize(c2,colors,1)
    picGray, errorGray = quantize(c2Gray,colors,1)
    print(error)
    plt.subplot(241)
    plt.imshow(c2)
    plt.title('original image')
    plt.subplot(242)
    plt.imshow(pic)
    plt.title('quant image')
    plt.subplot(245)
    plt.plot(error, 'b')
    plt.title('errors during iteration')
    plt.subplot(243)
    plt.imshow(c2Gray, cmap=grayCmap)
    plt.title('image in gray before')
    plt.subplot(244)
    plt.imshow(picGray, cmap=grayCmap)
    plt.title('image in gray after')
    plt.subplot(247)
    plt.plot(errorGray, 'b')
    plt.title('equalization process gray')


def test_Histo():
    c2 = read_image("jerusalem.JPG", 2)
    hist, bounds = np.histogram(makeItGray(c2),bins=256,range=(0,1))
    plt.tick_params(labelsize=20)
    plt.plot((bounds[:-1]+bounds[1:])/2, hist)


# test_quantization("jerusalem.jpg")
quantush("jerusalem.jpg",70,100)
plt.show()
