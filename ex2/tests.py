from sol2 import *
from scipy.fftpack import fft2, ifft2, fft
import numpy.fft as ft

def testDFTandIDFT():
    signal = np.array([5, 3, 2, 1, 2, 3, 4, 6, 5, 4, 3, 3, 4, 4, 4, 3, 1])
    fourier = DFT(signal)
    newSignal = IDFT(fourier)
    assert(np.array_equiv(signal,np.around(newSignal)))

def testDFT2():
    myImage = color.rgb2gray(imread("images\\basicSquare.jpg"))
    plt.subplot(221)
    plt.title("original")
    plt.imshow(myImage, cmap=plt.get_cmap("gray"))
    plt.subplot(222)
    plt.title("my dft")
    t = DFT2(myImage)
    t[t < 0.5] = 0
    t[t > 0.5] = 1
    plt.imshow(np.real(t), cmap=plt.get_cmap("gray"))
    plt.subplot(223)
    plt.title("scipy dft")
    t = np.real(fft2(myImage))
    t[t < 0.5] = 0
    t[t > 0.5] = 1
    plt.imshow(t, cmap=plt.get_cmap("gray"))
    plt.show()

def testDFT2andIDFT2():
    myImage = color.rgb2gray(imread("images\\basicSquare.jpg"))
    plt.subplot(221)
    plt.title("Original")
    plt.imshow(myImage, cmap=plt.get_cmap("gray"))
    plt.subplot(223)
    plt.title("My DFT2 Scipy IDFT2")
    t = DFT2(myImage)
    t = np.real(ifft2(t))
    plt.imshow(t, cmap=plt.get_cmap("gray"))
    plt.subplot(224)
    plt.title("Scipy DFT2 My IDFT2")
    t = fft2(myImage)
    t = np.real(IDFT2(t))
    plt.imshow(np.real(t), cmap=plt.get_cmap("gray"))
    plt.show()

def test2derivative():
    myImage = color.rgb2gray(imread("images\\basicSquare.jpg"))
    plt.subplot(221)
    plt.title("Original")
    plt.imshow(myImage, cmap=plt.get_cmap("gray"))
    plt.subplot(223)
    plt.title("h Derivative")
    z = conv_der(myImage)
    plt.imshow(z, cmap=plt.get_cmap("gray"))
    plt.show()

def testDerivativeComparison():
    myImage = color.rgb2gray(imread("images\\jerusalem.jpg"))
    plt.subplot(221)
    print(myImage.shape)
    plt.title("Original")
    plt.imshow(myImage, cmap=plt.get_cmap("gray"))
    z = conv_der(myImage)
    z2 = fourier_der(myImage)
    plt.subplot(223)
    plt.title("calculated Derivative")
    plt.imshow(z, cmap=plt.get_cmap("gray"))
    plt.subplot(224)
    plt.title("fourier Derivative")
    plt.imshow(z2, cmap=plt.get_cmap("gray"))
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()

def fourierVSregularDerivTest():
    myImage = color.rgb2gray(imread("images\\jerusalem.jpg"))
    four = fourier_der(myImage)
    deriv = conv_der(myImage)

    plt.subplot(131)
    plt.title("Original")
    plt.imshow(myImage, cmap=plt.get_cmap("gray"))

    plt.subplot(132)
    plt.title("Fourier")
    plt.imshow(four, cmap=plt.get_cmap("gray"))

    plt.subplot(133)
    plt.title("Regular")
    plt.imshow(deriv, cmap=plt.get_cmap("gray"))


def blurCompare():
    myImage = color.rgb2gray(imread("images\\monkey.jpg"))
    blured = blur_fourier(myImage, 25)
    blured2 = blur_spatial(myImage, 25)
    plt.subplot(221)
    plt.title('originals')
    plt.imshow(myImage, cmap=plt.get_cmap("gray"))
    plt.subplot(223)
    plt.title('fourier')
    plt.imshow(blured, cmap=plt.get_cmap("gray"))
    plt.subplot(224)
    plt.title('convolution')
    plt.imshow(blured2, cmap=plt.get_cmap("gray"))
    plt.show()
testDerivativeComparison()
plt.show()
blurCompare()
plt.show()