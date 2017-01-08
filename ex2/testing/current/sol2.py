import numpy as np
import matplotlib.pyplot as plt
import cmath as cm
from scipy.misc import imread
from skimage import color
from scipy.fftpack import fftshift, ifftshift, fft2, ifft2
from scipy.signal import convolve2d

derivativeMatrix = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
mathI = np.complex(0, 1)


def read_image(img, color):
    if color is 2:
        return imread(img)
    else:
        return color.rgb2gray(imread(img))
# section 1
# 1.1
'''
Implementing basic 1d fourier transformation (like vector like son)
getting signal (matrix or array) assuming len(shape)==2
'''


def DFT(signal):
    # assert (len(signal.shape) is 2)
    n = signal.shape
    if (len(n) is 1):
        signal.reshape(-1)
    func = getFourierFunc(n[0], "DFT")
    matrix = np.fromfunction(func, (n[0], n[0]))
    if (len(n) is not 1):
        func2 = getFourierFunc(n[1], "DFT")
        matrix2 = np.fromfunction(func2, (n[1], n[1]))
        return np.dot(matrix, np.dot(signal, matrix2))

    return np.dot(matrix, signal).reshape(n)


'''
Implementing basic 1d fourier anti transformation (like vector like son)
getting signal (matrix or array) assuming len(shape)==2
'''


def IDFT(fourie_signal):
    # assert (len(fourie_signal.shape) is 2)
    n = fourie_signal.shape
    if (len(n) is 1):
        fourie_signal.reshape(-1)

    func = getFourierFunc(n[0], "IDFT")
    matrix = np.fromfunction(func, (n[0], n[0]))
    if (len(n) is not 1):
        func2 = getFourierFunc(n[1], "IDFT")
        matrix2 = np.fromfunction(func2, (n[1], n[1]))
        fourie_signal = np.dot(matrix, np.dot(fourie_signal, matrix2))
        return fourie_signal / (n[0] * n[1])
    fourie_signal = np.dot(matrix, fourie_signal).reshape(n)

    return fourie_signal / n[0]


'''
func that returns func who build the fourier matrix
'''


def getFourierFunc(n, s):
    if s is "DFT":
        return lambda i, j: cm.e ** (-((2 * cm.pi * mathI * i * j) / n))
    return lambda i, j: cm.e ** ((2 * cm.pi * mathI * i * j) / n)


# 1.2
'''
converting from regular image into fourier one
using DFT function.
'''


def DFT2(image):
    return DFT(image)


'''
converting from fourier image into regular image
while implemented section1 bonus.
'''


def IDFT2(image):
    return IDFT(image)


# section 2 derivatives
'''
i/o - float32 image as np.ndarray
calculate horizontal and vertical derivative of the image and
'''


def conv_der(im):
    # horizontal
    hDeriv = convolve2d(im, derivativeMatrix, 'same')
    # veritcal
    vDeriv = convolve2d(im, derivativeMatrix.transpose(), 'same')
    return cal2dDeriv(hDeriv, vDeriv)


'''
function that gets two derivative
h = represent derivative repr on X
v = represent derivative repr on y
it returns new matrix represent the magnitude of them
'''


def cal2dDeriv(h, v):
    return np.sqrt((np.abs(h) ** 2) + (np.abs(v) ** 2))


'''
derivative the image using fourier transformation
according to the instructions as appear in the lecture notes.
'''


def fourier_der(im):
    rs, cs = im.shape
    rowMulMatrix, colMulMatrix = generateDerivFouriaMatrixes(rs, cs)
    # X derivative
    # Compute fourier transform
    fourier = DFT2(im)

    # Multiply each fourier coefficient F(u,v) by u
    xDeriv = np.dot(fourier, colMulMatrix)

    # Compute the inverse fourier transform
    xDeriv = np.abs(IDFT2(xDeriv) * ((2 * cm.pi * mathI) / (rs * cs)))

    # Y derivative

    # Compute fourier transform done in x
    # Multiply each fourier coefficient F(u,v) by v
    yDeriv = np.dot(rowMulMatrix, fourier)

    # Compute the inverse fourier transform
    yDeriv = np.abs(IDFT2(yDeriv) * ((2 * cm.pi * mathI) / (rs * cs)))

    return cal2dDeriv(xDeriv, yDeriv)

'''
This function will generate for us 2 matrixes according
the fouria derivative algorithm (0`s and diagonal with 0 to n/2 and -n/2-1 to -1)
'''
def generateDerivFouriaMatrixes(r, c):
    rArr = fftshift(range(-(np.floor(r / 2).astype(np.int32)) - 1, np.floor(r / 2).astype(np.int32)))
    cArr = fftshift(range(-(np.floor(c / 2).astype(np.int32)) - 1, np.floor(c / 2).astype(np.int32)))
    rowMulMatrix = np.zeros((r, r))
    colMulMatrix = np.zeros((c, c,))
    np.fill_diagonal(rowMulMatrix, rArr)
    np.fill_diagonal(colMulMatrix, cArr)
    return rowMulMatrix, colMulMatrix


# section 3
# 3.1


'''
this function will create an odd size gaussian using
simple 2dconvolution (my only loop in the program :( sry)
'''
def createGaussian(size):
    assert(size % 2 == 1)
    factor = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]).astype(np.float32)
    factor = factor / np.sum(factor)

    gaussian = factor.copy()
    for i in range(int((size - 3) / 2)):
        gaussian = convolve2d(gaussian, factor)
    print(np.sum(gaussian))
    return gaussian


'''
this function will blur the required picture. using
convolution gaussian smoothing technique
'''
def blur_spatial(im, kernel_size):
    gaussian = createGaussian(kernel_size)
    return convolve2d(im, gaussian, 'same')


def testGaussian(kerSize, im):
    temp_gaussian = createGaussian(kerSize)
    #### trying from the corent
    gaussian = np.zeros(im.shape)
    point = np.floor(kerSize / 2)
    print(temp_gaussian)
    gaussian[0:point + 1, 0:point + 1] = temp_gaussian[point:, point:]
    Ggaus = fftshift(DFT2(gaussian))
    ####### trying from center
    Newgaussian = np.zeros(im.shape)
    r,c = im.shape
    r = int(np.floor(r/2) + 1)
    c = int(np.floor(c/2) + 1)
    halfKer = int(np.floor(kerSize) + 1)
    r -= halfKer
    c -= halfKer
    Newgaussian[r:r+kerSize,c:c+kerSize] = temp_gaussian
    Newgaussian = ifftshift(Newgaussian)
    Newgaussian = DFT2(Newgaussian)
    NewgaussianAfterShift = fftshift(Newgaussian)
    plt.subplot(121)
    plt.title("Gaus cornert")
    plt.imshow(np.log10(np.abs(Ggaus) ** 2), cmap=plt.get_cmap("gray"))
    plt.subplot(122)
    plt.title("Gaus center")
    plt.imshow(np.log10(np.abs(NewgaussianAfterShift) ** 2), cmap=plt.get_cmap("gray"))

    plt.show()

def blur_fourier(im, kernel_size):
    # create gaussian
    temp_gaussian = createGaussian(kernel_size)
    Fim = fftshift(DFT2(im))

    # gaussian 2
    Newgaussian = np.zeros(im.shape)
    r, c = im.shape
    r = int(np.floor(r / 2)+1)
    c = int(np.floor(c / 2)+1)
    halfKer = int(np.floor(kernel_size/2) + 1)
    Newgaussian[r - halfKer:r + halfKer - 1, c - halfKer:c + halfKer - 1] = temp_gaussian
    Newgaussian = ifftshift(DFT2(fftshift(Newgaussian)))

    #multiply gaussians
    blured = Fim*Newgaussian

    #show
    return np.real(IDFT2(ifftshift(blured)))


