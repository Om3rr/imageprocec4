from sol4_utils import *
from sol4_add import non_maximum_suppression, least_squares_homography, spread_out_corners
from scipy.ndimage.filters import convolve
from scipy.ndimage import map_coordinates, affine_transform
from matplotlib import pyplot as plt
import numpy as np

K = 0.04


def showImg(im):
    plt.figure()
    plt.imshow(im, cmap=plt.gray())


def deriv(im):
    Ix = convolve(im, np.array([1, 0, -1]).reshape(1, 3))
    Iy = convolve(im, np.array([1, 0, -1]).reshape(3, 1))
    return Ix, Iy


'''
implementation of harris corner detection, kinda straight-forward from the lesson
'''


def harris_corner_detector(im):
    x, y = im.shape
    # 2X2 matrix for every pixel
    M = np.ndarray((x, y, 2, 2))
    # calculate derivatives
    Ix, Iy = deriv(im)
    M[:, :, 0, 0] = blur_spatial(Ix.copy() ** 2, 3)  # Ix^2
    M[:, :, 1, 1] = blur_spatial(Iy.copy() ** 2, 3)  # Iy^2
    temp = blur_spatial(Ix * Iy, 3)  # Ix * Iy
    M[:, :, 1, 0] = temp.copy()
    M[:, :, 0, 1] = temp

    # calc det for each 2X2
    detM = np.linalg.det(M[:, :])

    # calc trace for each 2X2
    traceM = M[:, :, 0, 0] + M[:, :, 1, 1]

    # some magic powder
    R = detM.copy() - K * ((traceM) ** 2)
    r = non_maximum_suppression(R)
    x, y = np.where(r == True)
    points = np.ndarray((x.shape[0], 2))
    points[:, 1] = x
    points[:, 0] = y
    return points


'''
samplic descriptors in specific image
'''


def sample_descriptor(im, pos, desc_rad):
    curreK = (desc_rad * 2) + 1
    desc = np.ndarray((curreK, curreK, pos.shape[0]))
    for index, feature in enumerate(pos):
        grid = np.meshgrid(np.arange(feature[1] - desc_rad, feature[1] + desc_rad + 1),
                           np.arange(feature[0] - desc_rad, feature[0] + desc_rad + 1))
        dRaw = map_coordinates(im, grid, order=1, prefilter='false')
        dRaw = dRaw.reshape((curreK, curreK))
        avg = np.average(dRaw)
        if ((np.linalg.norm(dRaw - avg)) == 0):
            continue
        desc[:, :, index] = (dRaw - avg) / (np.linalg.norm(dRaw - avg))
    return desc.astype(np.float32)


'''
finding features in pyramid (sample them on the 3rd level)
'''


def find_features(pyr, radius=3, m=5, n=5):
    points = spread_out_corners(pyr[0], m, n,
                                radius * 4)  # im using radius times 4 because in the next function im using orig radius
    return points, sample_descriptor(pyr[2], points / 4, radius)  # how cool is it to write 2 lines function :)


'''
matching features between two lists of features.
'''


def match_features(desc1, desc2, min_score):
    N1 = desc1.shape[2]
    N2 = desc2.shape[2]
    matches1 = np.array([-1, -1] * N1).reshape((N1, 2))
    matches = []
    for i in range(N1):
        matches1[i] = match_feature(desc1[:, :, i], desc2, min_score)
    for i in range(N2):
        k, secK = match_feature(desc2[:, :, i], desc1, min_score)
        if (k != -1 and i in matches1[k]):
            matches.append([k, i])
        elif (secK != -1 and i in matches1[secK]):
            matches.append([secK, i])
    matches = np.array(matches).astype(np.int)
    return matches[:, 0], matches[:, 1]


'''
finding feature with the best correlation from the second list
'''


def match_feature(feature, featureList, min_score):
    # preprocess
    temp, rad, length = featureList.shape
    featureList = featureList.reshape((rad * rad, length))
    feature = feature.reshape((rad * rad))
    products = np.dot(feature, featureList[:, :])
    if (np.max(products) >= min_score):
        k = np.argmax(products)
        sk = np.argmax(np.concatenate((products[0:k], products[k + 1:-1])))
        if (sk > k - 1):
            sk += 1
        if (products[sk] < min_score):
            return [k, -1]
        return [k, sk]
    return [-1, -1]


'''
apply homographi on specific list of points using np.dot
'''


def apply_homography(pos, H12):
    N = pos.shape[0]
    points = np.ndarray((N, 3))
    points[:, 0:2] = pos
    points[:, 2] = 1
    newPoints = np.dot(points, H12.T)
    newPoints[newPoints == 0] = 0.0000000000001  # bad method to check if we are dividing by zero.
    # if I was my CTO I probably cut my own salary or something.
    newPoints[:, 0] /= newPoints[:, 2]
    newPoints[:, 1] /= newPoints[:, 2]
    return newPoints[:, 0:2]


def ransac_homography(pos1, pos2, num_iters, inlier_tol):
    from random import sample
    assert (pos1.shape == pos2.shape)  # important!
    N = pos1.shape[0]
    maxInliers = np.ndarray((1,))
    maxH = None
    for i in range(num_iters):
        J = sample(range(N), 4)
        J1 = np.take(pos1, J, axis=0)
        J2 = np.take(pos2, J, axis=0)
        H12 = least_squares_homography(J1, J2)
        if (H12 is None):
            num_iters += 1  # ignore this round
            continue
        T1 = apply_homography(pos1, H12)
        dist = np.sqrt(np.abs(T1 - pos2) ** 2)
        dist[dist < inlier_tol] = True
        dist[dist >= inlier_tol] = False
        whr = np.argwhere(dist == True)
        whr = whr.reshape(-1)[::2]
        S = np.delete(whr, np.unique(whr, return_index=True))
        if (S.shape[0] > maxInliers.shape[0]):
            maxInliers = S
            maxH = H12
    return maxH, maxInliers


'''
displaying matches obtained after all the iteration
yellows are inliers
blues are bad matches
'''


def display_matches(im1, im2, pos1, pos2, inliers):
    r1, c1 = im1.shape
    r2, c2 = im2.shape
    r = max(r1, r2)
    c = c1 + c2
    canvas = np.ndarray((r, c))
    canvas[0:r1, 0:c1] = im1
    canvas[0:r2, c1:c] = im2
    pos2[:, 0] += c1
    plt.scatter(pos2[:, 0], pos2[:, 1])
    plt.scatter(pos1[:, 0], pos1[:, 1])
    in1 = np.take(pos1, inliers, axis=0)
    in2 = np.take(pos2, inliers, axis=0)
    for i in range(pos1.shape[0]):
        plt.plot([pos1[i, 0], pos2[i, 0]], [pos1[i, 1], pos2[i, 1]], color='blue', linestyle='-', linewidth=1)
    for i in range(in1.shape[0]):
        plt.plot([in1[i, 0], in2[i, 0]], [in1[i, 1], in2[i, 1]], color='yellow', linestyle='-', linewidth=1)
    plt.imshow(canvas, cmap=plt.gray())
    plt.figure()


'''
convert list of matrix to homographic super-duper list of matrixes
'''


def accumulate_homographies(H_successive, m):
    M = len(H_successive)
    matrixes = [None] * (M + 1)
    matrixes[m] = np.identity(3)
    for i in range(m + 1, len(H_successive) + 1):
        matrixes[i] = np.dot(matrixes[i - 1], np.linalg.inv(H_successive[i - 1]))
    for i in range(m - 1, -1, -1):
        matrixes[i] = np.dot(H_successive[i], matrixes[i + 1])
    return [matrixes[i] / matrixes[i][2, 2] for i in range(len(matrixes))]


'''
finding images corners
'''


def findCorners(ims, Hs):
    corners = [np.empty((0, 2), dtype=np.int)]
    for idx, im in enumerate(ims[0:]):
        x, y = im.shape[::-1]
        x -= 1
        y -= 1
        corners.append(apply_homography(np.array([[0, 0], [x, 0], [0, y], [x, y]]), Hs[idx]))
    corners = np.vstack(corners).astype(np.int)
    xmin, ymin = np.min(corners[:, 0]), np.min(corners[:, 1])
    xmax, ymax = np.max(corners[:, 0]), np.max(corners[:, 1])
    return xmin, xmax, ymin, ymax


'''
finding the images centers
'''


def getCenters(ims, Hs):
    centers = np.ndarray((0, 0))
    for idx in range(len(ims)):
        x, y = ims[idx].shape[::-1]
        v = np.dot(Hs[idx], np.array([x // 2, y // 2, 1]))
        v = v[0] / v[2]
        centers = np.append(centers, v)
    return centers




'''
returns random dumb mask for the blending (ones on 0:radius//2) and zeros otherwise
'''


def getMask(shape, radius):
    mask = np.zeros(shape)
    mask[:, 0:radius // 2] = 1
    return mask


'''
rendering panorama and stiching on the way.
'''
R = 50  # defining stiching radius

def render_panorama(ims, Hs):
    xmin, xmax, ymin, ymax = findCorners(ims, Hs)
    xmin, ymin = np.abs(xmin), np.abs(ymin)
    Realcntrs = getCenters(ims, Hs).astype(np.int)
    Accumcntrs = np.array([(Realcntrs[i] + Realcntrs[i + 1]) // 2 for i in range(Realcntrs.size - 1)] + [xmax]).astype(
        np.int)
    Accumcntrs = np.insert(Accumcntrs, 0, [-xmin])  # if xmin < 0 insert xmin  to0
    canvas = np.ndarray((ymin + ymax, xmin + xmax))
    length = len(Hs)
    radius = R
    for idx, im in enumerate(ims):
        if (idx == length - 1):
            radius = 0
        grid = np.array(np.meshgrid(np.arange(Accumcntrs[idx], Accumcntrs[idx + 1] + radius), np.arange(-ymin, ymax)))
        z, y, x = grid.shape
        ngrid = np.ndarray((y, x, z))
        ngrid[:, :, 0], ngrid[:, :, 1] = grid[0, :, :], grid[1, :, :]
        poss = apply_homography(ngrid.reshape((x * y, 2)), np.linalg.inv(Hs[idx]))
        mapped = map_coordinates(im, [poss[:, 1].reshape((y, x)), poss[:, 0].reshape((y, x))], order=1,
                                 prefilter='false')
        if (idx != 0 and idx != length - 1):
            mapped = pyramid_blending(canvas[:, Accumcntrs[idx] + xmin:Accumcntrs[idx + 1] + xmin + radius], mapped,
                                      getMask(mapped.shape, radius), 3, 15, 15)
        if (idx == length - 1):
            mapped = pyramid_blending(canvas[:, Accumcntrs[idx] + xmin:Accumcntrs[idx + 1] + xmin], mapped,
                                      getMask(mapped.shape, R), 3, 15, 15)
        showImg(mapped)
        canvas[:, Accumcntrs[idx] + xmin:Accumcntrs[idx + 1] + xmin + radius] = mapped
    return canvas.astype(np.float32)
