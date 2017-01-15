
from sol4 import *
from sol4_add import spread_out_corners
def showImg(im):
    plt.figure()
    plt.imshow(im, cmap=plt.gray())
import time
RADIUS = 5
MATCH_FEATURES_THRES = 0.7
ITERS = 500
TOL = 3

def matchToIm(abc=True):
    im = read_image("external\\backyard1.jpg", 1)
    pyr = build_gaussian_pyramid(im, 3, 3)[0]
    featurez, pos = find_features(pyr, RADIUS)

    im2 = read_image("external\\backyard2.jpg", 1)
    pyr2 = build_gaussian_pyramid(im2, 3, 3)[0]
    featurez2, pos2 = find_features(pyr2, RADIUS)

    im3 = read_image("external\\backyard3.jpg", 1)
    pyr3 = build_gaussian_pyramid(im3, 3, 3)[0]
    featurez3, pos3 = find_features(pyr3, RADIUS)


    a, b = match_features(featurez, featurez2, MATCH_FEATURES_THRES)
    posa = np.take(pos.copy(), a, axis=0)
    posb = np.take(pos2.copy(), b, axis=0)
    inLiners, H1 = ransac_homography(posa, posb, ITERS, TOL)

    b, c = match_features(featurez2, featurez3, MATCH_FEATURES_THRES)
    posa = np.take(pos2.copy(), b, axis=0)
    posb = np.take(pos3.copy(), c, axis=0)
    inLiners, H2 = ransac_homography(posa, posb, ITERS, TOL)
    # display_matches(pyr[0], pyr2[0], pos1, pos3, inLiners)
    if(abc):
        return im, im2,im3, H1, H2
    return im, im2, H1

## load and find features ##

def genPano():
    ims = []
    imsLow = []
    Hs = []
    for i in range(1,3):
        im = read_image('external/oxford%s.jpg'%i,1)
        ims.append(im)
        pyr = build_gaussian_pyramid(ims[i - 2], 3, 3)[0]
        imsLow.append(pyr[2])
        if(i == 1):
            continue
        featurez, pos = find_features(pyr, RADIUS)
        pyr2 = build_gaussian_pyramid(im, 3, 3)[0]
        featurez2, pos2 = find_features(pyr2, RADIUS)
        pos2 = pos2 * 4
        pos = pos * 4
        a, b = match_features(featurez, featurez2, MATCH_FEATURES_THRES)
        pos = np.take(pos, a, axis=0)
        pos2 = np.take(pos2, b, axis=0)
        inliner,H1 = ransac_homography(pos, pos2, ITERS, TOL)
        Hs.append(H1)
        display_matches(pyr[0], pyr2[0], pos, pos2, inliner)
    Hs = accumulate_homographies(Hs, (len(Hs)-1)//2)
    render_panorama(ims,Hs)

def testPano(im1, im2, im3, Hss, H1 = None):
    m = (len(Hss))//2
    Hs = accumulate_homographies(Hss,m)
    if(len(Hss)==1):
        render_panorama([im1,im2],Hs)
        return
    render_panorama([im1,im2, im3],Hs)

t = time.time()
im, im2,im3, H1, H2 = matchToIm()
Hs = [H1,H2]
testPano(im,im2,im3,Hs)
im, im2, H1 = matchToIm(False)
Hs = [H1]
testPano(im,im2,None,Hs)
print(time.time() - t)
plt.show()
# interp('')
# genPano()
# a = np.array([[1,2,3],[4,5,6],[7,8,9]])
# print(a.T)