
from sol4 import *
from sol4_add import spread_out_corners
def showImg(im):
    plt.figure()
    plt.imshow(im, cmap=plt.gray())
import time
RADIUS = 5
MATCH_FEATURES_THRES = 0.7
ITERS = 300
TOL = 3

def matchToIm():
    t = time.time()
    im = read_image("images\\backyard1.jpg", 1)
    pyr = build_gaussian_pyramid(im, 3, 3)[0]
    featurez, pos = find_features(pyr, RADIUS)

    im2 = read_image("images\\backyard2.jpg", 1)
    pyr2 = build_gaussian_pyramid(im2, 3, 3)[0]
    featurez2, pos2 = find_features(pyr2, RADIUS)

    for i in range(1):
        a, b = match_features(featurez, featurez2, MATCH_FEATURES_THRES)
        pos1 = np.take(pos.copy(), a, axis=0)
        pos3 = np.take(pos2.copy(), b, axis=0)
        inLiners, H1 = ransac_homography(pos1, pos3, ITERS, TOL)

        display_matches(pyr[0], pyr2[0], pos1, pos3, inLiners)
    # with open('pkl.pkl','wb') as f:
    #     pickle.dump([[im,im2],[H1]], f)

    return im, im2, H1

## load and find features ##

def genPano():
    ims = []
    imsLow = []
    Hs = []
    for i in range(1,3):
        im = read_image('images/oxford%s.jpg'%i,1)
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

def testPano(im1, im2, H1):
    Hs = [np.identity(3),np.linalg.inv(H1)]
    # Hs = accumulate_homographies(Hs,1)
    render_panorama([im1,im2],Hs)
t = time.time()
a,b,c = matchToIm()
testPano(a,b,c)
print(time.time() - t)
plt.show()
# interp('')
# genPano()
# a = np.array([[1,2,3],[4,5,6],[7,8,9]])
# print(a.T)