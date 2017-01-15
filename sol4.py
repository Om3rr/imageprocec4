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
    Ix = convolve(im, np.array([1,0,-1]).reshape(1,3))
    Iy = convolve(im, np.array([1,0,-1]).reshape(3,1))
    return Ix, Iy

def harris_corner_detector(im):
    x, y = im.shape
    M = np.ndarray((x, y, 2, 2))
    Ix, Iy = deriv(im)
    M[:,:,0,0] = blur_spatial(Ix.copy()**2,3) # Ix^2
    M[:,:,1,1] = blur_spatial(Iy.copy()**2,3) # Iy^2
    temp = blur_spatial(Ix*Iy,3) # Ix * Iy
    M[:,:,1,0] = temp.copy()
    M[:,:,0,1] = temp
    detM = np.linalg.det(M[:,:])
    traceM = M[:,:,0,0] + M[:,:,1,1]
    R = detM.copy() - K*((traceM)**2)
    r = non_maximum_suppression(R)
    x,y = np.where(r==True)
    points = np.ndarray((x.shape[0],2))
    points[:,1] = x
    points[:,0] = y
    return points

def sample_descriptor(im, pos, desc_rad):
    curreK = (desc_rad*2)+1
    desc = np.ndarray((curreK, curreK,pos.shape[0]))
    for index, feature in enumerate(pos):
        grid = np.meshgrid(np.arange(feature[1]-desc_rad, feature[1]+desc_rad+1),np.arange(feature[0]-desc_rad, feature[0]+desc_rad+1))
        dRaw = map_coordinates(im, grid, order=1, prefilter='false')
        dRaw = dRaw.reshape((curreK, curreK))
        avg = np.average(dRaw)
        if((np.linalg.norm(dRaw - avg)) == 0):
            continue
        desc[:,:,index] = (dRaw - avg)/(np.linalg.norm(dRaw - avg))
    return desc.astype(np.float32)


def find_features(pyr, radius=3, m=5,n=5):
    points= spread_out_corners(pyr[0],m,n,radius*4)
    return points ,sample_descriptor(pyr[2], points/4, radius)


def match_features(desc1, desc2, min_score):
    N1 = desc1.shape[2]
    N2 = desc2.shape[2]
    matches1 = np.array([-1,-1]*N1).reshape((N1,2))
    matches = []
    for i in range(N1):
        matches1[i] = match_feature(desc1[:,:,i], desc2, min_score)
    for i in range(N2):
        k,secK = match_feature(desc2[:,:,i], desc1, min_score)
        if(k != -1 and i in matches1[k]):
            matches.append([k,i])
        elif(secK != -1 and i in matches1[secK]):
            matches.append([secK,i])
    matches = np.array(matches).astype(np.int)
    return matches[:,0], matches[:,1]




def match_feature(feature, featureList, min_score):
    # preprocess
    temp, rad, length = featureList.shape
    featureList = featureList.reshape((rad*rad,length))
    feature = feature.reshape((rad*rad))
    products = np.dot(feature,featureList[:,:])
    if(np.max(products) >= min_score):
        k = np.argmax(products)
        sk = np.argmax(np.concatenate((products[0:k],products[k+1:-1])))
        if(sk>k-1):
            sk+=1
        if (products[sk] < min_score):
            return [k,-1]
        return [k, sk]
    return [-1,-1]



def apply_homography(pos1, H12):
    N = pos1.shape[0]
    points = np.ndarray((N,3))
    points[:,0:2] = pos1
    points[:,2] = 1
    newPoints = np.dot(points,H12.T)
    newPoints[:,0] /= newPoints[:,2]
    newPoints[:,1] /= newPoints[:,2]
    return newPoints[:,0:2]

def ransac_homography(pos1, pos2, num_iters, inlier_tol):
    from random import sample
    assert(pos1.shape == pos2.shape) #important!
    N = pos1.shape[0]
    maxInliers = np.ndarray((1,))
    maxH = None
    for i in range(num_iters):
        J = sample(range(N),4)
        J1 = np.take(pos1,J,axis=0)
        J2 = np.take(pos2,J,axis=0)
        H12 = least_squares_homography(J1,J2)
        if(H12 is None):
            num_iters += 1 # ignore this round
            continue
        T1 = apply_homography(pos1,H12)
        dist = np.sqrt(np.abs(T1-pos2)**2)
        dist[dist<inlier_tol] = True
        dist[dist>=inlier_tol] = False
        whr = np.argwhere(dist==True)
        whr = whr.reshape(-1)[::2]
        S = np.delete(whr,np.unique(whr, return_index=True))
        if(S.shape[0] > maxInliers.shape[0]):
            maxInliers = S
            maxH = H12
    return maxH,maxInliers



def display_matches(im1, im2, pos1,pos2,inliers):
    r1,c1 = im1.shape
    r2,c2 = im2.shape
    r = max(r1,r2)
    c = c1+c2
    canvas = np.ndarray((r,c))
    canvas[0:r1,0:c1] = im1
    canvas[0:r2,c1:c] = im2
    showImg(canvas)
    pos2[:,0] += c1
    plt.scatter(pos2[:,0],pos2[:,1])
    plt.scatter(pos1[:,0],pos1[:,1])
    in1 = np.take(pos1,inliers,axis=0)
    in2 = np.take(pos2,inliers,axis=0)
    for i in range(pos1.shape[0]):
        plt.plot([pos1[i, 0], pos2[i, 0]], [pos1[i, 1], pos2[i, 1]], color='blue', linestyle='-', linewidth=1)
    for i in range(in1.shape[0]):
        plt.plot([in1[i,0],in2[i,0]],[in1[i,1],in2[i,1]], color='yellow', linestyle='-', linewidth=1)


def accumulate_homographies(H_successive, m):
    M = len(H_successive)
    matrixes = [None]*(M+1)
    matrixes[m] = np.identity(3)
    for i in range(m+1,len(H_successive)+1):
        matrixes[i] = np.dot(matrixes[i-1], np.linalg.inv(H_successive[i-1]))
    for i in range(m-1,-1,-1):
        matrixes[i] = np.dot(H_successive[i],matrixes[i+1])
    return [matrixes[i]/matrixes[i][2,2] for i in range(len(matrixes))]

def findCorners(ims, Hs):
    corners = [np.empty((0, 2), dtype=np.int)]
    for idx, im in enumerate(ims[0:]):
        x, y = im.shape[::-1]
        x -= 1
        y -= 1
        corners.append(apply_homography(np.array([[0, 0], [x, 0], [0, y], [x,y]]), Hs[idx]))
    corners = np.vstack(corners).astype(np.int)
    xmin, ymin = np.min(corners[:, 0]), np.min(corners[:, 1])
    xmax, ymax = np.max(corners[:, 0]), np.max(corners[:, 1])
    return xmin, xmax, ymin, ymax

def getCenters(ims, Hs):
    centers = np.ndarray((0,0))
    for idx in range(len(ims)):
        x,y = ims[idx].shape[::-1]
        v = np.dot(Hs[idx], np.array([x//2, y//2, 1]))
        v = v[0] / v[2]
        centers = np.append(centers, v)
    return centers

R = 50 #defining stiching radius

def render_panorama(ims, Hs):
    xmin, xmax, ymin, ymax = findCorners(ims,Hs)
    xmin, ymin = np.abs(xmin), np.abs(ymin)
    print(xmin,xmax,ymin,ymax)
    Realcntrs = getCenters(ims,Hs).astype(np.int)
    Accumcntrs = np.array([(Realcntrs[i] + Realcntrs[i+1])//2 for i in range(Realcntrs.size-1)]+[xmax]).astype(np.int)
    print(Realcntrs)
    Accumcntrs = np.insert(Accumcntrs, 0, [-xmin]) # if xmin < 0 insert xmin  to0
    canvas = np.ndarray((ymin+ymax,xmin+xmax))

    # masks = np.ndarray((len(Accumcntrs)-2, ymin+ymax, R*2))
    for idx,im in enumerate(ims):
        grid = np.array(np.meshgrid(np.arange(Accumcntrs[idx],Accumcntrs[idx+1]),np.arange(-ymin,ymax)))
        print(Accumcntrs[idx],Accumcntrs[idx+1])
        z,y,x = grid.shape
        ngrid = np.ndarray((y,x,z))
        ngrid[:,:,0], ngrid[:,:,1] = grid[0,:,:], grid[1,:,:]
        poss = apply_homography(ngrid.reshape((x*y,2)),np.linalg.inv(Hs[idx]))
        mapped = map_coordinates(im,[poss[:,1].reshape((y,x)),poss[:,0].reshape((y,x))], order=1, prefilter='false')
        canvas[:,Accumcntrs[idx]+xmin:Accumcntrs[idx+1]+xmin] = mapped
        # if(idx!=len(ims)-1):
        #     masks[idx,:,:] = canvas[:,Accumcntrs[idx+1]+xmin-R:Accumcntrs[idx+1]+xmin+R]
    # return canvas
    return canvas.astype(np.float32)

# super function to handle image changes
def stichThingsTogether(masks, canvas, ims, Hs, Accumcntrs, ys,xmin):
    # we need to take a piece from old picture (radius length X ymax+ymin) from two different pictures and blend
    # together
    for i in range(len(ims)-1):
        accunCntrs = Accumcntrs[1:-1] # we dont need them, they are bad!!
        pos = accunCntrs[0]+xmin
        grid = np.array(np.meshgrid(np.arange(pos-R,pos+R),np.arange(-ys[0],ys[1])))
        z, y, x = grid.shape
        ngrid = np.ndarray((y, x, z))
        ngrid[:, :, 0], ngrid[:, :, 1] = grid[0, :, :], grid[1, :, :]
        poss1 = apply_homography(ngrid.reshape((x * y, 2)), np.linalg.inv(Hs[i]))
        mapped1 = map_coordinates(ims[i], [poss1[:, 1].reshape((y, x)), poss1[:, 0].reshape((y, x))], order=1, prefilter='false')
        poss2 = apply_homography(ngrid.reshape((x * y, 2)), np.linalg.inv(Hs[i+1]))
        mapped2 = map_coordinates(ims[i+1], [poss2[:, 1].reshape((y, x)), poss2[:, 0].reshape((y, x))], order=1, prefilter='false')
        mask = masks[i]
        mask[mask!=0] = 1
        mask = np.logical_not(mask)
        blended = pyramid_blending(mapped2,mapped1,mask,5,15,15)
        canvas[:,pos-R+xmin:pos+R+xmin] = blended
    return canvas.astype(np.float32)









