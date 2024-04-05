%matplotlib inline
# standard libraries
import numpy as np
import scipy as sp
import pandas as pd
import glob,os,re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# image related libraries
from skimage.filters import threshold_otsu
from scipy.ndimage import distance_transform_edt,convolve
import seaborn as sns
import skimage
from skimage import io
from PIL import Image
# TDA related libraries
import cripser, tcripser
from ripser import ripser
import persim
import gudhi

### define functions used later
# gaussian kernel for convolution
def gaussian(h,sigma):
    x = np.arange(-h[0],h[0],1)
    y = np.arange(-h[1],h[1],1)
    z = np.arange(-h[2],h[2],1)
    xx, yy,zz = np.meshgrid(x,y,z)
    return(np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2)).astype(np.float32))

# computing the heatmap of cycles with specified birth-death properties
# the heatmap image reveals what kind of image features are captured by PH
def heat_map(img,pd,h=3,sigma=1,min_life = 10,max_life =255,min_birth = 0,max_birth = 255,dimension = 0,life_weighted=True,location='birth'):
  if len(img.shape)==2:
    mx,my=img.shape
    mz = 1
    kernel = gaussian([h,h,1],sigma)
  else:
    mx,my,mz=img.shape
    kernel = gaussian([h,h,h],sigma)

  selected_cycle = np.zeros((mx,my,mz))
  ppd = pd[pd[:,0] == dimension]
  ppd = ppd[min_life < ppd[:,2]-ppd[:,1]]
  ppd = ppd[ppd[:,2]-ppd[:,1] < max_life]
  ppd = ppd[min_birth < ppd[:,1]]
  ppd = ppd[ppd[:,1] < max_birth]
  w = 1
  for c in ppd:
      if location=='birth':
        x,y,z=int(c[3]),int(c[4]),int(c[5])
      else:
        x,y,z=int(c[6]),int(c[7]),int(c[8])
      if life_weighted:
        w = c[2]-c[1]
      #selected_cycle[max(0,x-h):min(mx,x+h),max(0,y-h):min(my,y+h),max(0,z-h):min(mz,z+h)] += w
      selected_cycle[x,y,z] += w
  #print(np.min(selected_cycle),np.max(selected_cycle),np.sum(selected_cycle))
  cycle_conv = convolve(selected_cycle,kernel)
  #print(np.min(cycle_conv),np.max(cycle_conv),np.sum(cycle_conv))
  return(np.squeeze(cycle_conv))

# slice viewer for a 3D image
def explore_slices(data, cmap="gray"):
    from ipywidgets import interact
    N = data.shape[-1]
    @interact(plane=(0, N - 1))
    def display_slice(plane=N//2):
        fig, ax = plt.subplots(figsize=(20, 5))
        plt.imshow(data[:,:,plane], cmap=cmap)
        plt.show()
    return display_slice

# make binary and apply distance transform
def dt(img,radius=None,signed=False):
    if radius is not None:
      from skimage.filters import rank
      from skimage.morphology import disk, ball
      bw_img = (img >= rank.otsu(img, disk(radius)))
    else:
      bw_img = (img >= threshold_otsu(img))
    dt_img = distance_transform_edt(bw_img)
    if signed:
        dt_img -= distance_transform_edt(~bw_img)
    return(dt_img)


X = np.random.rand(200,3)-0.5  # 200 random points in [-0.5,0.5]^3
X = X / np.sqrt(np.sum(X**2,axis=1,keepdims=True)) # normalize to have the unit length
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2])


diag = ripser(X,maxdim=2,n_perm=100)['dgms']
persim.plot_diagrams(diag)


min_birth, max_death = 0,2
dclip = [np.clip(d,min_birth, max_death) for d in diag]
pimgr = persim.PersistenceImager(pixel_size=0.1, kernel_params={'sigma': [[0.01, 0.0], [0.0, 0.01]]})
pimgr.fit(dclip, skew=True)
pimgs = pimgr.transform(dclip, skew=True,n_jobs=-1)

plt.figure(figsize=(10,5))
for i in range(3):
    ax = plt.subplot(1,3,i+1)
    pimgr.plot_image(pimgs[i], ax)
    plt.title("persistence image of H_{}".format(i))


min_birth, max_death = 0,2
dclip = [np.clip(d,min_birth, max_death) for d in diag]
pimgr = persim.PersistenceImager(pixel_size=0.1, kernel_params={'sigma': [[0.01, 0.0], [0.0, 0.01]]})
pimgr.fit(dclip, skew=True)
pimgs = pimgr.transform(dclip, skew=True,n_jobs=-1)

plt.figure(figsize=(10,5))
for i in range(3):
    ax = plt.subplot(1,3,i+1)
    pimgr.plot_image(pimgs[i], ax)
    plt.title("persistence image of H_{}".format(i))


simplex_tree = gudhi.AlphaComplex(points=X).create_simplex_tree()
diag = simplex_tree.persistence()

gudhi.plot_persistence_diagram(diag, legend=True)



def sample_sphere(n,radius=1):
    X = np.random.rand(n,3)-0.5
    X = X / np.sqrt(np.sum(X**2,axis=1,keepdims=True))
    X *= radius
    X += np.random.normal(scale=0.2,size=(len(X),3))
    X += np.random.uniform(-1,1,3)
    return(X)
def sample_cube(n,radius=1):
    X = 2*np.random.rand(n,3)-1
    X *= radius
    X += np.random.uniform(-1,1,3)
    return(X)


n=5   # number of spheres (cubes). In total, we'll have 2n sets of point clouds.
X=[sample_sphere(200) for i in range(n)]
Y=[0]*n
X.extend([sample_cube(200) for i in range(n)])
Y.extend([1]*n)

## Plot the point clouds: for human eyes, it is not very easy to distinguish 3d point clouds
# sphere
fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(121, projection='3d')
ax.scatter(X[0][:, 0], X[0][:, 1], X[0][:, 2])
# cube
ax = fig.add_subplot(122, projection='3d')
ax.scatter(X[n][:, 0], X[n][:, 1], X[n][:, 2])

# compute PH and distance
pd = []
for i in range(2*n):
    pd.append(ripser(X[i],maxdim=2,n_perm=100)['dgms'][2])

D = np.zeros((2*n,2*n))
for i in range(2*n-1):
    for j in range(i+1,2*n):
        D[i,j]=persim.bottleneck(pd[i], pd[j])



from sklearn.manifold import MDS
mds = MDS(n_components=2,dissimilarity='precomputed')
D = D+D.T
X_mds = mds.fit_transform(D)
col = ['r','b']
plt.scatter(X_mds[:,0],X_mds[:,1],c=[col[y] for y in Y])



import gudhi,gudhi.hera,gudhi.wasserstein,persim
X,Y =np.array([[0., 0.01]]), np.array([[0., 13.],[0.,12.]])
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(121)
persim.plot_diagrams(X, ax=ax)
ax = fig.add_subplot(122)
persim.plot_diagrams(Y, ax=ax)

print("Distances")
print("Bottleneck (GUDHI-hera) with L-infty metric:", gudhi.bottleneck_distance(X,Y))
print("Bottleneck (persim) with L-infty metric:", persim.bottleneck(X,Y))
#print(gudhi.wasserstein.wasserstein_distance(X, Y, order=1, internal_p=2)) ## requires pot
print("2-Wasserstein (GUDHI-hera) with L-infty metric:", gudhi.hera.wasserstein_distance(X, Y, order=1, internal_p=np.inf))
print("2-Wasserstein (GUDHI-hera) with L2 Euclidean metric:", gudhi.hera.wasserstein_distance(X, Y, order=1, internal_p=2))
print("99-Wasserstein (GUDHI-hera) with L2 Euclidean metric:", gudhi.hera.wasserstein_distance(X, Y, order=99, internal_p=2))
print("99-Wasserstein (GUDHI-hera) with L-infty metric (approx. Bottleneck distance):", gudhi.hera.wasserstein_distance(X, Y, order=99, internal_p=np.inf))



import networkx as nx
G=nx.dodecahedral_graph()
nx.draw(G)
D=np.array(nx.floyd_warshall_numpy(G)) # distance matrix



diag = ripser(D,distance_matrix=True)['dgms']
persim.plot_diagrams(diag)



def create_figure8(num_samples=200):
  t = np.linspace(0, 2*np.pi, num=num_samples)
  X = np.stack((np.sqrt(2)*np.cos(t) / (np.sin(t)**2+1), np.sqrt(2)*np.cos(t)*np.sin(t) / (np.sin(t)**2+1))).T
  X += 0.1*np.random.random(X.shape)
  return(X)



Xs = []
diags = []
frames = 30
for fr in range(frames):
  a = frames/2
  X = create_figure8() * (a**2-(fr-a)**2)/(a**2)
  Xs.append(X)
  diags.append(ripser(Xs[fr],maxdim=1,n_perm=100)['dgms'])



import matplotlib.colors as colors
import matplotlib.cm as cm
def cycle_trajectory(diags,dim):
  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1)
  plt.xlim(0,0.3)
  plt.ylim(0,1)
  usercmap = plt.get_cmap('jet')
  cNorm  = colors.Normalize(vmin=0, vmax=len(diags))
  scalarMap = cm.ScalarMappable(norm=cNorm, cmap=usercmap)
  for i in range(len(diags)-1):
    D1 = diags[i][dim]
    D2 = diags[i+1][dim]
    d, M= persim.wasserstein(D1,D2,matching=True)
    for m in M:
      m0 = int(m[0]) # matched ids
      m1 = int(m[1])
      if -1<m0<len(D1) and -1<m1<len(D2):
        plt.plot( [D1[m0][0], D2[m1][0]],[D1[m0][1], D2[m1][1]],  'k-', lw=1)
#        plt.plot(*D1[m0],'ko')
        plt.plot(*D2[m1],'o',color = scalarMap.to_rgba(i))
  cax = fig.add_axes([1,0,0.1,0.05,0.8])
  plt.colorbar(scalarMap, cax=cax, label="time")
  plt.xlabel("AD")
  plt.ylabel("MCI")
  plt.show()




cycle_trajectory(diags,dim=1)



import cripser, tcripser
import persim

# define a 2D array
simple_img=np.array([[0,0,0,0,0],
                      [0,0,1,2,0],
                      [1,1,1,2,0],
                      [0,1,0,0,0]])

# plot the array
fig,axs = plt.subplots(1,3,figsize=(12,4))
sns.heatmap(simple_img, annot=True, square=True, yticklabels=False, xticklabels=False, annot_kws={"fontsize":20}, cbar=False, ax=axs[0])

# compute PH of the 2D array
# in this particular example, we do not see any difference
pd = cripser.computePH(simple_img) # V-construction
pdt = tcripser.computePH(simple_img) # T-construction

# each line contains (dim,birth,death,x1,y1,z1,x2,y2,z2), where (x1,y1,z1) is the location of the creator cell of the cycle and (x2,y2,z2) is the location of the destroyer cell of the cycle"
print("[dim,AD,MCI,x1,y1,z1,x2,y2,z2]: V-construction")
print(np.where(pd<9, pd, 9).astype(int)) # replace infty with 9 for printing
print("[dim,AD,MCI,x1,y1,z1,x2,y2,z2]: T-construction")
print(np.where(pdt<9, pdt, 9).astype(int)) # replace infty with 9 for printing

# plot persistence diagram
persim.plot_diagrams([p[:,1:3] for p in [pd[pd[:,0] == i] for i in range(2)]], ax=axs[1], title='V-construction')
persim.plot_diagrams([p[:,1:3] for p in [pdt[pdt[:,0] == i] for i in range(2)]], ax=axs[2], title='T-construction')






simple_img=np.array([[0,0,0,0,0],
                      [0,0,1,2,0],
                      [1,0,1,2,0],
                      [0,1,0,0,0]])

# plot the array
fig,axs = plt.subplots(1,3,figsize=(12,4))
sns.heatmap(simple_img, annot=True, square=True, yticklabels=False, xticklabels=False, annot_kws={"fontsize":20}, cbar=False, ax=axs[0])

# compute PH of the 2D array
pd = cripser.computePH(simple_img) # V-construction
pdt = tcripser.computePH(simple_img) # T-construction

# each line contains (dim,birth,death,x1,y1,z1,x2,y2,z2), where (x1,y1,z1) is the location of the creator cell of the cycle and (x2,y2,z2) is the location of the destroyer cell of the cycle"
print("[dim,birth,death,x1,y1,z1,x2,y2,z2]: V-construction")
print(np.where(pd<9, pd, 9).astype(int)) # replace infty with 9 for printing
print("[dim,birth,death,x1,y1,z1,x2,y2,z2]: T-construction")
print(np.where(pdt<9, pdt, 9).astype(int)) # replace infty with 9 for printing

# plot persistence diagram
persim.plot_diagrams([p[:,1:3] for p in [pd[pd[:,0] == i] for i in range(2)]], ax=axs[1], title='V-construction')
persim.plot_diagrams([p[:,1:3] for p in [pdt[pdt[:,0] == i] for i in range(2)]], ax=axs[2], title='T-construction')




simple_img=np.array([[0,1,1,1,0,0,0],
                      [1,1,1,0,0,0,0],
                      [1,1,0,0,0,0,0],
                      [0,0,0,0,0,1,1]],dtype=bool)

# apply distance transform
dt_img = dt(simple_img,signed=True)

# plot the array
fig,axs = plt.subplots(1,3,figsize=(12,4))
sns.heatmap(simple_img, annot=True, square=True, yticklabels=False, xticklabels=False, annot_kws={"fontsize":20}, cbar=False, ax=axs[0])
sns.heatmap(dt_img, annot=True, square=True, yticklabels=False, xticklabels=False, annot_kws={"fontsize":20}, cbar=False, ax=axs[1])

# compute PH of the 2D array (the V-construction)
pdt = cripser.computePH(dt_img)

# each line contains (dim,birth,death,x1,y1,z1,x2,y2,z2), where (x1,y1,z1) is the location of the creator cell of the cycle and (x2,y2,z2) is the location of the destroyer cell of the cycle"
# print("[dim,birth,death,x1,y1,z1,x2,y2,z2]")
# print(np.where(pdt<9, pdt, 9).astype(int)) # replace infty with 9 for printing

# plot persistence diagram
persim.plot_diagrams([p[:,1:3] for p in [pdt[pdt[:,0] == i] for i in range(2)]], ax=axs[2])




import skimage
import io
from google.colab import files

# download an image from the 1yr 1.5t and upload it to your google drive

uploaded = files.upload()

img = skimage.io.imread(list(uploaded.keys())[0])

img = skimage.color.rgb2gray(img)

# print(np.min(img), np.max(img))

# plt.imshow(img, cmap='gray')




print(np.min(img), np.max(img))
pd = cripser.computePH(img, maxdim=1)

# each line contains (dim,birth,death,x1,y1,z1,x2,y2,z2), where (x1,y1,z1) is the location of the creator cell of the cycle and (x2,y2,z2) is the location of the destroyer cell of the cycle
# print("[dim,birth,death,x1,y1,z1,x2,y2,z2]")
# print(pd[:5])

# arrange degree-wise
pds = [pd[pd[:,0] == i] for i in range(2)]
# print("Betti numbers: ",[len(pds[i]) for i in range(len(pds))])

# T-construction of the original image (pixel value filtration)
pdt = tcripser.computePH(img,maxdim=1)
pdst = [pdt[pdt[:,0] == i] for i in range(2)]
# print("Betti numbers: ",[len(pdst[i]) for i in range(len(pdst))])

## plot persistent diagram using persim
fig,axs = plt.subplots(1,2)
persim.plot_diagrams([p[:,1:3] for p in pds], ax=axs[0], title='V-construction')
persim.plot_diagrams([p[:,1:3] for p in pdst], ax=axs[1], title='T-construction')



from scipy.ndimage import distance_transform_edt
from skimage.filters import threshold_otsu
bw_img = (img >= threshold_otsu(img)) # binarise by Otsu's method
dt_img = dt(img,signed=True)
fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(121)
ax.imshow(bw_img,cmap='gray')
ax = fig.add_subplot(122)
# ax.imshow(dt_img, cmap='gray')



pd = cripser.computePH(dt_img)
pds = [pd[pd[:,0] == i] for i in range(2)]
print("Betti numbers: ",[len(pds[i]) for i in range(len(pds))])

# T-construction of the distance transformed image
pdt = tcripser.computePH(dt_img)
pdst = [pdt[pdt[:,0] == i] for i in range(2)]
print("Betti numbers: ",[len(pdst[i]) for i in range(len(pdst))])

## plot persistent diagram using persim
fig,axs = plt.subplots(1,2)
persim.plot_diagrams([p[:,1:3] for p in pds], ax=axs[0], title='V-construction')
persim.plot_diagrams([p[:,1:3] for p in pdst], ax=axs[1], title='T-construction')



pd = cripser.computePH(img,maxdim=1)
fig = plt.figure(figsize=(8, 4))
for i in range(2):
  heat=heat_map(img,pd,h=20,sigma=10,min_birth=10,min_life=30,dimension=i,location='birth')
  heat = (heat/np.max(heat) * 255).astype(np.uint8)
  ax = fig.add_subplot(1,2,i+1)
  ax.imshow(np.dstack([heat,img//2,img//2]))
  ax.set_title("density of H{}".format(i))




from skimage import data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report

## create a binary classification problem consisting of random blobs of different size
n = 100
X = [data.binary_blobs(length=100, blob_size_fraction=0.1) for i in range(n)] # class A: small size
X.extend([data.binary_blobs(length=100, blob_size_fraction=0.15) for i in range(n)]) # class B: large size

# class label
Y = [0 for i in range(n)]
Y.extend([1 for i in range(n)])

# plot
fig,axs = plt.subplots(2,10,figsize=(20,4))
for i in range(10):
  axs[0,i].imshow(X[i])
  axs[1,i].imshow(X[i+n])
  axs[0,i].axis('off')
  axs[1,i].axis('off')




import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

pds = []
for i in range(len(X)):
    pd = cripser.computePH(dt(X[i],signed=True))    # compute PH of distance transform
    pds.append(pd)

# plot some of them
fig,axs = plt.subplots(4,10,figsize=(20,8))
for i in range(10):
  axs[0,i].imshow(dt(X[i],signed=True))
  axs[0,i].axis('off')
  pd=pds[i]
  persim.plot_diagrams([p[:,1:3] for p in [pd[pd[:,0] == i] for i in range(2)]],ax = axs[1,i], legend=False)
  axs[2,i].imshow(dt(X[i+n],signed=True))
  axs[2,i].axis('off')
  pd=pds[i+n]
  persim.plot_diagrams([p[:,1:3] for p in [pd[pd[:,0] == i] for i in range(2)]],ax = axs[3,i], legend=False)





import PersistenceImages.persistence_images as pimg
max_life = 8
pixel_size = 1
pim = pimg.PersistenceImager(birth_range=(-max_life,max_life), pers_range=(0,max_life), pixel_size=pixel_size)

idx = [0,n]
nn=len(idx)
fig,axs = plt.subplots(nn,5,figsize=(nn*5,4))
for i in range(nn):
  img = X[idx[i]]*255
  pd = pds[idx[i]]
  for d in range(2):
    heat=heat_map(img,pd,h=5,sigma=1,min_life=0,max_life=max_life,min_birth =-max_life,max_birth = max_life,dimension=d,location='death')
    heat = (heat/max(1,np.max(heat))*255).astype(np.uint8)
    axs[i,0+d].imshow(np.dstack([heat,img//2,img//2]))
    axs[i,0+d].set_title("density of H{}".format(d))
    axs[i,0+d].axis('off')
    axs[i,2+d].imshow(pim.transform(pd[pd[:,0]==d,1:3]))
    axs[i,2+d].set_title("image of H{}".format(d))
    axs[i,2+d].axis('off')
  persim.plot_diagrams([p[:,1:3] for p in [pd[pd[:,0] == i] for i in range(2)]],ax = axs[i,4])
