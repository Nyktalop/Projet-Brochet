from imageio import imread
from skimage import exposure, filters, color, morphology
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_hit_or_miss
import scipy.ndimage as ndi
import numpy as np

#Use gamma correction or log correction
gamma = True

#gamma correction factor
gamma_corr = 3

#log correction factor
log_corr = 2

#struct elem disk diameter
disk_size = 5

#proportion cutoff of identified spots based on their mean
mean_reg = 1.5

##-- Assume pre identification of ROI

# im = imread("IMG_0050.jpg")
# impoiss = im[1100:1375,1510:2600]

# im = imread("4677_101114_04G.JPG")
# impoiss = im[1320:1670, 950:2675]

# im = imread("002B_101114_20D.JPG")
# impoiss = im[1500:1800,1350:2450]

im = imread("003A_101114_12D.JPG")
impoiss = im[1380:1720,1200:2300]

##--

#Image exposure correction
if gamma :
    impoiss_corr = exposure.rescale_intensity(exposure.adjust_gamma(impoiss,gamma_corr))
else :
    impoiss_corr = exposure.rescale_intensity(exposure.adjust_log(impoiss))
    
#from color to grey    
imp_cor_g = color.rgb2gray(impoiss_corr)
#otsu binarisation
bin_cor = imp_cor_g > filters.threshold_otsu(imp_cor_g)
#removal of non significant artifacts (scales reflects etc..)
mor_cor = morphology.binary_opening(bin_cor,selem=morphology.disk(disk_size))

#automatic labelling
labeled_spots, _ = ndi.label(mor_cor)

#extractions of label and counts
unique, counts = np.unique(labeled_spots,return_counts=True)
#label 0 is background
counts = counts[1:]

#elimination of outliers
count_mean = np.mean(counts)
for i, count in enumerate(counts):
    if count > count_mean*mean_reg  or count < count_mean/(mean_reg*2) :
        #label i+1 is set as background
        unique[i+1] = 0

#repercussion of the eleminitation
labeled_spots2 = np.copy(labeled_spots)

for i in range(len(labeled_spots)) :
    for j in range(len(labeled_spots[i])) :
        if not labeled_spots[i][j] in unique :
            labeled_spots2[i][j] = 0

#"graphical" display of the results
centers = np.zeros_like(labeled_spots)
for val in unique :
    if val != 0 :
        indices = np.where(labeled_spots == val)
        x = int(np.mean(indices[0]))
        y = int(np.mean(indices[1]))

        centers[x,y] = 1
        centers[x + 1, y] = 1
        centers[x + 2, y] = 1
        centers[x - 1, y] = 1
        centers[x - 2, y] = 1
        centers[x + 3, y] = 1
        centers[x - 3, y] = 1
        centers[x,y+1] = 1
        centers[x,y+2] = 1
        centers[x,y+3] = 1
        centers[x,y-2] = 1
        centers[x,y-1] = 1
        centers[x,y-3] = 1

#graphical display over the original picture
comp = np.copy(impoiss)

for i in range(len(centers)):
    for j in range(len(centers[i])) :
        if centers[i][j] == 1 :
            comp[i,j,:] = (255,0,0)

#display
base = 421
plt.figure(1)

plt.subplot(base)
plt.imshow(impoiss)
plt.subplot(base+1)
plt.imshow(impoiss_corr)
plt.subplot(base+2)
plt.imshow(bin_cor,cmap="Greys_r")
plt.subplot(base+3)
plt.imshow(mor_cor,cmap="Greys_r")
plt.subplot(base+4)
plt.imshow(labeled_spots)
plt.subplot(base+5)
plt.imshow(labeled_spots2)
plt.subplot(base+6)
plt.imshow(centers,cmap="Greys_r")
plt.subplot(base+7)
plt.imshow(comp)





plt.show()
