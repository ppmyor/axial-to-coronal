import pydicom
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import sys
import glob
import cv2
from PIL import Image



#file path
path = 'axial_bonemeta/training/meta_data/*.dcm'
path_nii = 'axial_bonemeta/training/meta_data/meta_data.nii.gz'
path_img_save = 'axial_bonemeta/training_stacking/meta_data'


# load the NifTi file 
data = nib.load(path_nii)
img = data.get_fdata()


# load the DICOM files
files = []

print('glob: {}'.format(sys.argv[1]))
for fname in glob.glob(path, recursive=False):
    print("loading: {}".format(fname))
    files.append(pydicom.dcmread(fname))
    
files_count = int(len(files))
print("file count: {}".format(files_count))


# skip files with no SliceLocation (eg scout views)
slices = []
skipcount = 0
for f in files:
    if hasattr(f, 'SliceLocation'):
        slices.append(f)
    else:
        skipcount = skipcount + 1

print("skipped, no SliceLocation: {}".format(skipcount))

# ensure they are in the correct order
slices = sorted(slices, key=lambda s: s.SliceLocation)


# pixel aspects, assuming all slices are the same
ps = slices[0].PixelSpacing
ss = slices[0].SliceThickness
ax_aspect = ps[1]/ps[0]
sag_aspect = ps[1]/ss
cor_aspect = ss/ps[0]

# create 3D array
img_shape = list(slices[0].pixel_array.shape)
img_shape.append(len(slices))
img3d = np.zeros(img_shape)


# fill 3D array with the images from the files
for i, s in enumerate(slices):
    img2d = s.pixel_array
    img3d[:, :, i] = img2d
    
############show coronal############ 


n_co = 310

# plot 3 orthogonal slices_coronal
a3 = plt.subplot(2, 2, 3)
img_mirror = cv2.flip(img3d[n_co-1, :, :].T, 0) #img mirror
plt.imshow(img_mirror, cmap="gray")
plt.imsave(path_img_save+"dcm_coronal_image%d.png"%(n_co), img_mirror, cmap="gray") #dcm_coronal_img_save
a3.set_aspect(cor_aspect)


# NifTi masking_coronal
a4 = plt.subplot(2, 2, 4)
img_mirror = cv2.flip(img[:,n_co-1,:].T, 0) #img mirror
plt.imshow(img_mirror, cmap="gray")
plt.imsave(path_img_save+"nii_coronal_image%d.png"%(n_co), img_mirror, cmap=plt.cm.Reds)
plt.imsave(path_img_save+"nii_coronal_gray_image%d.png"%(n_co), img_mirror, cmap="gray")#nii_coronal_img_save
a4.set_aspect(cor_aspect)

#plt.rcParams["figure.figsize"] = (15,15)
#plt.show()

#dcm coronal image resize
dcm_co_img = Image.open(path_img_save+"dcm_coronal_image%d.png"%(n_co))

dcm_co_img_resize = dcm_co_img.resize((300, 500))
dcm_co_img_resize.save(path_img_save+"dcm_coronal_resize_image%d.png"%(n_co))


#nii coronal image resize
nii_co_img = Image.open(path_img_save+"nii_coronal_image%d.png"%(n_co))

nii_co_img_resize = nii_co_img.resize((300, 500))
nii_co_img_resize.save(path_img_save+"nii_coronal_resize_image%d.png"%(n_co))


#nii and dcm merge img code!!

alpha = 0.65


#coronal merge img!!
dcm_coronal_image = cv2.imread(path_img_save+'dcm_coronal_resize_image%d.png'%(n_co))
nii_coronal_image = cv2.imread(path_img_save+'nii_coronal_resize_image%d.png'%(n_co))

co_result = cv2.addWeighted(dcm_coronal_image, alpha, nii_coronal_image, (1-alpha), 0)
plt.imsave(path_img_save+'merge_coronal_image%d.png'%(n_co), co_result)
print("merge axial image index:", n_co)