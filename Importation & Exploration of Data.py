import zipfile

# Put on the same directory
from zipfile import ZipFile

# specifying the name of the zip file
file = '/content/ADNI1_Complete 1Yr 1.5T.zip'

# open the zip file in read mode
with ZipFile(file, 'r') as zip_ref:
    # extract all files
    zip_ref.extractall('/content')
with ZipFile(file, 'r') as zip:
    # list all the contents of the zip file
    zip.printdir()

    # extract all files
    print('extraction...')
    zip.extractall()
    print('Done!')


#Import necessary libraries
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import cv2

#Define the filepath to your NIfTI scan
scanFilePath = 'ADNI/023_S_0084/MPR__GradWarp__B1_Correction__N3__Scaled/2006-07-27_16_34_30.0/I31161/ADNI_023_S_0084_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20061130201833308_S17395_I31161.nii'

#Load the scan and extract data using nibabel
scan = nib.load(scanFilePath)
scanArray = scan.get_fdata()

#Get and print the scan's shape
scanArrayShape = scanArray.shape
# print('The scan data array has the shape: ', scanArrayShape)

# #Get and print the scan's header
scanHeader = scan.header
# print('The scan header is as follows: \n', scanHeader)

# #Display scan array's middle slices
# fig, axs = plt.subplots(1,3)
# fig.suptitle('Scan Array (Middle Slices)')
# axs[0].imshow(scanArray[scanArrayShape[0]//2,:,:], cmap='gray')
# axs[1].imshow(scanArray[:,scanArrayShape[1]//2,:], cmap='gray')
# axs[2].imshow(scanArray[:,:,scanArrayShape[2]//2], cmap='gray')
# fig.tight_layout()
# plt.show()

#Calculate proper aspect ratios
pixDim = scanHeader['pixdim'][1:4]
aspectRatios = [pixDim[1]/pixDim[2],pixDim[0]/pixDim[2],pixDim[0]/pixDim[1]]
# print('The required aspect ratios are: ', aspectRatios)

#Display scan array's middle slices with proper aspect ratio
# fig, axs = plt.subplots(1,3)
# fig.suptitle('Scan Array w/ Proper Aspect Ratio (Middle Slices)')
# axs[0].imshow(scanArray[scanArrayShape[0]//2,:,:], aspect = aspectRatios[0], cmap='gray')
# axs[1].imshow(scanArray[:,scanArrayShape[1]//2,:], aspect = aspectRatios[1], cmap='gray')
# axs[2].imshow(scanArray[:,:,scanArrayShape[2]//2], aspect = aspectRatios[2], cmap='gray')
# fig.tight_layout()
# plt.show()

#Calculate new image dimensions from aspect ratio
newScanDims = np.multiply(scanArrayShape, pixDim)
newScanDims = (round(newScanDims[0]),round(newScanDims[1]),round(newScanDims[2]))
# print('The new scan size is: ', newScanDims)

#Set the output file path
outputPath = 'output/'

#Iterate and save scan slices along 0th dimension
for i in range(scanArrayShape[0]):
    #Resample the slice
    outputArray = cv2.resize(scanArray[i,:,:], (newScanDims[2],newScanDims[1]))
    #Save the slice as .png image
    cv2.imwrite(outputPath+'Dim0_Slice'+str(i)+'.png', outputArray)

#Iterate and save scan slices along 1st dimension
for i in range(scanArrayShape[1]):
    #Resample the slice
    outputArray = cv2.resize(scanArray[:,i,:], (newScanDims[2],newScanDims[0]))
    #Save the slice as .png image
    cv2.imwrite(outputPath+'Dim1_Slice'+str(i)+'.png', outputArray)

#Iterate and save scan slices along 2nd dimension
for i in range(scanArrayShape[2]):
    #Resample the slice
    outputArray = cv2.resize(scanArray[:,:,i], (newScanDims[1],newScanDims[0]))
    #Rotate slice clockwise 90 degrees
    outputArray = cv2.rotate(outputArray, cv2.ROTATE_90_CLOCKWISE)
    #Save the slice as .png image
    cv2.imwrite(outputPath+'Dim2_Slice'+str(i)+'.png', outputArray)



#Import necessary libraries
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import cv2

#Define the filepath to your NIfTI scan
scanFilePath = 'ADNI/005_S_0546/MPR-R__GradWarp__B1_Correction__N3__Scaled/2007-02-02_11_11_15.0/I68423/ADNI_005_S_0546_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_Br_20070818113852943_S25839_I68423.nii'

#Load the scan and extract data using nibabel
scan = nib.load(scanFilePath)
scanArray = scan.get_fdata()

#Get and print the scan's shape
scanArrayShape = scanArray.shape
# print('The scan data array has the shape: ', scanArrayShape)

#Get and print the scan's header
scanHeader = scan.header
# print('The scan header is as follows: \n', scanHeader)

#Display scan array's middle slices
# fig, axs = plt.subplots(1,3)
# fig.suptitle('Scan Array (Middle Slices)')
# axs[0].imshow(scanArray[scanArrayShape[0]//2,:,:], cmap='gray')
# axs[1].imshow(scanArray[:,scanArrayShape[1]//2,:], cmap='gray')
# axs[2].imshow(scanArray[:,:,scanArrayShape[2]//2], cmap='gray')
# fig.tight_layout()
# plt.show()

#Calculate proper aspect ratios
pixDim = scanHeader['pixdim'][1:4]
aspectRatios = [pixDim[1]/pixDim[2],pixDim[0]/pixDim[2],pixDim[0]/pixDim[1]]
# print('The required aspect ratios are: ', aspectRatios)

#Display scan array's middle slices with proper aspect ratio
# fig, axs = plt.subplots(1,3)
# fig.suptitle('Scan Array w/ Proper Aspect Ratio (Middle Slices)')
# axs[0].imshow(scanArray[scanArrayShape[0]//2,:,:], aspect = aspectRatios[0], cmap='gray')
# axs[1].imshow(scanArray[:,scanArrayShape[1]//2,:], aspect = aspectRatios[1], cmap='gray')
# axs[2].imshow(scanArray[:,:,scanArrayShape[2]//2], aspect = aspectRatios[2], cmap='gray')
# fig.tight_layout()
# plt.show()

#Calculate new image dimensions from aspect ratio
newScanDims = np.multiply(scanArrayShape, pixDim)
newScanDims = (round(newScanDims[0]),round(newScanDims[1]),round(newScanDims[2]))
# print('The new scan size is: ', newScanDims)

#Set the output file path
outputPath = 'output/'

#Iterate and save scan slices along 0th dimension
for i in range(scanArrayShape[0]):
    #Resample the slice
    outputArray = cv2.resize(scanArray[i,:,:], (newScanDims[2],newScanDims[1]))
    #Save the slice as .png image
    cv2.imwrite(outputPath+'Dim0_Slice'+str(i)+'.png', outputArray)

#Iterate and save scan slices along 1st dimension
for i in range(scanArrayShape[1]):
    #Resample the slice
    outputArray = cv2.resize(scanArray[:,i,:], (newScanDims[2],newScanDims[0]))
    #Save the slice as .png image
    cv2.imwrite(outputPath+'Dim1_Slice'+str(i)+'.png', outputArray)

#Iterate and save scan slices along 2nd dimension
for i in range(scanArrayShape[2]):
    #Resample the slice
    outputArray = cv2.resize(scanArray[:,:,i], (newScanDims[1],newScanDims[0]))
    #Rotate slice clockwise 90 degrees
    outputArray = cv2.rotate(outputArray, cv2.ROTATE_90_CLOCKWISE)
    #Save the slice as .png image
    cv2.imwrite(outputPath+'Dim2_Slice'+str(i)+'.png', outputArray)



#Import necessary libraries
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import cv2

#Define the filepath to your NIfTI scan
scanFilePath = 'ADNI/023_S_0042/MPR-R__GradWarp__B1_Correction__N3__Scaled/2005-10-31_12_03_30.0/I31084/ADNI_023_S_0042_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_Br_20061130173634563_S8852_I31084.nii'

#Load the scan and extract data using nibabel
scan = nib.load(scanFilePath)
scanArray = scan.get_fdata()

#Get and print the scan's shape
scanArrayShape = scanArray.shape
# print('The scan data array has the shape: ', scanArrayShape)

#Get and print the scan's header
scanHeader = scan.header
# print('The scan header is as follows: \n', scanHeader)

#Display scan array's middle slices
# fig, axs = plt.subplots(1,3)
# fig.suptitle('Scan Array (Middle Slices)')
# axs[0].imshow(scanArray[scanArrayShape[0]//2,:,:], cmap='gray')
# axs[1].imshow(scanArray[:,scanArrayShape[1]//2,:], cmap='gray')
# axs[2].imshow(scanArray[:,:,scanArrayShape[2]//2], cmap='gray')
# fig.tight_layout()
# plt.show()

#Calculate proper aspect ratios
pixDim = scanHeader['pixdim'][1:4]
aspectRatios = [pixDim[1]/pixDim[2],pixDim[0]/pixDim[2],pixDim[0]/pixDim[1]]
# print('The required aspect ratios are: ', aspectRatios)

#Display scan array's middle slices with proper aspect ratio
# fig, axs = plt.subplots(1,3)
# fig.suptitle('Scan Array w/ Proper Aspect Ratio (Middle Slices)')
# axs[0].imshow(scanArray[scanArrayShape[0]//2,:,:], aspect = aspectRatios[0], cmap='gray')
# axs[1].imshow(scanArray[:,scanArrayShape[1]//2,:], aspect = aspectRatios[1], cmap='gray')
# axs[2].imshow(scanArray[:,:,scanArrayShape[2]//2], aspect = aspectRatios[2], cmap='gray')
# fig.tight_layout()
# plt.show()

#Calculate new image dimensions from aspect ratio
newScanDims = np.multiply(scanArrayShape, pixDim)
newScanDims = (round(newScanDims[0]),round(newScanDims[1]),round(newScanDims[2]))
# print('The new scan size is: ', newScanDims)

#Set the output file path
outputPath = 'output/'

#Iterate and save scan slices along 0th dimension
for i in range(scanArrayShape[0]):
    #Resample the slice
    outputArray = cv2.resize(scanArray[i,:,:], (newScanDims[2],newScanDims[1]))
    #Save the slice as .png image
    cv2.imwrite(outputPath+'Dim0_Slice'+str(i)+'.png', outputArray)

#Iterate and save scan slices along 1st dimension
for i in range(scanArrayShape[1]):
    #Resample the slice
    outputArray = cv2.resize(scanArray[:,i,:], (newScanDims[2],newScanDims[0]))
    #Save the slice as .png image
    cv2.imwrite(outputPath+'Dim1_Slice'+str(i)+'.png', outputArray)

#Iterate and save scan slices along 2nd dimension
for i in range(scanArrayShape[2]):
    #Resample the slice
    outputArray = cv2.resize(scanArray[:,:,i], (newScanDims[1],newScanDims[0]))
    #Rotate slice clockwise 90 degrees
    outputArray = cv2.rotate(outputArray, cv2.ROTATE_90_CLOCKWISE)
    #Save the slice as .png image
    cv2.imwrite(outputPath+'Dim2_Slice'+str(i)+'.png', outputArray)
