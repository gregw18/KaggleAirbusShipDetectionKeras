# Program to ensure have same filenames in images and masks directory - no more and no less.
# May 31, 2019

# For each file in first directory, see if same file exists in second.
# If not, delete from first directory. If yes, mark file in second directory as confirmed.
# At end, go through list of files in second directory, delete those not confirmed.

import os

baseDir = '/home/ubuntu/notebooks/shipid/data/morebigboats'
imgDir = os.path.join(baseDir, 'images/boats/')
maskDir = os.path.join(baseDir, 'masks/boats/')

imgFiles = [name for name in os.listdir(imgDir)
        if os.path.isfile(os.path.join(imgDir, name)) and
                name.endswith('.jpg')]
maskFiles = [name for name in os.listdir(maskDir)
        if os.path.isfile(os.path.join(maskDir, name)) and
                name.endswith('.png')]

# Create dictionary of mask files, default value to 0 for each.
# Will increment count when find corresponding image file.
# At end, delete files from mask dir whose value is still 0.
foundMasks = {}
for maskFile in maskFiles:
        foundMasks[maskFile] = 0

# Increment value in foundMasks for each corresponding image file.
# If don't find corresponding mask file, delete image.
deletedImages = ''
numDelImages = 0
for imgFile in imgFiles:
        base, ext = os.path.splitext(imgFile)
        targetMask = base + ".png"
        if targetMask in foundMasks:
                foundMasks[targetMask] += 1
        else:
                # Delete the image.
                deletedImages = deletedImages + imgFile + ", "
                numDelImages += 1
                print( "Deleting image file ", imgFile)
                os.remove(os.path.join(imgDir, imgFile))

# Delete masks which don't have corresponding image files.
deletedMasks = ''
numDelMasks = 0
for maskFile in foundMasks:
        if foundMasks[maskFile] == 0:
                # Delete the mask.
                deletedMasks = deletedMasks + maskFile + ", "
                numDelMasks += 1
                print( "Deleting mask file ", maskFile )
                os.remove(os.path.join(maskDir, maskFile))

print( "Deleted ", numDelImages, " image files and ", numDelMasks, " mask files.")
print( "Deleted image files: ", deletedImages)
print( "Deleted mask files: ", deletedMasks)

