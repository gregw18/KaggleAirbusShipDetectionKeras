# ****************** 2nd attempt
# Utility to display original image, target mask and predicted mask, for each
# file in given list.

import os, shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def displayFiles(file_names, image_dir, targ_dir, predictions):
    # Display image, name, target mask and predictions for each file.
    # Receives: list of filenames, paths to image and label directories, corresponding list of predictions.

    numFiles = len(file_names)
    i = 0
    n = 1
    for filename in file_names:
        filename = filename.rstrip()
        if len(filename) > 0:
            fig = plt.figure(figsize=(12,8))
            axarr = fig.subplots(1, 3, subplot_kw=dict(frameon=False))
            base, ext = os.path.splitext(filename)
            imgFname = os.path.join(image_dir, filename)
            targetFname = os.path.join(targ_dir, base + ".png")
            targArr = mpimg.imread(targetFname) * 254
            axarr[0].imshow(mpimg.imread(imgFname))
            axarr[1].imshow(targArr)
            axarr[2].imshow(predictions[i,:,:,0])

        axarr[0].axis('off')
        axarr[1].axis('off')
        axarr[2].axis('off')
        plt.show()
        n += 1
        i += 1

    return
