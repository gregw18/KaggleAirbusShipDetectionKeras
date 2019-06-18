# Kaggle Airbus Ship Detection Using Keras

A series of Jupyter notebooks and associated python files for training a resnet50 on a subset of the images from the Kaggle Airbus Ship Detection Challenge. Manually selected subsets of the images containing boats and not containing boats. Wanted good boat images as seemed that there were many flagged as containing boats that didn't have any visible signs of boats. Picked a representative set of no-boat images, as most were very similar images of open water and I wanted to balance those with ones including shoreline, docks, land, etc. Also, didn't want to use the entire set at this point, as training with everything was rather slow.
See TrainingNotes.txt for details on process followed.

## Getting Started

Copy the Jupyter notebooks and helper python files. Download the source data from the Kaggle competition (https://www.kaggle.com/c/airbus-ship-detection/overview). Run CreateSegMasks.py (from my KaggleAirbusShipDetection project), after setting up appropriate paths, to create segmentation masks. Run viewsegs.py, after setting up appropriate paths, to pick out images to train on. (I used around 650, 75% containing boats.) Then, run the notebook, see if you can get it to train!


### Prerequisites

Python (tested with 3.6), matplotlib, numpy, pandas, fastai, unittest, 


## Authors

* **Greg Walker** - *Initial work* - (https://github.com/gregw18)


## License

MIT


