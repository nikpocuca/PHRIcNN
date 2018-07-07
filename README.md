# PHRI Convolutional Neural Network


## Setting up python

module avail python
module load version


## virtualenv

source ENV/bin/activate
deactivate

## June 14th  

The new batch are in png form, and they are in the following location

data/newRescored/pngsNeedConvert

The fully sorted and rescored data is here:

	data/newRescored
with the train and test sets in that dir. 

Everything has been fully renamed correctly as in everything in a pos directory is n1,
 and everything in a neg is n0.

* convert pngs add them to the full dataset, and then re-run the model. 
You should have now 9K images.


## June 15th 

Data is now fully converted over to jpgs in a dir location:
data/newRescored/pngsNeedConvert/raw-data


## June 15th 

renamed some directories, everything is basically lower case now.

* wrote a new image conversion script image2square which pads a image
up to a square based on largest dimension.
* ran the conversion on all the png
* split the converted images into pos and neg, and 
* the new directory to be trained from can be found in.

	data/new-rescored/train

* new images were only added to the training directory not test.

## June 16th 

All images less than 200x200 have been removed, keep this in mind for future. 

## June 24th

The script datasetup takes two parameters now.

usage:

./datasetup n1 .93, 

Means to seperate positive examples that have n1 and negative examples that 
don't. .93 means to use 93% of the data as training.


TODO:

script that checks for false positives and false negatives.


