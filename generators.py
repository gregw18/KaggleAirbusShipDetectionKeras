# Run training on my manual unet. Since it starts with a pretrained resnet50,
# start training by finetuning just the decoder, then unfreeze and try entire network.
# Thus, want functions to train decoder and entire thing - manually freeze/unfreeze
# as appropriate. 
# Believe that my curated data and masks on AWS are not separated by train or validation,
# and haven't split out test, so want to use validation_split rather than creating
# separate directories. Thus, separate function for setting up generators for each.
# Since images are from satellites, should be able to rotate or flip in every way.

from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import os
from keras.callbacks import ModelCheckpoint

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# AWS Settings
base_dir = 'data'

# Local settings
#base_dir = 'E:/code/kaggle/Airbus ShipId/data'

seed = 42


def getGenerators(imgDir, maskDir, imgDataGenerator, maskDataGenerator, imgSize, dataSubSet, batch_size):
    # Generate image and mask generators for given ImageDataGenerator, for requested
    # data subset (training or validation.)
    if len(dataSubSet) > 0:
        img_generator = imgDataGenerator.flow_from_directory(imgDir, 
            target_size=imgSize,
            color_mode='rgb',
            class_mode=None,
            seed=seed,
            subset=dataSubSet,
            batch_size=batch_size
            )

        mask_generator = maskDataGenerator.flow_from_directory(maskDir,
            target_size=imgSize,
            color_mode='grayscale',
            class_mode=None,
            seed=seed,
            subset=dataSubSet,
            batch_size=batch_size
            )
    else:
        img_generator = imgDataGenerator.flow_from_directory(imgDir, 
            target_size=imgSize,
            color_mode='rgb',
            class_mode=None,
            seed=seed,
            batch_size=batch_size
            )

        mask_generator = maskDataGenerator.flow_from_directory(maskDir,
            target_size=imgSize,
            color_mode='grayscale',
            class_mode=None,
            seed=seed,
            batch_size=batch_size
            )

    return img_generator, mask_generator


def getTrainValidGenerators(imgDir, maskDir, validPercent, img_size, batch_size):
    # Return training and validation generators containing images and masks, with
    # default augmentation shown below.
    # validPercent should be provided as value between 0 and 1 - i.e. 20% given as 0.2.

    # Tested, get better results with augmentation.
    data_gen_args = dict(rotation_range=90,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='nearest',
                    validation_split=validPercent
                    )
    #data_gen_args = dict(horizontal_flip=False,
    #                vertical_flip=False,
    #                validation_split=validPercent
    #                )
    img_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    train_img_generator, train_mask_generator = getGenerators(imgDir,
                                                                maskDir,
                                                                img_datagen, 
                                                                mask_datagen, 
                                                                img_size, 
                                                                'training',
                                                                batch_size
                                                                )

    valid_img_generator, valid_mask_generator = getGenerators(imgDir,
                                                                maskDir,
                                                                img_datagen, 
                                                                mask_datagen, 
                                                                img_size, 
                                                                'validation',
                                                                batch_size
                                                                )

    train_generator = zip(train_img_generator, train_mask_generator)
    valid_generator = zip(valid_img_generator, valid_mask_generator)
 
    return train_generator, valid_generator


def getTestGenerator(imgDir, maskDir, img_size, batch_size):
    # Return test generators containing images and masks.

    # Don't want augmentation for testing.
    data_gen_args = dict('')
    img_datagen  = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    test_img_generator, test_mask_generator = getGenerators(imgDir,
                                                                maskDir,
                                                                img_datagen, 
                                                                mask_datagen, 
                                                                img_size, 
                                                                '',
                                                                batch_size
                                                                )

    test_generator = zip(test_img_generator, test_mask_generator)
 
    return test_generator

