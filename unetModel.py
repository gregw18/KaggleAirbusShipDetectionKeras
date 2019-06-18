# Create a unet, based on the standard keras resnet50 with pretrained weights.
# May 24, 2019.

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Conv2D, UpSampling2D, Input, concatenate
import numpy as np

lastResNetLayer = 0


def getFeedForwardLayers(model, startIdx, endIdx):
    # Return list of layers where inputs and outputs are different sizes,
    # so can feed those inputs to the decoding part of the network.

    # Start with actual input - first layer.
    feedForwardTensors = []
    feedForwardTensors.append(model.get_layer(index=0))

    # Then add every layer where input size is not equal to output size, with specific
    # exceptions listed below.
    for layer in model.layers[startIdx:endIdx+1]:
      if not (layer.output_shape[1] == layer.input_shape[1]):
        if not (layer.name.startswith("add_")):
            if not (layer.name.endswith("branch2a")):
                if not (layer.name.startswith("max_pooling")):
                    feedForwardTensors.append(layer)
    
    return feedForwardTensors


def addUpConvModule(model, prevActivationsLayer, feedForwardActivationsLayer):
    # Add an upsizing convolution layer to the given model, whose input is the
    # output from the given previous layer and the input from the given
    # encoder layer.

    prevLayerOutput = prevActivationsLayer.output
    feedForwardOutput = feedForwardActivationsLayer.input

    # Am doubling the image size and halving the number of filters, compared to the
    # input layer.
    numFilters = round(prevActivationsLayer.output_shape[3] / 2)
    up = UpSampling2D(size=(2, 2))(prevLayerOutput)
    concat = concatenate([up, feedForwardOutput])
    conva = Conv2D(numFilters, (3, 3), activation='relu', padding='same')(concat)
    convb = Conv2D(numFilters, (3, 3), activation='relu', padding='same')(conva)
    
    return Model(inputs=model.input, outputs=convb)


def freezeResnet(model):
    # Freeze the resnet portion of the model.
    for layer in model.layers[0:lastResNetLayer]:
      layer.trainable = False


def unfreezeResnet(model):
    # Unfreeze the resnet portion of the model.
    for layer in model.layers[0:lastResNetLayer]:
      layer.trainable = True


def getUnet(pretrained_weights = None):
    # Create the unet, starting with the standard resnet 50, then manually adding the decoder.
    # Default resnet to frozen.
    if (pretrained_weights):
        base_model = ResNet50(weights=None, include_top=False, input_shape=(768, 768, 3))
    else:
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(768, 768, 3))

    # Remember last index for resnet, so can freeze/unfreeze later.
    global lastResNetLayer
    lastResNetLayer = len(base_model.layers)
    feedForwardList = getFeedForwardLayers(base_model, 3, len(base_model.layers))

    for layer in reversed(feedForwardList):
        base_model = addUpConvModule(base_model, base_model.layers[-1], layer)

    # Add sigmoid at end to convert to a mask.
    convFinal = Conv2D(1, (1, 1), activation='sigmoid')(base_model.layers[-1].output)
    base_model = Model(inputs=base_model.input, outputs=convFinal)

    #base_model.summary()

    if (pretrained_weights):
        base_model.load_weights(pretrained_weights)

    freezeResnet(base_model)

    return base_model


