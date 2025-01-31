# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""
CODE DERIVED FROM : 
Source -> https://github.com/keras-team/keras/blob/v2.13.1/keras/applications/resnet.py
"""

"""ResNet models for Keras.

Reference:
  - [Deep Residual Learning for Image Recognition](
      https://arxiv.org/abs/1512.03385) (CVPR 2015)
"""

from keras import backend
from keras.applications import imagenet_utils
from keras.engine import training
from keras.layers import VersionAwareLayers
from keras.utils import data_utils
from keras.utils import layer_utils
import tensorflow as tf
from tensorflow.keras import regularizers

layers = None
kern_init= "he_normal"
other_init= "he_normal"

def ResNet(
    stack_fn,
    preact,
    use_bias,
    model_name="resnet",
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    init = "he_normal",
    init3x3 = "he_normal",
    **kwargs,
):
    """Instantiates the ResNet, ResNetV2, and ResNeXt architecture.

    Args:
      stack_fn: a function that returns output tensor for the
        stacked residual blocks.
      preact: whether to use pre-activation or not
        (True for ResNetV2, False for ResNet and ResNeXt).
      use_bias: whether to use biases for convolutional layers or not
        (True for ResNet and ResNetV2, False for ResNeXt).
      model_name: string, model name.
      include_top: whether to include the fully-connected
        layer at the top of the network.
      weights: one of `None` (random initialization),
        'imagenet' (pre-training on ImageNet),
        or the path to the weights file to be loaded.
      input_tensor: optional Keras tensor
        (i.e. output of `layers.Input()`)
        to use as image input for the model.
      input_shape: optional shape tuple, only to be specified
        if `include_top` is False (otherwise the input shape
        has to be `(224, 224, 3)` (with `channels_last` data format)
        or `(3, 224, 224)` (with `channels_first` data format).
        It should have exactly 3 inputs channels.
      pooling: optional pooling mode for feature extraction
        when `include_top` is `False`.
        - `None` means that the output of the model will be
            the 4D tensor output of the
            last convolutional layer.
        - `avg` means that global average pooling
            will be applied to the output of the
            last convolutional layer, and thus
            the output of the model will be a 2D tensor.
        - `max` means that global max pooling will
            be applied.
      classes: optional number of classes to classify images
        into, only to be specified if `include_top` is True, and
        if no `weights` argument is specified.
      classifier_activation: A `str` or callable. The activation function to use
        on the "top" layer. Ignored unless `include_top=True`. Set
        `classifier_activation=None` to return the logits of the "top" layer.
        When loading pretrained weights, `classifier_activation` can only
        be `None` or `"softmax"`.
      **kwargs: For backwards compatibility only.

    Returns:
      A `keras.Model` instance.
    """

    global kern_init 
    global other_init 
    global layers

    kern_init = init3x3
    other_init = init 
    if "layers" in kwargs:
        layers = kwargs.pop("layers")
    else:
        layers = VersionAwareLayers()


    # Determine proper input shape
    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=224,
        min_size=32,
        data_format=backend.image_data_format(),
        require_flatten=include_top,
        weights=weights,
    )

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1
    print("DATA FORMAT : ", backend.image_data_format(), bn_axis)
    '''if preprocess_input:
        img_input = layers.ZeroPadding2D(padding=4)(img_input)
        img_input = layers.RandomCrop(input_shape[1], input_shape[2])(img_input, training = True)'''

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name="conv1_pad")(
        img_input
    )
    x = layers.Conv2D(64, 7, strides=2, use_bias=use_bias, name="conv1_conv7x7", kernel_initializer=other_init, kernel_regularizer = regularizers.l2(0.00005))(x)

    if not preact:
        x = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.0e-5, momentum=0.9, name="conv1_bn"
        )(x)
        x = layers.Activation("relu", name="conv1_relu")(x)
    
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name="pool1_pad")(x)
    x = layers.MaxPooling2D(3, strides=2, name="pool1_pool")(x)

    x = stack_fn(x)

    if preact:
        x = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.0e-5, momentum=0.9, name="post_bn"
        )(x)
        x = layers.Activation("relu", name="post_relu")(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        imagenet_utils.validate_activation(classifier_activation, weights)
        x = layers.Dense(
            classes, activation=classifier_activation, name="predictions", kernel_initializer="he_normal", kernel_regularizer = regularizers.l2(0.00005)
        )(x)
    else:
        if pooling == "avg":
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        elif pooling == "max":
            x = layers.GlobalMaxPooling2D(name="max_pool")(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = training.Model(inputs, x, name=model_name)

    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model


def block1(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    """A residual block.

    Args:
      x: input tensor.
      filters: integer, filters of the bottleneck layer.
      kernel_size: default 3, kernel size of the bottleneck layer.
      stride: default 1, stride of the first layer.
      conv_shortcut: default True, use convolution shortcut if True,
          otherwise identity shortcut.
      name: string, block label.

    Returns:
      Output tensor for the residual block.
    """
    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1
    print("BN axis :", bn_axis)

    if conv_shortcut:
        shortcut = layers.Conv2D(
            4 * filters, 1, strides=stride, name=name + "_0_conv1x1", kernel_initializer=other_init, kernel_regularizer = regularizers.l2(0.00005)
        )(x)
        shortcut = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.0e-5, momentum=0.9, name=name + "_0_bn"
        )(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, 1, strides=stride, name=name + "_1_conv1x1", kernel_initializer=other_init, kernel_regularizer = regularizers.l2(0.00005))(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.0e-5, momentum=0.9, name=name + "_1_bn"
    )(x)
    x = layers.Activation("relu", name=name + "_1_relu")(x)

    x = layers.Conv2D(
        filters, kernel_size, padding="SAME", name=name + "_2_conv3x3", kernel_initializer = kern_init, kernel_regularizer = regularizers.l2(0.00005)
    )(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.0e-5, momentum=0.9, name=name + "_2_bn"
    )(x)
    x = layers.Activation("relu", name=name + "_2_relu")(x)

    x = layers.Conv2D(4 * filters, 1, name=name + "_3_conv1x1", kernel_initializer=other_init, kernel_regularizer = regularizers.l2(0.00005))(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.0e-5, momentum=0.9, name=name + "_3_bn"
    )(x)

    x = layers.Add(name=name + "_add")([shortcut, x])
    x = layers.Activation("relu", name=name + "_out")(x)
    return x


def stack1(x, filters, blocks, stride1=2, name=None):
    """A set of stacked residual blocks.

    Args:
      x: input tensor.
      filters: integer, filters of the bottleneck layer in a block.
      blocks: integer, blocks in the stacked blocks.
      stride1: default 2, stride of the first layer in the first block.
      name: string, stack label.

    Returns:
      Output tensor for the stacked blocks.
    """
    x = block1(x, filters, stride=stride1, name=name + "_block1")
    for i in range(2, blocks + 1):
        x = block1(
            x, filters, conv_shortcut=False, name=name + "_block" + str(i)
        )
    return x




def ResNet18(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    init = "he_normal",
    init3x3 = "he_normal",
    **kwargs,
):
    """Instantiates the ResNet50 architecture."""

    def stack_fn(x):
        x = stack1(x, 64, 2, stride1=1, name="conv2")
        x = stack1(x, 128, 2, name="conv3")
        x = stack1(x, 256, 2, name="conv4")
        return stack1(x, 512, 2, name="conv5")

    return ResNet(
        stack_fn,
        False,
        True,
        "resnet18",
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        classes,
        init = init,
        init3x3= init3x3,
        **kwargs,
    )



def ResNet50(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    init = "he_normal",
    init3x3 = "he_normal",
    **kwargs,
):
    """Instantiates the ResNet50 architecture."""

    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, name="conv2")
        x = stack1(x, 128, 4, name="conv3")
        x = stack1(x, 256, 6, name="conv4")
        return stack1(x, 512, 3, name="conv5")

    return ResNet(
        stack_fn,
        False,
        True,
        "resnet50",
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        classes,
        init = init,
        init3x3= init3x3,
        **kwargs,
    )



def ResNet101(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    init = "he_normal",
    init3x3 = "he_normal",
    **kwargs,
):
    """Instantiates the ResNet101 architecture."""

    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, name="conv2")
        x = stack1(x, 128, 4, name="conv3")
        x = stack1(x, 256, 23, name="conv4")
        return stack1(x, 512, 3, name="conv5")

    return ResNet(
        stack_fn,
        False,
        True,
        "resnet101",
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        classes,
        init = init,
        init3x3= init3x3,
        **kwargs,
    )



def ResNet152(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    init = "he_normal",
    init3x3 = "he_normal",
    **kwargs,
):

    """Instantiates the ResNet152 architecture."""

    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, name="conv2")
        x = stack1(x, 128, 8, name="conv3")
        x = stack1(x, 256, 36, name="conv4")
        return stack1(x, 512, 3, name="conv5")

    return ResNet(
        stack_fn,
        False,
        True,
        "resnet152",
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        classes,
        init = init,
        init3x3= init3x3,
        **kwargs,
    )


def decode_predictions(preds, top=5):
    return imagenet_utils.decode_predictions(preds, top=top)

