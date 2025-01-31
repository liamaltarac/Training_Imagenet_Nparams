from models.resnet_keras import ResNet18, ResNet50, ResNet101, ResNet152
from models.vgg16_keras import VGG16
from models.vgg19_keras import VGG19

def getModel(model_name,     
             include_top=True,
             weights="imagenet",
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             classifier_activation="softmax",
             init = "he_normal",
             init3x3 = "he_normal"):

    model_name = model_name.lower()

    if model_name == "resnet18":
        model = ResNet18(
                    include_top=include_top,
                    weights=weights,
                    input_tensor=input_tensor,
                    input_shape=input_shape,
                    classes=classes,
                    init = init,
                    init3x3 = init3x3
                    )

    if model_name == "resnet50":
        model = ResNet50(
                    include_top=include_top,
                    weights=weights,
                    input_tensor=input_tensor,
                    input_shape=input_shape,
                    classes=classes,
                    init = init,
                    init3x3 = init3x3
                    )

    if model_name == "resnet101":
        model = ResNet101(
                    include_top=include_top,
                    weights=weights,
                    input_tensor=input_tensor,
                    input_shape=input_shape,
                    classes=classes,
                    init = init,
                    init3x3 = init3x3
                    )

    if model_name == "vgg16":
        model = VGG16(
                    include_top=include_top,
                    weights=weights,
                    input_tensor=input_tensor,
                    input_shape=input_shape,
                    classes=classes,
                    init = init,
                    init3x3 = init3x3
                    )
    if model_name == "vgg19":
        model = VGG19(
                    include_top=include_top,
                    weights=weights,
                    input_tensor=input_tensor,
                    input_shape=input_shape,
                    classes=classes,
                    init = init,
                    init3x3 = init3x3
                    )
    return model

