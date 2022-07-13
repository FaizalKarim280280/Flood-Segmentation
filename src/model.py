from src.imports import *

class Model:
    def __init__(self, backbone, input_shape, output_classes):
        self.INPUT_SHAPE = input_shape
        self.OUTPUT_CLASSES = output_classes
        self.BACKBONE = backbone

    def unet(self, lr = 1e-3):
        model = sm.Unet(self.BACKBONE, input_shape = self.INPUT_SHAPE + (3,),
                        classes = self.OUTPUT_CLASSES, activation = 'sigmoid', encoder_weights = 'imagenet')
        model.compile(
            optimizer = keras.optimizers.Adam(learning_rate = lr),
            loss = keras.losses.BinarCrossEntropy(),
            metrics = [sm.metrics.IOUScore]
        )

        return model

    @staticmethod
    def lr_scheduler(epoch, lr):
        factor, step = 0.3, 5
        if epoch % step == 0 and epoch != 0:
            print("lr changed from {} to {}".format(lr, lr*factor))
            return lr * factor
        else:
            return lr

