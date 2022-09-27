from src.model import  Model
from src.dataset import DataManager
from src.imports import *

# plt.style.use('seaborn')

INPUT_SHAPE = (224, 224)
BACKBONE = efficientnetb2
CLASSES = 2
BATCH_SIZE = 16
UNITS = 64
DROPOUT = 0.2
LEARNING_RATE = 5e-4

def train_model():
    model = Model(BACKBONE, INPUT_SHAPE, CLASSES)
    data = DataManager(INPUT_SHAPE)

    unet = model.unet(UNITS, DROPOUT, LEARNING_RATE)
    callback = keras.callbacks.LearningRateSchedular(model.lr_scheduler)
    train_dataset = data.create_dataset(BATCH_SIZE, BUFFER_SIZE=1000)

    unet.fit(
        train_dataset,
        callback = [callback],
        epochs = 15
    )


def main():
    pass


if __name__ == "__main__":
    main()
