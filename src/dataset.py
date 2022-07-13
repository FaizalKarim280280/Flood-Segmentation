
from src.imports import *
from src.augmentation import Augmentation

class DataManager:
    def __init__(self, img_shape, data_path, batch_size):
        self.IMG_SHAPE = img_shape
        self.PATH = data_path
        self.BATCH_SIZE = batch_size
        self.BUFFER_SIZE = 100

    def build_df(self):
        df = pd.read_csv(self.PATH)

        for i in range(len(df)):
            df['Image'][i] = self.PATH + 'Image/' + df['Image'][i]
            df['Mask'][i] = self.PATH + 'Mask/' + df['Mask'][i]

        return df

    @staticmethod
    def modify_mask(mask, threshold = 0.5):
        mask = np.expand_dims(mask, axis = 2)
        t_mask = np.zeros(mask.shape)
        np.place(t_mask[:, :, 0], mask[:, :, 0] >= threshold, 1)
        return t_mask

    def map_function(self, img, mask, training):
        img, mask = plt.imread(img.decode())[:, :, :3], plt.imread(mask.decode())
        img = op.resize(img, self.IMG_SHAPE)
        mask = self.modify_mask(op.resize(mask, self.IMG_SHAPE))

        img = img/255.0
        if training:
            transformed = transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

        return img.astype(np.float64), mask.astype(np.float64)

    def create_dataset(self, training = True):
        data = self.build_df()
        dataset = tf.data.Dataset.from_tensor_slices((data['Image'], data['Mask']))
        dataset = dataset.shuffle(self.BUFFER_SIZE)
        dataset = dataset.map(lambda img, mask : tf.numpy_function(
            self.map_function, [img, mask, training], [tf.float64, tf.float64]),
                              num_parallel_calls = tf.data.experimental.AUTOTUNE).batch(self.BATCH_SIZE)

        dataset = dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
        return dataset

