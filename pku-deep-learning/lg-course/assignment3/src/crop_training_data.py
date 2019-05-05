from keras_preprocessing.image import ImageDataGenerator
import pandas as pd
from keras.utils import to_categorical
df = pd.read_csv('../data/train_meta_info.csv')
train_datagen = ImageDataGenerator(featurewise_center=False,validation_split=0.2)
train_generator=train_datagen.flow_from_dataframe(
                        dataframe=df,
                        directory="../data/train",
                        x_col="filenames",
                        y_col=["digit1","digit2","digit3","digit4","digit5"],
                        batch_size=32,
                        seed=42,
                        shuffle=True,
                        class_mode='other',
                        target_size=(64,64),
                        subset='training')
validation_generator = train_datagen.flow_from_dataframe(
    dataframe=df,
    directory="../data/train",
    x_col="filenames",
    y_col=["digit1", "digit2", "digit3", "digit4", "digit5"],
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode='other',
    target_size=(64, 64),
    subset='validation')
def generator_wrapper(generator):
    for batch_x,batch_y in generator:
        batch_y = to_categorical(batch_y)
        yield (batch_x,[batch_y[:,i,:] for i in range(5)])


# g = generator_wrapper(train_generator)


