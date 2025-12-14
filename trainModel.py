import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, GlobalAveragePooling2D

# annotation files and training images
train_annotations_path = 'anno_train.csv'
train_dir = 'train'
train_annotations = pd.read_csv(train_annotations_path)

# to filter out rows with no corresponding images in case of corruption
train_annotations = train_annotations[train_annotations['image_name'].apply(lambda x: os.path.exists(os.path.join(train_dir, x)))]
# image and batch size
image_size = (150, 150)
batch_size = 10

def load_and_preprocess_image(image_path, target_size):
    """
    Helper function to load images and preprocess them
    """
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    return img_array

def create_data_generator(annotations, base_dir, target_size, batch_size, class_indices):
    """
    Helper function to create data generator
    """
    while True:
        for start in range(0, len(annotations), batch_size):
            end = min(start + batch_size, len(annotations))
            batch_annotations = annotations[start:end]
            images, labels, bboxes = [], [], []
            for _, row in batch_annotations.iterrows():
                image_path = os.path.join(base_dir, row['image_name'])
                if os.path.exists(image_path):
                    img_array = load_and_preprocess_image(image_path, target_size)
                    images.append(img_array)
                    labels.append(row['label'])
                    bboxes.append([row['x_min'], row['y_min'], row['x_max'], row['y_max']])
            images = np.array(images)
            labels = np.array([class_indices[label] for label in labels])
            labels = to_categorical(labels, num_classes=len(class_indices))
            bboxes = np.array(bboxes)
            yield images, labels, bboxes

# class indices based on annotations
class_indices = {label: i for i, label in enumerate(train_annotations['label'].unique())}

# training data generator
train_generator = create_data_generator(train_annotations, train_dir, image_size, batch_size, class_indices)
steps_per_epoch_train = max(1, len(train_annotations) // batch_size)      # steps per epoch

# Define the custom Fast R-CNN-like model
input_layer = Input(shape=(150, 150, 3))

# layers
x = Conv2D(32, (3, 3), activation='relu')(input_layer)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(256, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(512, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
# simplified RoI pooling
roi_pool = GlobalAveragePooling2D()(x)

# classification network
dense1 = Dense(512, activation='relu')(roi_pool)
dropout = Dropout(0.5)(dense1)
classifier_output = Dense(len(class_indices), activation='softmax')(dropout)
bbox_regressor_output = Dense(4)(dropout)
model = Model(inputs=input_layer, outputs=[classifier_output, bbox_regressor_output])


# w\ two loss functions
model.compile(optimizer=Adam(), loss={'dense_2': 'categorical_crossentropy', 'dense_3': 'mean_squared_error'}, metrics={'dense_2': 'accuracy'})
# early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


# failsafe to always save the model no matter as long as the program runs
try:
    # train
    history = model.fit(train_generator, steps_per_epoch=steps_per_epoch_train, epochs=60, callbacks=[early_stopping])
finally:
    # save model as JSON
    model_json = model.to_json()
    with open("car_classifier_model.json", "w") as json_file:
        json_file.write(model_json)

    # save the weights
    model.save_weights("car_classifier_model_weights.h5")
    print("process is completed")

