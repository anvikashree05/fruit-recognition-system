import tensorflow as tf
from tensorflow.keras import layers, models

# Load Training Data
train_dataset = tf.keras.utils.image_dataset_from_directory(
    "datasmall/train",
    image_size=(100, 100),
    batch_size=32
)

# Load Test Data
test_dataset = tf.keras.utils.image_dataset_from_directory(
    "datasmall/test",
    image_size=(100, 100),
    batch_size=32
)

class_names = train_dataset.class_names

print("Classes Found:")
print(class_names)

# CNN Model
model = models.Sequential([

    layers.Rescaling(1./255, input_shape=(100,100,3)),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),

    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=5
)

model.save("fruit_classifier_model.keras")

print("Model Saved Successfully!")