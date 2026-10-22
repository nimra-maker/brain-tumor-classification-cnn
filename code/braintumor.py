import matplotlib
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# PATHS FOR DATASET
train_path = "C:\\Users\\Syed Farjad\\Desktop\\FYP\\DATASET\\TRAINING"
test_path ="C:\\Users\\Syed Farjad\\Desktop\\FYP\\DATASET\\TESTING"

# IMG DIMENSIONS
img_height = 224
img_width = 224
batch_size = 32    
epochs = 40 
    
# Creating data generators 
train_datagen = ImageDataGenerator(
    rescale=1./255,    
    shear_range=0.2, 
    zoom_range=0.2, 
    rotation_range = 90,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path, target_size=(img_height, img_width),
    batch_size=batch_size, class_mode='categorical' 
)
  
test_generator = test_datagen.flow_from_directory(
    test_path, target_size=(img_height, img_width),
    batch_size=batch_size, class_mode='categorical'
)
              
# 23-Layers CNN model
model_cnn = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),  
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    Conv2D(filters=256, kernel_size=(3, 3), activation='relu'), 
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=512,kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(), 
    Dense(units=4096, activation='relu'),
    Dropout(0.5),
    Dense(units=4, activation='softmax')  # Output layer with 4 units for multi-class classification
])

# Compiling the    model
optimizer = Adam(learning_rate=0.001)
model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history_cnn = model_cnn.fit(train_generator, epochs=epochs, validation_data=test_generator)

# Visualization for 23-Layers CNN Model
plt.plot(history_cnn.history['accuracy'])
plt.plot(history_cnn.history['val_accuracy'])
plt.title('Model Accuracy - 23-Layers CNN')
plt.xlabel('Epoch') 
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
#input("Press enter to continue")


# Fine-tuned VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freezing conv layers
for layer in base_model.layers:
    layer.trainable = False

# Creating a Sequential model
model_vgg16 = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(4, activation='softmax')  # Output layer with 4 units for multi-class classification
])

# Model compilation
model_vgg16.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


# Training
history_vgg16  = model_vgg16.fit(train_generator, epochs=epochs, validation_data=test_generator)

# Visualization for Fine-tuned VGG16 Model
plt.plot(history_vgg16.history['accuracy'])
plt.plot(history_vgg16.history['val_accuracy'])
plt.title('Model Accuracy - Fine-tuned VGG16')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
#input("press enter to continue")

#saving the models
model_cnn.save("23_layers_cnn_model.h5")
model_vgg16.save("fine_tuned_vgg16_model.h5")

# Evaluate both models on the test set
results_cnn = model_cnn.evaluate(test_generator)
results_vgg16 = model_vgg16.evaluate(test_generator)

# Print the results
print("23-Layers CNN Model Results:")
print("Loss:", results_cnn[0])
print("Accuracy:", results_cnn[1]) 

print("\nFine-tuned VGG16 Model Results:")
print("Loss:", results_vgg16[0])
print("Accuracy:", results_vgg16[1])

# Predictions for confusion matrix
y_pred_cnn = model_cnn.predict(test_generator) 
y_pred_vgg16 = model_vgg16.predict(test_generator)

# Convert predictions to class labels
y_true = test_generator.classes
y_pred_cnn_classes = y_pred_cnn.argmax(axis=1)
y_pred_vgg16_classes = y_pred_vgg16.argmax(axis=1)

# Print confusion matrix and classification report for both models
print("\nConfusion Matrix and Classification Report for 23-Layers CNN Model:")
print(confusion_matrix(y_true, y_pred_cnn_classes))
print(classification_report(y_true, y_pred_cnn_classes))
  
print("\nConfusion Matrix and Classification Report for Fine-tuned VGG16 Model:")
print(confusion_matrix(y_true, y_pred_vgg16_classes))
print(classification_report(y_true, y_pred_vgg16_classes))

