from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import multilabel_confusion_matrix, classification_report
import numpy as np

# Define your test data path
test_path = "C:\\Users\\Syed Farjad\\Desktop\\FYP\\DATASET\\TESTING"
img_height = 224
img_width = 224
batch_size = 32

# Set up the test data generator
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_path, 
    target_size=(img_height, img_width),
    batch_size=batch_size, 
    class_mode='categorical'
)

# Load the models
model_cnn = load_model("C:\\Users\\Syed Farjad\\anaconda3\\envs\\myenv\\23_layers_cnn_model.h5")
model_vgg16 = load_model("C:\\Users\\Syed Farjad\\anaconda3\\envs\\myenv\\fine_tuned_vgg16_model.h5")


# Predictions for confusion matrix
y_pred_cnn = model_cnn.predict(test_generator)
y_pred_vgg16 = model_vgg16.predict(test_generator)

# Convert predictions to class labels based on thresholds
y_pred_cnn_classes = np.argmax(y_pred_cnn, axis=1)
y_pred_vgg16_classes = np.argmax(y_pred_vgg16, axis=1)

# Convert y_true to integer labels
y_true = test_generator.classes

# Calculate confusion matrix for both models
conf_matrix_cnn = confusion_matrix(y_true, y_pred_cnn_classes)
conf_matrix_vgg16 = confusion_matrix(y_true, y_pred_vgg16_classes)

# Print confusion matrix for both models
print("\nConfusion Matrix for 23-Layers CNN Model:")
print(conf_matrix_cnn)

print("\nConfusion Matrix for Fine-tuned VGG16 Model:")
print(conf_matrix_vgg16)

# Function to calculate and print classification report for both models
def print_classification_report(y_true, y_pred, model_name):
    report = classification_report(y_true, y_pred, target_names=[str(i) for i in range(4)], zero_division=1)
    print(f"\nClassification Report for {model_name}:\n{report}")

# Print classification report for both models
print_classification_report(y_true, y_pred_cnn_classes, "23-Layers CNN Model")
print_classification_report(y_true, y_pred_vgg16_classes, "Fine-tuned VGG16 Model")