import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

# Provided confusion matrices
conf_matrix_23layers = np.array([[62, 44, 102, 92],
                                 [59, 51, 104, 92],
                                 [112, 53, 121, 119],
                                 [ 62, 42, 94, 102]])

conf_matrix_vgg16 = np.array([[42, 61, 94, 103],
                              [46, 73, 96, 91],
                              [54, 103, 128, 120],
                              [39, 68, 108, 85]])

class_names = ['GLIOMA', 'MENINGIOMA', 'NO-TUMOR', 'PITUITARY']

def plot_confusion_matrix(conf_matrix, classes, model_name):
    plt.figure(figsize=(8, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# Plotting confusion matrix for the 23-Layers CNN Model
plot_confusion_matrix(conf_matrix_23layers, class_names, '23-Layers CNN Model')

# Plotting confusion matrix for the Fine-tuned VGG16 Model
plot_confusion_matrix(conf_matrix_vgg16, class_names, 'Fine-tuned VGG16 Model')
