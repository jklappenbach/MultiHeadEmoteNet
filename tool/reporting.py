from h5py._hl import dataset
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import seaborn as sns

# -----------------------------
# Confusion Matrix
def plot_confusion_matrix(labels, preds, class_names):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# -----------------------------
# Per-class Accuracy
def per_class_accuracy(labels, preds, class_names):
    cm = confusion_matrix(labels, preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    plt.figure(figsize=(10,5))
    sns.barplot(x=class_names, y=per_class_acc)
    plt.ylabel('Accuracy')
    plt.ylim(0,1)
    plt.title('Per-Class Accuracy')
    plt.show()
    return per_class_acc

# -----------------------------
# Classification Report
def print_classification_report(labels, preds, class_names):
    report = classification_report(labels, preds, target_names=class_names)
    print("Classification Report:\n")
    print(report)

# -----------------------------
# Plot
plot_confusion_matrix(labels, preds, dataset.classes)
acc_per_class = per_class_accuracy(labels, preds, dataset.classes)
print_classification_report(labels, preds, dataset.classes)
