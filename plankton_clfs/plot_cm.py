import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

y_pred = np.load("./laps_nobg_100/predictions_1.npy")
y_test = np.load("./laps_nobg_100/real_labels_1.npy")

def plot_confusion_matrix(cm,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
     
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange((20))
    plt.xticks(tick_marks, rotation=45)
    plt.yticks(tick_marks)
    """
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    """
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

# Compute barchart data
classes_real, counts_reall = np.unique(y_test, return_counts=True)
classes_pred, counts_pred = np.unique(y_pred, return_counts=True)

print("\nNumber of images in the validation set by class:\n", counts_reall)
print("\nNumber of images predicted in the validation set by class:\n", counts_pred)

mistakes = np.zeros(len(classes_real))
for i in range(len(y_test)):
    if y_test[i] != y_pred[i]:
        classe = y_test[i]   
        mistakes[classe] = mistakes[classe] + 1

percent = np.zeros(len(classes_real), dtype='float')
percent = mistakes/counts_reall

total_images = np.sum(counts_reall)
total_mistakes = np.sum(mistakes)
total_accuracy = (total_images - total_mistakes) / total_images

print("\nAccuracy:\n", total_accuracy)

counts_real = np.copy(counts_reall)
classes_real_sorted = np.zeros(len(classes_real), dtype='uint32')
counts_real_sorted = np.zeros(len(counts_real), dtype='uint32')
counts_pred_sorted = np.zeros(len(counts_pred), dtype='uint32')
mistakes_sorted = np.zeros(len(mistakes), dtype='uint32')
percent_sorted = np.zeros(len(classes_real), dtype='float')
for c in range(len(classes_real)):
    max_idx = np.argmax(counts_real)
    counts_real_sorted[-1-c] = counts_real[max_idx]
    classes_real_sorted[-1-c] = classes_real[max_idx]
    counts_pred_sorted[-1-c] = counts_pred[max_idx]
    mistakes_sorted[-1-c] = mistakes[max_idx]
    percent_sorted[-1-c] = percent[max_idx]
    counts_real[max_idx] = -1

print("\nClasses sorted by number of images:\n", classes_real_sorted)
print("\nSorted number of images in the validation set by class:\n", counts_real_sorted)
print("\nSorted number of images predicted in the validation set by class:\n", counts_pred_sorted)
print("\nSorted number of mistakes in the validation set by class:\n", mistakes_sorted)
print("\nSorted percent of mistakes in the validation set by class:\n", percent_sorted)


# Plot barchart
fig, ax = plt.subplots()

index = np.arange(len(classes_real))
bar_width = 0.35

rects1 = ax.bar(index, counts_real_sorted, bar_width,
               color='b')

ax2 = ax.twinx()

rects2 = ax2.bar(index + bar_width, percent_sorted, bar_width,
                color='r')

ax.set_xlabel('Classes')
ax.set_ylabel('Number of examples', color='b')
ax2.set_ylabel('Percent', color='r')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels((classes_real_sorted))

fig.tight_layout()
plt.show()

print(sklearn.metrics.classification_report(y_test, y_pred))
