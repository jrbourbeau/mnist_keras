
import numpy as np
import matplotlib.pyplot as plt


def plot_digits(X, y_true, y_pred=None, sample_idx=0, ax=None):

    if ax is None:
        ax = plt.gca()

    img = X[sample_idx].reshape(28,28)
    ax.imshow(img, cmap='Greys')
    true_label = np.argmax(y_true[sample_idx])
    if y_pred is not None:
        pred_label = np.argmax(y_pred[sample_idx])
        correct = true_label == pred_label
        color = 'C2' if correct else 'C3'
        ax.set_title('true label: {} \n pred label: {}'.format(true_label, pred_label), color=color)
    else:
        ax.set_title('true label: {}'.format(true_label))
