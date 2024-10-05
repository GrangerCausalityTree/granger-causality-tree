from sklearn.metrics import confusion_matrix
from codes.utils import *

def calculate_purity(y_true, y_pred):
    # Compute the confusion matrix
    contingency_matrix = confusion_matrix(y_true, y_pred)
    
    # Find the maximum values along the columns and sum them
    max_contingency_sum = np.sum(np.amax(contingency_matrix, axis=0))
    
    # Divide by the number of samples to get the purity score
    purity = max_contingency_sum / np.sum(contingency_matrix)
    
    return purity

def par_set_accuracy(true_set, pred_set, true_labels, pred_labels):
    return np.sum([true_set[true_labels[i]] == pred_set[pred_labels[i]] for i in range(len(true_labels))]) / len(true_labels)

if __name__ == "__main__":
    pass