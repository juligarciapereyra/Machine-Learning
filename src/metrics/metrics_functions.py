from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def display_confusion_matrix(y_true, y_pred, emotion_labels, normalize=False):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=emotion_labels)
    disp.plot(cmap='Reds')
    plt.title('Normalized Confusion Matrix' if normalize else 'Confusion Matrix')
    plt.xticks(rotation=45)
    plt.show()
    return cm

def get_accuracy(cm):
    accuracies = []
    total_correct = 0
    total_instances = np.sum(cm)
    
    for i in range(cm.shape[0]):
        TP = cm[i, i]
        accuracies.append('-')
        total_correct += TP
    
    total_accuracy = total_correct / total_instances
    
    return np.array(accuracies), total_accuracy


def get_precision(cm):
    precisions = np.zeros(cm.shape[0])
    for i in range(cm.shape[0]):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        if TP + FP == 0:
            precision = 0
        else:
            precision = TP / (TP + FP)
        precisions[i] = precision
    precision_macro = np.mean(precisions)
    return precisions, precision_macro


def get_recall(cm):
    recalls = np.zeros(cm.shape[0])
    for i in range(cm.shape[0]):
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        if TP + FN == 0:
            recall = 0
        else:
            recall = TP / (TP + FN)
        recalls[i] = recall
    recall_macro = np.mean(recalls)
    return recalls, recall_macro


def get_fpr(cm):
    fpr = np.zeros(cm.shape[0])
    for i in range(cm.shape[0]):
        FP = np.sum(cm[:, i]) - cm[i, i]
        TN = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        if FP + TN == 0:
            fpr[i] = 0
        else:
            fpr[i] = FP / (FP + TN)
    fpr_macro = np.mean(fpr)
    return fpr, fpr_macro


def get_f1_score(precisions, recalls): # receives the list of precisions and recalls (per emotion, not macro)
    f1_scores = np.zeros(precisions.shape[0])
    for i in range(precisions.shape[0]):
        if precisions[i] + recalls[i] == 0:
            f1 = 0
        else:
            f1 = 2 * (precisions[i] * recalls[i]) / (precisions[i] + recalls[i])
        f1_scores[i] = f1
    f1_macro = np.mean(f1_scores)
    return f1_scores, f1_macro


def get_scores(Y_true, Y_pred, emotion_labels):
    y_true = Y_true
    y_pred = Y_pred

    metrics = {}
    metrics['cm'] = display_confusion_matrix(y_true, y_pred, emotion_labels)
    metrics['cm_normalized'] = display_confusion_matrix(y_true, y_pred, emotion_labels, normalize=True)
    metrics['accuracy'], metrics['accuracy_macro'] = get_accuracy( metrics['cm'])
    metrics['precisions'], metrics['precision_macro'] = get_precision( metrics['cm'])
    metrics['recalls'], metrics['recall_macro'] = get_recall( metrics['cm']) 
    metrics['fpr'], metrics['fpr_macro'] = get_fpr( metrics['cm'])
    metrics['f1_scores'], metrics['f1_macro'] = get_f1_score(metrics['precisions'], metrics['recalls'])
    return metrics


def display_metrics_as_table(metrics, emotion_labels):

    df_metrics = pd.DataFrame({
        'Accuracy': metrics['accuracy'],
        'Precision': metrics['precisions'],
        'Recall': metrics['recalls'],
        'FPR': metrics['fpr'],
        'F1-Score': metrics['f1_scores']
    }, index=emotion_labels[:len(metrics['accuracy'])])

    macro_avg = {
        'Accuracy': metrics['accuracy_macro'],
        'Precision': metrics['precision_macro'],
        'Recall': metrics['recall_macro'],
        'FPR': metrics['fpr_macro'],
        'F1-Score': metrics['f1_macro']
    }
    df_macro_avg = pd.DataFrame(macro_avg, index=['Macro Avg'])
    df_metrics = pd.concat([df_metrics, df_macro_avg])

    fig = go.Figure(data=[go.Table(
        header=dict(values=[""] + list(df_metrics.columns),
                    fill_color='paleturquoise',
                    align='center'),
        cells=dict(values=[df_metrics.index] + [df_metrics[col] for col in df_metrics.columns],
                   fill_color='lavender',
                   align='center'))
    ])

    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), autosize=False, width=1000, height=400)
    fig.show()