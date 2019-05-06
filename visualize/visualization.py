"""
@file: model.py
@time: 2018/6/17 15:03
@desc:

"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve


def compute_er(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=True)

    sum_sensitivity_specificity_train = tpr + (1 - fpr)
    best_threshold_id = np.argmax(sum_sensitivity_specificity_train)
    best_threshold = thresholds[best_threshold_id]

    y = y_prob > best_threshold

    cm_test = confusion_matrix(y_true, y)
    acc_test = accuracy_score(y_true, y)
    auc_test = roc_auc_score(y_true, y)

    print('Test Accuracy: %s ' % acc_test)
    print('Test AUC: %s ' % auc_test)
    print('Test Confusion Matrix:')
    print(cm_test)

    tpr_score = float(cm_test[1][1]) / (cm_test[1][1] + cm_test[1][0])
    fpr_score = float(cm_test[0][1]) / (cm_test[0][0] + cm_test[0][1])

    return fpr, tpr


def read_y_prob(filename):
    TwoChannel2logit = np.loadtxt(filename)
    siamese = np.split(TwoChannel2logit, 2, axis=1)
    y_true = siamese[1]
    y_prob = siamese[0]
    return y_true, y_prob


def visualize_roc():
    y_true_2logit, y_prob_2logit = read_y_prob('distribution_2Channel2logit_CEDAR.txt')
    y_true_1logit, y_prob_1logit = read_y_prob('distribution_2ChannelsCNN_CEDAR.txt')
    y_true_siamese, y_prob_siamese = read_y_prob('distribution_siamese_CEDAR.txt')
    fpr_siamese, tpr_siamese = compute_er(y_true_siamese, y_prob_siamese)
    fpr_1logit, tpr_1logit = compute_er(y_true_1logit, y_prob_1logit)
    fpr_2logit, tpr_2logit = compute_er(y_true_2logit, y_prob_2logit)

    fig = plt.figure(figsize=(5, 5))
    ax2 = fig.add_subplot(111)
    curve1 = ax2.plot(fpr_siamese, tpr_siamese)
    curve2 = ax2.plot(fpr_1logit, tpr_1logit)
    curve3 = ax2.plot(fpr_2logit, tpr_2logit)
    curve4 = ax2.plot([0, 1], [0, 1], color='navy', linestyle='--')
    # dot = ax2.plot(fpr_score, tpr_score, marker='o', color='black')
    # ax2.text(fpr_score, tpr_score, s='(%.3f,%.3f)' % (fpr_score, tpr_score))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('ROC curve (Test), AUC = %.4f' % auc_test)
    params = {'legend.fontsize': 15,
              'legend.handlelength': 2}
    plt.rcParams.update(params)
    plt.legend(['Siamese', '2ChannelCNN', '2Channel2logit'])
    plt.savefig('ROC_CEDAR_with_backgroud', dpi=500)
    plt.show()


visualize_roc()


def get_auc():
    writer_val = tf.summary.FileWriter('C:\work\Projects\HWS_ID\\test\\2Channels\\val')
    writer_train = tf.summary.FileWriter('C:\work\Projects\HWS_ID\\test\\2Channels\\train')
    auc_var = tf.Variable(0.0)
    tf.summary.scalar("auc", auc_var)
    write_op = tf.summary.merge_all()
    session = tf.InteractiveSession()
    session.run(tf.global_variables_initializer())

    for e in tf.train.summary_iterator(
            "C:\work\Projects\HWS_ID\\test\\2Channels\\2channelscnn.Deep-Ubantu"):
        for v in e.summary.value:
            if 'auc' in v.tag:
                summary = session.run(write_op, {auc_var: v.simple_value})
                writer_train.add_summary(summary, e.step)
        writer_train.flush()

    for e in tf.train.summary_iterator(
            "C:\work\Projects\HWS_ID\\test\\2Channels\\2channelsoftmax.Deep-Ubantu"):
        for v in e.summary.value:
            if 'auc' in v.tag:
                summary = session.run(write_op, {auc_var: v.simple_value})
                writer_val.add_summary(summary, e.step)
        writer_val.flush()
