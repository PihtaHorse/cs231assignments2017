import numpy as np
from random import shuffle
from past.builtins import xrange


def get_gradient(W, x, y_true, scores):
    dW = np.zeros(W.shape)

    for i, row in enumerate(W.T):
        dw_i = 0

        if i == y_true:
            for j, y_pred in enumerate(scores):
                if j != y_true:
                    dw_i -= (y_pred - scores[i] + 1) > 0
        else:
            dw_i = (scores[i] - scores[y_true] + 1) > 0

        dW[:, i] = dw_i*x

    return dW


def get_gradient_vectorized(x, y_true, margins_vector):
    margins_vector = margins_vector.astype(float)
    margins_vector[y_true] = -(np.sum(margins_vector) - 1)
    x, margins_vector = x[np.newaxis], margins_vector[np.newaxis]

    return x.T.dot(margins_vector)


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin

        dW += get_gradient(W, X[i], y[i], scores)

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.
    Inputs and outputs are the same as svm_loss_naive.
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    num_train = X.shape[0]
    scores = X.dot(W)
    correct_class_scores = np.choose(y, scores.T)

    margins = scores - correct_class_scores[:, np.newaxis] + 1
    positive_margins_indicator = margins > 0
    margins *= positive_margins_indicator

    loss = (np.sum(margins) - 1 * num_train)/num_train
    loss += reg * np.sum(W * W)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################

    for x, y_true, margins_vector in zip(X, y, positive_margins_indicator):
        dW += get_gradient_vectorized(x, y_true, margins_vector)

    dW /= num_train
    dW += reg * 2 * W
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
