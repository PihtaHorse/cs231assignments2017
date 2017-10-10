import numpy as np
from random import shuffle
from past.builtins import xrange


def get_gradient(W, x, y_true, scores):
    dW = np.zeros(W.shape)

    scores_exp_sum = np.sum([np.exp(score) for score in scores])

    for i in range(dW.shape[0]):
        for j in range(dW.shape[1]):
            dw_ij = (np.exp(scores[j])*x[i])/scores_exp_sum

            if j == y_true:
                dw_ij -= x[i]

            dW[i, j] = dw_ij

    return dW


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train, num_classes = X.shape

    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]

        current_loss = np.log(np.exp(correct_class_score)/np.sum(np.exp(scores)))
        loss -= current_loss

        dW += get_gradient(W, X[i], y[i], scores)
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train, num_classes = X.shape

    scores = X.dot(W)
    correct_class_scores = np.choose(y, scores.T)
    scores_exponents, correct_class_scores_exponents = np.exp(scores), np.exp(correct_class_scores)

    loss += -np.sum(np.log((correct_class_scores_exponents / np.sum(scores_exponents, axis=1))))
    loss += reg * np.sum(W * W)
    loss /= num_train

    for x, y_true, s_exp in zip(X, y, scores_exponents):
        dw_current = np.dot(x[:, np.newaxis], s_exp[np.newaxis, :])/np.sum(s_exp)
        dw_current[:, y_true] -= x

        dW += dw_current

    dW /= num_train
    dW += reg * 2 * W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
