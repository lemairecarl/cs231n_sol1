# Adapted from Stanford CS231n -- http://cs231n.stanford.edu
# By Carl Lemaire <carl.lemaire@usherbrooke.ca>

import numpy as np

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    sum_incorrect = 0  # <<< for gradient
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        sum_incorrect += 1  # <<< for gradient
        dW[:, j] += X[i]  # <<< for gradient
    dW[:, y[i]] += -sum_incorrect * X[i]  # <<< for gradient

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  dW /= num_train
  dW += reg * W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  delta = 1.0

  scores = X.dot(W)
  correct_mask = np.arange(0, num_classes)[None, :] == y[:, None]  # element (i,j) = 1 if gt_label(i) = j; else 0
  correct_class_score = scores[correct_mask][:, None]
  temp = scores - correct_class_score + np.array([delta])
  temp[correct_mask] = 0
  margins = np.maximum(0, temp)
  loss = np.sum(margins) / num_train

  loss += 0.5 * reg * np.sum(np.square(W))
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
  dloss_dmargins = np.ones_like(margins) / num_train
  dmargins_dtemp = (temp > 0)
  dmargins_dccs = np.zeros_like(scores)  # ccs: correct_class_score
  dmargins_dccs[correct_mask] = np.sum(dmargins_dtemp, axis=1)
  dmargins_dscores = dmargins_dtemp - dmargins_dccs  # 'scores' receives gradients from 'temp' and from 'ccs'
  dloss_dscores = dloss_dmargins * dmargins_dscores
  dW = X.T.dot(dloss_dscores)

  dW += reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
