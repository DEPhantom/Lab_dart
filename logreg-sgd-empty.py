#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import sys

import math
import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.metrics
import sklearn.model_selection
import sklearn.linear_model
import sklearn.preprocessing
import random


def load_train_test_data(train_ratio=.5):
    data = pandas.read_csv('./HTRU_2.csv', header=None, names=['x%i' % (i) for i in range(8)] + ['y'])
    X = numpy.asarray(data[['x%i' % (i) for i in range(8)]])
    X = numpy.hstack((numpy.ones((X.shape[0],1)), X))
    y = numpy.asarray(data['y'])

    return sklearn.model_selection.train_test_split(X, y, test_size = 1 - train_ratio, random_state=0)


def scale_features(X_train, X_test, low=0, upp=1):
    minmax_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(low, upp)).fit(numpy.vstack((X_train, X_test)))
    X_train_scale = minmax_scaler.transform(X_train)
    X_test_scale = minmax_scaler.transform(X_test)
    return X_train_scale, X_test_scale

def update_y_hat( X, y_hat, theta, theta_0 ):

  for i in range( len(X) ):
    linear_reg = numpy.dot( X[i], theta ) + theta_0
    y_hat_i = 1/( 1+math.exp( -(linear_reg) ) )
    y_hat[i] = y_hat_i
  # end for
  
  return y_hat
  
# end update_y_hat()

def cross_entropy(y, y_hat):
    loss = 0
    for i in range(len(y)):
        loss += -(y[i]*math.log(y_hat[i]) + (1-y[i])*math.log(1-y_hat[i]))
    return loss

def stochastic_gradient_descent(x, y, y_hat, theta, theta_0, alpha):
  # optimizer sgd
  x = numpy.array( x )
  i = random.randint( 0, len(x)-1 )

  gradient = ( ( y[i]-y_hat[i] )*x[i] )*alpha
  theta = theta+numpy.reshape( gradient, ( len(theta), 1 ) )
  gradient_0 = (y[i]-theta_0)*1*alpha 
  theta_0 = theta_0+gradient_0
  
  return theta, theta_0, gradient
  
# end stochastic_gradient_descent()

def logreg_sgd(X, y, alpha = .001, epochs = 100, eps=1e-4):
    # TODO: compute theta
    # alpha: step size
    # epochs: max epochs
    # eps: stop when the thetas between two epochs are all less than eps 
   
    count = 0
    is_converge = False
    n, d = X.shape
    theta = numpy.zeros((d, 1))
    theta_0 = 0
    y_hat = numpy.array([])
    for i in range( len(y) ):
      linear_reg = numpy.dot( X[i], theta ) + theta_0
      y_hat_i = 1/( 1+math.exp( -(linear_reg) ) )
      y_hat = numpy.append( y_hat, [y_hat_i], axis=0 )
    # end for

    cross_entropy(y, y_hat)   

    while( count < epochs and not is_converge ):
      y_hat = update_y_hat( X, y_hat, theta, theta_0 )
      loss = cross_entropy(y, y_hat)
      theta, theta_0, gradient = stochastic_gradient_descent(X, y, y_hat, theta, theta_0, alpha)
      
      is_converge = True # assumpt that all less than eps
      for i in range( len(gradient) ):
        if( abs(gradient[i]) > eps ): 
          is_converge = False
        # end if
        
      # end for
      
      if ( count%10 == 0 ) :
        print( "Train Epoch", '%02d' % (count+1), "\nLoss=", "{:.9f}".format(loss), "\n")
      # end if
      
      count = count+1
      
    # end while
    
    return theta
    
# end logreg_sgd()


def predict_prob(X, theta):
    return 1./(1+numpy.exp(-numpy.dot(X, theta)))


def plot_roc_curve(y_test, y_prob):
    # TODO: compute tpr and fpr of different thresholds
    tpr = []
    fpr = []
    tpr_current = 0
    fpr_current = 0
    tp_count = 0
    fp_count = 0
    true_positive = list(y_test).count( 1 )
    false_positive = list(y_test).count( 0 )
    y_test_sort = []
    y_prob_sort = []
    """
    list( set(a) )
    [0, 1]
    """
    y_prob_sort.append( y_prob[0] )
    y_test_sort.append( y_test[0] )
    n = len(y_prob)
    for i in range(n-1):
        key = y_prob[i+1]
        j = i
        while j >=0 and key > y_prob_sort[j] :            
            j -= 1
        # end while
        
        y_prob_sort.insert( j+1, y_prob[i+1] )
        y_test_sort.insert( j+1, y_test[i+1] )
        
    # end while
    
    
    tpr.append( tpr_current )
    fpr.append( fpr_current )
    for i in range( len( y_test_sort ) ) :
      if ( y_test_sort[i] == 1 ) : # true
        tp_count = tp_count + 1
        tpr_current = tp_count/true_positive
      # end if
      else : # false
        fp_count = fp_count + 1
        fpr_current = fp_count/false_positive
      # end else
      
      tpr.append( tpr_current )
      fpr.append( fpr_current )
      
    # end if
    
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("roc_curve.png")


def main(argv):
    X_train, X_test, y_train, y_test = load_train_test_data(train_ratio=.5)
    X_train_scale, X_test_scale = scale_features(X_train, X_test, 0, 1)
    
    theta = logreg_sgd(X_train_scale, y_train)
    print(theta)
    y_prob = predict_prob(X_train_scale, theta)
    y_pred = (y_prob > .5)
    y_pred = y_pred.flatten()
    y_pred[0] = True # 最後要拿掉
    print("Logreg train accuracy: %f" % (sklearn.metrics.accuracy_score(y_train, y_pred)))
    print("Logreg train precision: %f" % (sklearn.metrics.precision_score(y_train, y_pred)))
    print("Logreg train recall: %f" % (sklearn.metrics.recall_score(y_train, y_pred)))
    y_prob = predict_prob(X_test_scale, theta)
    y_pred = (y_prob > .5)
    y_pred = y_pred.flatten()
    y_pred[0] = True # 最後要拿掉
    print("Logreg test accuracy: %f" % (sklearn.metrics.accuracy_score(y_test, y_pred)))
    print("Logreg test precision: %f" % (sklearn.metrics.precision_score(y_test, y_pred)))
    print("Logreg test recall: %f" % (sklearn.metrics.recall_score(y_test, y_pred)))
    plot_roc_curve(y_test.flatten(), y_prob.flatten())


if __name__ == "__main__":
    main(sys.argv)


