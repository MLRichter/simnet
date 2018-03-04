
## Task description

In most machine learning approaches a large amount of data is key to achieve good or human-level performance in classification task.
But while large data collections exists for some well-defined problems, in many real world applications the data provided for
each class is often very small and more or less unbalanced. Furthermore any additional class leads to very expensive retraining of the entire model.

To address this issue of missing training data, unbalanced data and the possibility to retrain the entire model,
a procedure called one-shot classification (Fe-Fei et. al. 2003) can be used. This must be distinguished from zero-shot
classification in which the model have no access to any sample of the target classes (Palatucci et al., 2009).

In this work we use a deep neural network proposed in Koch et. al, 2015 called siamese neural network. In order to validate the siamese neural networks
single shot-classifications performance and the ability to solve the problems of unbalanced data sets we compare the classification
performance between a simple convolutional neural network and a siamese neural network with the same convolutional network structure on
the EMNIST unbalanced 62-classes data set.


References:

Fe-Fei, Li, Fergus, Robert, and Perona, Pietro. A bayesian
approach to unsupervised one-shot learning of object
categories. In Computer Vision, 2003. Proceedings.
Ninth IEEE International Conference on, pp. 1134–
1141. IEEE, 2003.

Palatucci, Mark, Pomerleau, Dean, Hinton, Geoffrey E,
and Mitchell, Tom M. Zero-shot learning with semantic
output codes. In Advances in neural information processing
systems, pp. 1410–1418, 2009.