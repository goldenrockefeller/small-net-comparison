# small-net-comparison
A comparison in the behavior of different small neural network architectures on a concave function.

Takeaways:

- Nodes have to be distributed. The model will hit a local minimum on underspread, regardless of neural network architecture.
- Randomly distributing nodes is not good enough, fixing the nodes in a uniform distribution works so much better.
- Fuzzy architectures needs a smoothing factor. Fuzzy-linear is generally better than fuzzy-constant.
- Fuzzy-linear model works well even on underspread (when preforming gradient descent on the nodes too). Does not work well in overspread, so don't need to apply gradient descent to nodes.
- Batch normalization may be very important, even in small models.
- Fuzzy models have ripples on the exponential, but it is not clear that a Relu's error would be better or worse. Truly a case of there being no free-lunch in gradient descent


Distributing nodes:
- fuzzy c-means [Y]
- fuzzy c-means weighted by error (or square of output derivative) [Y]
- (will this cause seperation on same axis? won't investigate now, but if I would, I would weight the others by the dot product of the axis)
- Alternate between nodes and model training (won't investigate now) [_]

Another must have is batch normalization.

Models to try:
- relu
- fuzzy-linear [Y]
- -fuzzy-constant [Y]

How to smooth:
1) separate or one smoothing factor?
-smooth is gradient descent,
- smooth is based on per point statistics [Y], but not some much as to make the error worse than before
