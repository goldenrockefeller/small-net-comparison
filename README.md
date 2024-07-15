# small-net-comparison
A comparison in the behavior of different small neural network architectures on a concave function.

Performed all different types of small neural networks and settled on one that is reasonably robust.

I am working with a modified version of the inverse quadratic. It is fairly robust to a little overspread, overspread, and offset of the knots.
Maybe in the future, I can look into fixing parameters to the data to further reduce the effects of knots (something to mimic the effects of scale and/or shifting the data, but still the model to make fine tuning adjustments).
The question becomes, underwhat metric for scaling will the model learn the faster: e.g. maximize the norm of the gradient with respect to shifting (a separate but parallel learning process)

Automatic Scaling:
- Must make the model learning process invariant to the scale and shift of the distribution of data
- Should only depend on the distribution of data, and not the target values
- Put the model learning process in a good place to learn
- Should work reasonably, even if there is only one knot.
- Should be reasonably fast, adaptive and combine well with model learning without making learning too suboptimal or too unstable.

AutoScaling Ideas:
- Take the cross entropy between the bases, and the data distribution.
- Which direction of cross entropy? bases to data or vice versa?
- Should the bases mixture model total responsibilities be fixed or not? Probably fixed, but other is possible.
- In addition to cross entropy, should I maximize or minimize the entropy of the bases distribution (or neither).
- Perhaps, use the average entropy evalution after removing one bases (for better generalization ability). i.e. leave-one-out cross validation

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
