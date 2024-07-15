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
- Extend leave-one-out cross validation to remove/reduce other bases, based on a comparability value between the bases.


Furthermore, what kind of batch normalization is needed in addition to scale to ensure that derivative in relation to the knots shift position does not explode or vanish?
