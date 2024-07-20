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
The best autoscaling idea that works is to perform cross entropy minimization with the normalized squared gradient of the Gaussian distribution. This will put the data points at roughly the place where the gradient
 of the Gaussian is maximum, while also allowing for a smooth fuzzy clustering of the data. Hopefully, at this position, the gradient is large enough to move knots closer to the data.
 
Furthermore, what kind of batch normalization is needed in addition to scale to ensure that derivative in relation to the knots shift position does not explode or vanish?
