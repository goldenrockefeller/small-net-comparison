# small-net-comparison
A comparison in the behavior of different small neural network architectures on a concave function.

Relus get stuck at poor minima with poor "out-of-distribution" initialization. This is because, the relu activation attempts only linear regression at far initialization, and changing the bias weights (or knots) for Relu will not change the regression (derivative is zero).

Inverse quadratic also work well with somewhat bad initialization, but activation is very local, and the derivative falls drastically outside a range that prevents the centers (knots) of the inverse quadratic from better representing the data. This version 
of the inverse quadratic alleviates this by make the derivative at x=inf to be not zero. Further investigation is encouraged.

Swish, Gelu, Mish, Softplus family of activations work well with "out-of-distribution" initialization. They mix the semi-global effects of relus with the semi-local effects of sigmoid to ensure derivatives are non-zero at x=inf, but also that information exist to push the knots inwards toward the input (x) distribution when necessary. Swish works the best at this and is able to approximate a sinusoid function with enough activation nodes. 

Radial basis functions are harder to get right with an automatic schedule. For 1 dimension, one can use cross entropy minimization to set the radial basis knots and centers. Smoothing criteria can ensure that the rbfs overlap nicely and that nearby rbfs don't differ too much in their width. It is unclear how such constraints can apply to multiple dimensions or multiple overlapping 1d rbfs. Additionally, how does the learning algorithms work with rbf knots setting simultaneous? Fi

Performed all different types of small neural networks and settled on one that is reasonably robust.

I am working with a modified version of the inverse quadratic. It is fairly robust to a little overspread, overspread, and offset of the knots.
Maybe in the future, I can look into fixing parameters to the data to further reduce the effects of knots (something to mimic the effects of scale and/or shifting the data, but still the model to make fine tuning adjustments).
The question becomes, underwhat metric for scaling will the model learn the faster: e.g. maximize the norm of the gradient with respect to shifting (a separate but parallel learning process)

The best activation from a derivative perspective is the exponentially decay sine wave. This is a global activation, and the representational space increases as more sine waves are overlapped over each other. Regardless of initial frequency, with enough machine precision, a single sine wave of any frequency can be found. Thus, it is likely that any function can be approximated to arbitrary precision with enough exponentially decay sine wave activations. The only problem is that the exponential component explodes quickly for out-of-distribution input, which might not work well with stochastic gradient.

Ultimately, the results of this study is that the best solution is to use swish and make sure that the knots for the activation are placed inside the distribution as best as possible.
