# small-net-comparison
A comparison in the behavior of different small neural network architectures on a concave function.

Takeaways:

- Nodes have to be distributed. The model will hit a local minimum on underspread, regardless of neural network architecture.
- Randomly distributing nodes is not good enough, fixing the nodes in a uniform distribution works so much better.
- Fuzzy-linear model works ok but has too many ripples, Fuzzy-constant is way worse. Both is good with a bit of smoothing (how much smoothing?)
- 
- Batch normalization may be very important, even in small models.


Distributing nodes:
- fuzzy c-means
- fuzzy c-means weighted by error (or square of output derivative) (will this cause seperation on same axis? won't investigate now, but if I would, I would weight the others by the dot product of the axis)
- Alternate between nodes and model training

Another must have is batch normalization.

Models to try:
- relu
- fuzzy-linear
- -fuzzy-constant

How to smooth:
1) separate or one smoothing factor?
-smooth is gradient descent,
- smooth is based on per point statistics
