# small-net-comparison
A comparison in the behavior of different small neural network architectures on a concave function.

Takeaways:

- Nodes have to be distributed. The model will hit a local minimum on underspread, regardless of neural network architecture.
- Randomly distributing nodes is not good enough, fixing the nodes in a uniform distribution works so much better.


Distributing nodes:
fuzzy c-means
fuzzy c-means weighted by error
Alternate between nodes and model training

Another must have in batch normalization.

Models to try:
fuzzy linear
relu
piecewise linear, fixed ends
piecewise linear
