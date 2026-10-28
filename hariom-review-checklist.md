# Hariom's review checklist

Things to review or understand for the STL-10 mini-Inception work, in the order I would do them.

- the data lives in a manifold, everyone says so. very good post here. https://stats.stackexchange.com/questions/66939/what-is-the-manifold-assumption-in-semi-supervised-learning
- The same thing is true inside the layers.
- When I do feature visualisation to optimise the image to the minimal portion that recreates the neurons activation map, I optimise the input using gradient descent. The problem is that gradient descent can go wonk and use any path. But the path of usefulness to the neuron is already defined by our density forest. It is a small set of paths compared to the full set. I need to use feature visualisation to slowly move the hadamard product to background somehow and see how the image changes.
