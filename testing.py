import numpy as np
import matplotlib.pyplot as plt

from noah_neural.tensors import Tensor
from noah_neural.layers import LinearLayer, Network, ReLU




x = np.linspace(-np.pi, np.pi, 150)
y = np.sin(10*x) 

network = Network([
    LinearLayer(1, 128),
    ReLU(),
    LinearLayer(128, 128),
    ReLU(),
    LinearLayer(128, 128),
    ReLU(),
    LinearLayer(128, 1)
])

lr = 0.05
batch_size = 32 
epochs = 10000

plt.ion()
fig, ax = plt.subplots()
ax.plot(x, y, label="y=sin(10x)")
(pred_line,) = ax.plot(x, np.zeros_like(y), label="prediction")
ax.legend()

for epoch in range(epochs):
    shuffled_indices = np.random.permutation(len(x))
    for start in range(0, len(x), batch_size):
        batch_idxs = shuffled_indices[start:start + batch_size]
        x_batch = Tensor(data=x[batch_idxs].reshape(-1, 1))
        targets_batch = y[batch_idxs].reshape(-1, 1)

        pred = network(x_batch)
        loss = ((pred - targets_batch) ** 2).mean()
        loss.bprop()
        network.gradient_descent(lr)

    if epoch % 10 == 0:
        pred_line.set_ydata(network(Tensor(data=x.reshape(-1, 1))).val.ravel())
        fig.canvas.draw_idle()
        plt.title(f"epoch {epoch}/{epochs}")
        plt.pause(0.001)

plt.ioff()
plt.show()


