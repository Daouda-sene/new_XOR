import torch
import numpy as np
from model import Net

# Données XOR fixes
X = torch.tensor([
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.]
])

model = Net()

# Entraînement rapide
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

y = torch.tensor([[0.], [1.], [1.], [0.]])

for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

# Prédictions
with torch.no_grad():
    preds = model(X)
    preds = (preds > 0.5).int().numpy()

np.savetxt("predictions.txt", preds, fmt="%d")
