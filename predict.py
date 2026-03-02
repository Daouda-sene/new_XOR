
import torch
import numpy as np
from model import XORNet

X = torch.tensor([[0.,0.],
                  [0.,1.],
                  [1.,0.],
                  [1.,1.]])

model = XORNet()

model.eval()

with torch.no_grad():
    outputs = model(X)
    preds = (outputs > 0.5).int().numpy()

np.savetxt("predictions.txt", preds, fmt="%d")
print("Predictions saved.")

