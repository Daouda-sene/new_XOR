import numpy as np
import json
import sys

if len(sys.argv) < 2:
    print("Submission file missing")
    sys.exit(1)

submission_path = sys.argv[1]
preds = np.loadtxt(submission_path)

y_true = np.array([0,1,1,0])

accuracy = np.mean(preds == y_true)

with open("score.json", "w") as f:
    json.dump({"accuracy": float(accuracy)}, f)

print("Accuracy:", accuracy)
