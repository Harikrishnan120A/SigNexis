"""Print final evaluation results from production model."""
import json
import numpy as np

with open("models/evaluation_results.json") as f:
    d = json.load(f)

print(f"n_train  : {d['n_train']}")
print(f"n_test   : {d['n_test']}")
print(f"accuracy : {d['accuracy']*100:.2f}%")

cm = np.array(d["confusion_matrix"])
classes = d["classes"]
print("\nConfusion Matrix (rows=True label, cols=Predicted):")
header = "".join(f"{c[:7]:>9s}" for c in classes)
print(f"{'':18s}{header}")
for i, cls in enumerate(classes):
    row = "".join(f"{cm[i,j]:9d}" for j in range(len(classes)))
    print(f"{cls[:18]:18s}{row}")

per_class = cm.diagonal() / cm.sum(axis=1)
print("\nPer-class recall:")
for c, r in zip(classes, per_class):
    print(f"  {c:30s}: {r:.3f}")

print("\nTop 5 features by importance:")
for name, imp in sorted(d["feature_importance"].items(), key=lambda x: -x[1])[:5]:
    print(f"  {name:30s}: {imp:.4f}")
