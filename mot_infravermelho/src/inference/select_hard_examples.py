import pandas as pd
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "runs" / "train_predictions"

TOP_K = 20

df = pd.read_csv(RESULTS_DIR / "train_summary.csv", index_col=0)

df = df.drop(index="OVERALL", errors="ignore")

# Score de dificuldade
df["score"] = (
    2 * (100 - df["MOTA(%)"]) +
    1 * (100 - df["IDF1(%)"]) +
    0.02 * df["FN"] +
    0.005 * df["FP"] +
    0.20 * df["IDs"]
)

hard = df.sort_values("score", ascending=False).head(TOP_K)

out = RESULTS_DIR / "hard_examples.txt"

with open(out, "w") as f:
    for seq in hard.index:
        f.write(seq + "\n")

print("=" * 70)
print(hard[["MOTA(%)", "IDF1(%)", "FP", "FN", "IDs", "score"]])
print("=" * 70)
print(f"\n{len(hard)} sequências salvas em:")
print(out)