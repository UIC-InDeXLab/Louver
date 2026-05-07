import pandas as pd
import matplotlib.pyplot as plt

result = pd.read_csv("result_v1.0_values.csv")

LAST_POSITIONS_TO_SHOW = 2
last_positions = sorted(result["position"].dropna().unique())[-LAST_POSITIONS_TO_SHOW:]
result_for_update_print = result[result["position"].isin(last_positions)].sort_values(
    "position"
)
print(f"Showing positions: {last_positions}")

# Print average update timings per position.
update_time_cols = [
    col
    for col in ["update_time", "update_time_v2", "update_time_v3"]
    if col in result_for_update_print
]
if update_time_cols:
    avg_update_by_position = (
        result_for_update_print.groupby("position", as_index=False)[update_time_cols]
        .mean()
        .sort_values("position")
    )
    print("Average update times per position:")
    for row in avg_update_by_position.itertuples(index=False):
        values = [f"position={int(row.position)}"]
        for col in update_time_cols:
            values.append(f"avg_{col}={getattr(row, col):.6f}s")
        print(", ".join(values))
else:
    print("No update time columns found in CSV.")

# avg_pruning_by_position = result.groupby("position", as_index=False)[
#     "pruning_ratio"
# ].mean()
# print("Average pruning_ratio per position:")
# for row in avg_pruning_by_position.itertuples(index=False):
#     print(f"position={row.position}, avg_pruning_ratio={row.pruning_ratio:.6f}")

pivot = result.pivot(index="position", columns="method", values="time")
pivot = pivot.rolling(1).mean()

plt.figure(figsize=(10, 6))

for method in pivot.columns:
    avg_time = pivot[method].mean()
    plt.plot(pivot.index, pivot[method], label=f"{method} (avg={avg_time:.6f}s)")

plt.xlabel("Query Position")
plt.ylabel("Time (s)")
plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig("benchmark_results_GQA_values.png", dpi=300)
plt.show()
