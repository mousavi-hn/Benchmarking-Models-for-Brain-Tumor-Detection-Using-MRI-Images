import os
import pandas as pd

from src import data, train, models

# RUN ALL MODELS
all_results = []

for model_name in models.MODEL_NAMES:
    try:
        result = train.train_and_evaluate(model_name, data.train_df, data.val_df, data.test_df)
        all_results.append(result)
    except Exception as e:
        print(f"\nModel {model_name} failed with error:")
        print(str(e))

# SAVE FINAL COMPARISON TABLE
if all_results:
    results_df = pd.DataFrame(all_results)

    # Sort by ROC-AUC first, then F1, then recall
    results_df = results_df.sort_values(
        by=["roc_auc", "f1_score", "recall_sensitivity"],
        ascending=False
    ).reset_index(drop=True)

    results_df.to_csv(os.path.join(data.OUTPUT_DIR, "cnn_benchmark_summary.csv"), index=False)

    print("\nFinal ranking:")
    print(results_df[
        [
            "model_name",
            "accuracy",
            "precision",
            "recall_sensitivity",
            "specificity",
            "f1_score",
            "roc_auc",
            "tp", "tn", "fp", "fn",
            "training_time_sec"
        ]
    ])

else:
    print("No model finished successfully.")