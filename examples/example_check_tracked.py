from pathlib import Path
from typing import Dict

import pandas as pd

from octoflow.tracking import SQLAlchemyTrackingStore, TrackingClient
from octoflow.tracking.models import TreeNode

run_path = Path("./examples/logs/experiment_1")

dburi = f"sqlite:///{run_path / 'tracking.db'}"

store = SQLAlchemyTrackingStore(dburi)

client = TrackingClient(store)

expr = client.get_experiment_by_name("example_hf_model_finetune")

branches: Dict[str, list] = {}

for run in expr.search_runs():
    values = run.get_values()
    tree = TreeNode.from_values(values)
    for key, value in tree.flatten().items():
        if key not in branches:
            branches[key] = []
        branches[key].extend(value)


for _, branch in branches.items():
    df = pd.DataFrame.from_records(branch)
    df.columns = [c if isinstance(c, str) else ".".join(c) for c in df.columns]
    print(df)
