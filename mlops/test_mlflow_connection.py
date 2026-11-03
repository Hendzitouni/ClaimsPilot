import subprocess

import mlflow
from dotenv import load_dotenv

load_dotenv()  # reads .env and sets the MLFLOW_* variables

git_commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()

mlflow.set_experiment("claimspilot-platform-test")

with mlflow.start_run(run_name="connection-test"):
    mlflow.set_tags({"owner": "mohamed", "git_commit": git_commit, "sprint": "1"})
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("epochs", 5)
    for epoch in range(5):
        mlflow.log_metric("accuracy", 0.6 + epoch * 0.07, step=epoch)

print("Run logged. Check the Experiments tab on DagsHub.")