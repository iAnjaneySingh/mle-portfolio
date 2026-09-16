.PHONY: mlflow run sweep walkforward test

mlflow:      ; mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db
run:         ; python run_backtest.py --strategy multi_agent
sweep:       ; python run_backtest.py --strategy multi_agent --sweep
walkforward: ; python run_backtest.py --strategy multi_agent --walk-forward
baseline:    ; python run_backtest.py --strategy buy_hold --no-mlflow
test:        ; pytest -q
