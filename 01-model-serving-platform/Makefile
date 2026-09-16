.PHONY: up down data materialize train serve dashboard traffic drift test lint demo

up:          ; docker compose up -d redis mlflow
down:        ; docker compose down -v
data:        ; python -m src.training.generate_data --out data/events.parquet
materialize: ; python -m src.featurestore.materialize --events data/events.parquet
train:       ; python -m src.training.train --events data/events.parquet
serve:       ; uvicorn src.serving.app:app --reload --port 8000
dashboard:   ; streamlit run src/monitoring/dashboard.py
traffic:     ; python scripts/send_traffic.py --n 3000
drift:       ; python -m src.monitoring.drift
test:        ; pytest -q
demo: up data materialize train
	@echo "now run: make serve  (then make traffic, make dashboard)"
