.PHONY: up down topic produce dump stream backfill parity test

up:       ; docker compose up -d kafka redis
down:     ; docker compose down -v
produce:  ; python -m src.producer.synthetic_producer --rate 200 --duration 600
dump:     ; python -m src.producer.synthetic_producer --duration 120 --dump data/events.parquet
stream:   ; docker compose up spark
backfill: ; python -m src.batch.backfill --events data/events.parquet --out data/training_features.parquet
parity:   ; python scripts/run_parity.py --batch data/training_features.parquet
test:     ; pytest -q
