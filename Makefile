.PHONY: install run format test test-unit test-eval docker-build docker-run

SERVICE_NAME = multi-model-trust

install:
	python -m pip install -r requirements.txt

run:
	@if [ -f .env ]; then \
		python -m dotenv run -- uvicorn main:app --host 0.0.0.0 --port 8000 --reload; \
	else \
		uvicorn main:app --host 0.0.0.0 --port 8000 --reload; \
	fi

format:
	python -m black .

test:
	python -m pytest -q

test-unit:
	python -m pytest -q tests/unit

test-eval:
	python -m pytest -q tests/evaluation

docker-build:
	docker build -t $(SERVICE_NAME):local .

docker-run:
	docker run --rm -p 8000:8000 $(SERVICE_NAME):local
