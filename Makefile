install:
	pip install -r requirements.txt

train:
	python src/train.py

api:
	uvicorn app.api:app --reload

test:
	python -m pytest -v

coverage:
	pytest --cov=src

docker:
	docker compose up --build