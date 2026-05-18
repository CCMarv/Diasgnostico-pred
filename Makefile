.PHONY: dev test notebook lint

dev:
	pip install -e .[dev]

test:
	pytest pruebas/ -v --tb=short

test-q:
	pytest pruebas/ -q

notebook:
	jupyter nbconvert --to notebook --execute \
		notebooks/02_fenotipado_kmeans.ipynb \
		--output notebooks/02_fenotipado_kmeans_ejecutado.ipynb

lint:
	python -m flake8 entrenamiento/ api/ inferencia/ --max-line-length=100 --ignore=E501
