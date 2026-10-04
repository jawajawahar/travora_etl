.PHONY: install test dataset extract clean enrich matrix load card all

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v

extract:
	python -m etl.extract_osm

clean:
	python -m etl.clean

enrich:
	python -m etl.enrich

matrix:
	python -m etl.travel_matrix

load:
	python -m etl.load_neo4j

card:
	python -m etl.data_card

dataset: extract clean enrich matrix card

all: install test dataset load card
