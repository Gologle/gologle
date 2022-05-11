clean:
	rm *.db

test:
	python -m pytest

run:
	uvicorn src.main:app
