.PHONY : typecheck test

all : typecheck test

typecheck:
	python3 -m mypy --ignore-missing-imports burp.py

test :
	python3 -m unittest
