.PHONY : typecheck test

all : typecheck test run

typecheck:
	python3 -m mypy --ignore-missing-imports burp.py levenshtein.py

test :
	python3 -m unittest

run :
	python3 -m burp -vv example_nofunc_predicted.bracket example_nofunc_gold.bracket
