.PHONY : typecheck test

all : typecheck test run

typecheck:
	python3 -m mypy --ignore-missing-imports burp.py levenshtein.py

test :
	python3 -m unittest

run :
	python3 burp.py -vv example_nofunc_predicted.bracket example_nofunc_gold.bracket
