PKGNAME=km3pipe
ALLNAMES = $(PKGNAME)
ALLNAMES += km3modules 

default: build

all: install

install:
	pip install .

install-dev:
	pip install -e ".[dev]"
	pip install -e ".[extras]"
	python -m ipykernel install --user --name="km3pipe-dev"

clean:
	python setup.py clean --all

test: 
	py.test --junitxml=./reports/junit.xml -o junit_suite_name=$(PKGNAME) $(ALLNAMES)

benchmark:
	scripts/run_tests.py benchmarks

test-cov:
	py.test --cov ./ --cov-report term-missing --cov-report xml:reports/coverage.xml --cov-report html:reports/coverage $(ALLNAMES)

test-loop: 
	py.test $(ALLNAMES)
	ptw --ext=.py,.pyx --ignore=doc $(ALLNAMES)

.PHONY: black
black:
	black km3pipe
	black km3modules
	black examples
	black pipeinspector
	black doc/conf.py
	black setup.py

.PHONY: black-check
black-check:
	black --check km3pipe
	black --check km3modules
	black --check examples
	black --check pipeinspector
	black --check doc/conf.py
	black --check setup.py

.PHONY: all black black-check clean install install-dev test test-cov test-loop benchmark
