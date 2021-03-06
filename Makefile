PKGNAME=km3pipe
ALLNAMES = $(PKGNAME)
ALLNAMES += km3modules 

default: build

all: install

build: 
	@echo "No need to build anymore :)"

install: 
	pip install numpy
	pip install .

install-dev: dependencies
	pip install numpy
	pip install -e .

clean:
	python setup.py clean --all
	rm -f $(PKGNAME)/*.cpp
	rm -f $(PKGNAME)/*.c
	rm -f -r build/
	rm -f $(PKGNAME)/*.so

test: 
	py.test --junitxml=./reports/junit.xml -o junit_suite_name=$(PKGNAME) $(PKGNAME)

test-km3modules: 
	py.test --junitxml=./reports/junit_km3modules.xml -o junit_suite_name=km3modules km3modules

benchmark:
	scripts/run_tests.py benchmarks

test-cov:
	py.test --cov ./ --cov-report term-missing --cov-report xml:reports/coverage.xml --cov-report html:reports/coverage $(ALLNAMES)

test-loop: 
	py.test $(PKGNAME) km3modules
	ptw --ext=.py,.pyx --ignore=doc $(PKGNAME) km3modules

flake8: 
	py.test --flake8
	py.test --flake8 km3modules

pep8: flake8

docstyle: 
	py.test --docstyle
	py.test --docstyle km3modules

lint: 
	py.test --pylint
	py.test --pylint km3modules

dependencies:
	pip install numpy
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

.PHONY: yapf
yapf:
	yapf -i -r km3pipe
	yapf -i -r km3modules
	yapf -i -r examples
	yapf -i -r pipeinspector
	yapf -i doc/conf.py
	yapf -i setup.py

.PHONY: all clean build install install-dev test test-cov test-loop benchmark test-km3modules flake8 pep8 dependencies docstyle
