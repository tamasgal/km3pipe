PKGNAME=km3pipe

default: build

all: install

build: 
	@echo "No need to build anymore :)"

install: 
	pip install ".[full]"

install-dev: dev-dependencies
	pip install -e ".[full]"

clean:
	python setup.py clean --all
	rm -f $(PKGNAME)/*.cpp
	rm -f $(PKGNAME)/*.c
	rm -f -r build/
	rm -f $(PKGNAME)/*.so

test: 
	py.test --junitxml=./reports/junit.xml km3pipe

test-km3modules: 
	py.test --junitxml=./reports/junit_km3modules.xml km3modules

test-cov:
	py.test --cov ./ --cov-report term-missing --cov-report xml:reports/coverage.xml --cov-report html:reports/coverage km3pipe km3modules pipeinspector

test-loop: 
	# pip install -U pytest-watch
	py.test
	ptw --ext=.py,.pyx --ignore=doc

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
	pip install -Ur requirements.txt

dev-dependencies:
	pip install -Ur dev-requirements.txt

doc-dependencies:
	pip install -Ur sphinx_requirements.txt

.PHONY: all clean build install test test-km3modules test-nocov flake8 pep8 dependencies dev-dependencies doc-dependencies docstyle
