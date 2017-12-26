PKGNAME=km3pipe

default: build

all: install

build: 
	python setup.py build_ext --inplace

install: 
	pip install -e ".[full]"

clean:
	python setup.py clean --all
	rm -f $(PKGNAME)/*.cpp
	rm -f -r build/
	rm -f $(PKGNAME)/*.so

test: build
	py.test --junitxml=./junit.xml \
		--cov ./ --cov-report term-missing --cov-report xml || true
	py.test km3modules || true

test-nocov: build
	py.test --junitxml=./junit.xml || true
	py.test km3modules || true

test-loop: build
	# pip install -U pytest-watch
	py.test || true
	ptw --ext=.py,.pyx --beforerun "make build"

flake8: 
	py.test --flake8 || true
	py.test --flake8 km3modules || true

pep8: flake8

lint: 
	py.test --pylint || true
	py.test --pylint km3modules || true

dependencies:
	pip install -Ur requirements.txt

dev-dependencies:
	pip install -Ur dev-requirements.txt

doc-dependencies:
	pip install -Ur sphinx_requirements.txt

.PHONY: all clean build install test test-nocov flake8 pep8 dependencies dev-dependencies doc-dependencies
