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
	py.test --junitxml=./junit.xml km3pipe || true

test-km3modules: 
	py.test --junitxml=./junit_km3modules.xml km3modules || true

test-cov:
	py.test --junitxml=./junit.xml \
		--cov ./ --cov-report term-missing --cov-report xml || true
	py.test km3modules || true

test-loop: 
	# pip install -U pytest-watch
	py.test || true
	ptw --ext=.py,.pyx

flake8: 
	py.test --flake8 || true
	py.test --flake8 km3modules || true

pep8: flake8

docstyle: 
	py.test --docstyle  || true
	py.test --docstyle km3modules || true

lint: 
	py.test --pylint || true
	py.test --pylint km3modules || true

dependencies:
	pip install -Ur requirements.txt

dev-dependencies:
	pip install -Ur dev-requirements.txt

doc-dependencies:
	pip install -Ur sphinx_requirements.txt

.PHONY: all clean build install test test-km3modules test-nocov flake8 pep8 dependencies dev-dependencies doc-dependencies docstyle
