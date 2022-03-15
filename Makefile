PKGNAME=km3pipe

default: build

all: install

install:
	python3 -m pip install .

install-dev:
	python3 -m pip install -e ".[dev]"
	python3 -m pip install -e ".[extras]"
	python3 -m ipykernel install --user --name="km3pipe-dev"

clean:
	python3 setup.py clean --all

test: 
	py.test --junitxml=./reports/junit.xml -o junit_suite_name=$(PKGNAME) src/

benchmark:
	scripts/run_tests.py benchmarks

test-cov:
	py.test --cov src/ --cov-report term-missing --cov-report xml:reports/coverage.xml --cov-report html:reports/coverage src/

test-loop: 
	py.test src/
	ptw --ext=.py,.pyx --ignore=doc src/

.PHONY: black
black:
	black  --exclude 'version.py' src/km3pipe
	black src/km3modules
	black examples
	black src/pipeinspector
	black doc/conf.py
	black setup.py

.PHONY: black-check
black-check:
	black --check  --exclude 'version.py' src/km3pipe
	black --check src/km3modules
	black --check examples
	black --check src/pipeinspector
	black --check doc/conf.py
	black --check setup.py

.PHONY: all black black-check clean install install-dev test test-cov test-loop benchmark
