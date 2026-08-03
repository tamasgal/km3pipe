PKGNAME=km3pipe
BLACK_TARGETS=src/km3pipe src/km3modules src/pipeinspector examples doc/conf.py

default: build

all: install

install:
	python3 -m pip install .

install-dev:
	python3 -m pip install -e ".[all]"
	python3 -m ipykernel install --user --name="$(PKGNAME)"

build:
	python3 -m build

clean:
	rm -rf build dist src/km3pipe.egg-info reports/junit.xml reports/coverage reports/coverage.xml
	find src -name '__pycache__' -type d -exec rm -rf {} +

test:
	python3 -m pytest --junitxml=./reports/junit.xml src/

benchmark:
	scripts/run_tests.py benchmarks

test-cov:
	python3 -m pytest --cov src/ --cov-report term-missing --cov-report xml:reports/coverage.xml --cov-report html:reports/coverage src/

test-loop:
	python3 -m pytest src/
	ptw --ext=.py,.pyx --ignore=doc src/

doc:
	$(MAKE) -C doc clean html

black:
	black --exclude 'version.py' $(BLACK_TARGETS)

black-check:
	black --check --exclude 'version.py' $(BLACK_TARGETS)

.PHONY: all black black-check build clean doc install install-dev test test-cov test-loop benchmark
