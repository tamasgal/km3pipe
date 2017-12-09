PKGNAME=km3pipe

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

test:
	py.test --junitxml=./junit.xml \
		--cov ./ --cov-report term-missing --cov-report xml
	py.test km3modules

test-nocov:
	py.test --junitxml=./junit.xml
	py.test km3modules

docs:
	cd docs
	make clean
	make html

flake8: 
	py.test --flake8

pep8: flake8

dependencies:
	pip install -Ur requirements.txt

dev-dependencies:
	pip install -Ur dev-requirements.txt

doc-dependencies:
	pip install -Ur sphinx_requirements.txt

.PHONY: all clean build install test test-nocov flake8 pep8 dependencies dev-dependencies doc-dependencies
