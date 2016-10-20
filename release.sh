#!/bin/bash
set -e

if [ $# -eq 0 ]
  then
    echo "No version number supplied"
    exit 1
fi

VERSION=$1
SVN="http://svn.km3net.de/auxiliaries/KM3Pipe"

git checkout develop
git pull
git push

git checkout master
git pull

git checkout develop
git pull

git flow release start "${VERSION}"

vim km3pipe/__version__.py
git add km3pipe/__version__.py

git commit -m "Bumps version number"

echo "KM3Pipe $(km3pipe --version)\n=============" > docs/version.txt
git add docs/version.txt
git commit -m "update version tag in docs"

vim CHANGELOG.rst
git add CHANGELOG.rst
git commit -m "Bumps changelog"

git flow release finish "${VERSION}"

python setup.py sdist register upload

git checkout master
git push
git checkout develop
git push
git push --tags

git checkout svn
git merge master
git svn dcommit
git checkout develop

svn copy "${SVN}/git" "${SVN}/tag/v${VERSION}" -m "KM3Pipe Release ${VERSION}"
