#!/bin/bash
set -e

if [ $# -eq 0 ]
  then
    echo "No version number supplied"
    exit 1
fi

VERSION=$1

git checkout develop
git pull

git flow release start "${VERSION}"

vim km3pipe/__version__.py
git add km3pipe/__version__.py
git commit -m "Bumps version number"

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
