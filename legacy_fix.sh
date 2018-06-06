#!/usr/bin/env zsh
set -e

if [ $# -eq 0 ]
  then
    echo "No version number supplied"
    exit 1
fi

export VERSION=$1

git checkout legacy
git pull
git push

vim km3pipe/__version__.py
git add km3pipe/__version__.py

git commit -m "[LEGACY] Bumps version number"

TITLE="KM3Pipe ${VERSION}"
echo "${TITLE}" > docs/version.txt
echo "$(printf '=%.0s' {1..${#TITLE}})" >> docs/version.txt
git add docs/version.txt
git commit -m "[LEGACY] update version tag in docs"

vim CHANGELOG.rst
git add CHANGELOG.rst
git commit -m "[LEGACY] Bumps changelog"

git tag ${VERSION}

rm -rf dist
python setup.py sdist
twine upload dist/*

git push
git push --tags
