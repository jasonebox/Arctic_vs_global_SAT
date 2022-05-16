#!/bin/sh
echo "enter commit text"
git add .
git commit -m "$1"
git push --all
