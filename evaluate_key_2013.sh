#!/usr/bin/env bash

if [ -z "$1" ]
then
  echo "please supply a key file as argument";
  exit 1
fi;

echo FNMI
java -jar resources/SemEval-2013-Task-13-test-data/scoring/fuzzy-nmi.jar  resources/SemEval-2013-Task-13-test-data/keys/gold/all.key $1
echo
echo FBC
java -jar resources/SemEval-2013-Task-13-test-data/scoring/fuzzy-bcubed.jar  resources/SemEval-2013-Task-13-test-data/keys/gold/all.key $1