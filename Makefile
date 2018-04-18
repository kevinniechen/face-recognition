SHELL = /bin/sh

define PROJECT_HELP_MSG

Usage:
	make help			show this message
	make clean 			remove intermediate files
	make run			quickstart classification

endef
export PROJECT_HELP_MSG

help:
	echo $$PROJECT_HELP_MSG | less

CLEAN_EXT = *.pyc
CLEAN_DIR = data/interim/illumination/* data/interim/data/* data/interim/pose/* data/processed/*

clean:
	rm -rf ${CLEANUP}
	rm -rf ${CLEAN_DIR}

run:
	cd src/; \
	python3 -W ignore test.py

.PHONY: help clean
