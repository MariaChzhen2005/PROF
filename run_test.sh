#!/usr/bin/env bash
set -e
python3 main_inverter.py
python3 inverter_baselines/inverter_AE.py