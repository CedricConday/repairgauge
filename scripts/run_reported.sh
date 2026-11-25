#!/bin/bash
# Reported backtests at frozen defaults. Log ends "reported done".
cd ~/repos/repairgauge; V=~/.venvs/repairgauge/bin; L=work/logs/reported.log
echo "$(date "+%F %T") start $(git rev-parse --short HEAD)" >> $L
$V/repairgauge backtest --out results/phantom --jobs 4 > work/logs/reported_phantom.log 2>&1
echo "$(date "+%F %T") phantom exit=$?" >> $L
$V/python scripts/real_backtest.py --out results/real_mirror --subjects sub-24762 sub-60476 sub-60501 sub-60505 --seeds 0 1 2 3 4 --jobs 4 > work/logs/reported_real.log 2>&1
echo "$(date "+%F %T") real exit=$?" >> $L
echo "$(date "+%F %T") reported done" >> $L
