#! /bin/bash

echo " --- 0.25 Degree ---"
awk '!/ 0 NaN/ && (/NaN/) {print FILENAME, $1, $2, $3, $4}' slurm/verify-0.25-degree/march15/*.out
echo ""
echo ""

echo " --- 1.00 Degree ---"
awk '!/ 0 NaN/ && (/NaN/) {print FILENAME, $1, $2, $3, $4}' slurm/verify-1.00-degree/*.out
echo ""
echo ""

echo " --- 0.25 Degree Subsampled ---"
awk '!/ 0 NaN/ && (/NaN/) {print FILENAME, $1, $2, $3, $4}' slurm/verify-0.25-degree-subsampled/*.out
echo ""
echo ""
