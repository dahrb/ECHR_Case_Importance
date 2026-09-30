#!/bin/bash
#SBATCH --job-name=audit_tgn_kg
#SBATCH --output=data/data_collection/logs/audit_tgn_kg_art%a_%j.out
#SBATCH --error=data/data_collection/logs/audit_tgn_kg_art%a_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=nodes

set -euo pipefail
: "${ARTICLE:?Must set ARTICLE (3, 6, or 8)}"
cd /users/sgdbareh/scratch/ECHR_Importance
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
/mnt/data1/users/sgdbareh/venvs/ECHR_Importance/bin/python \
    echr/retrieval/audit_tgn_results.py --article "$ARTICLE"
