rm -f scripts/mbart.mk
touch scripts/mbart.mk

declare -i counter=1
targets=()

for BATCH in 1024; do
  for SMOOTH in 0 0.1; do
    for LR in 3e-3