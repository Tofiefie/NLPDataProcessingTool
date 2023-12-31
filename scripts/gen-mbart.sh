rm -f scripts/mbart.mk
touch scripts/mbart.mk

declare -i counter=1
targets=()

for BATCH in 1024; do
  for SMOOTH in 0 0.1; do
    for LR in 3e-3 7e-3 3e-4 7e-4 3e-5 7e-5; do
      for BETA2 in 0.98 0.999; do
        for DECAY in 1e-1 1e-2 1e-4; do
          for TUNING in "qof" "full"; do

            for SEED in 42; do

              echo "exp${counter}:" >>scripts/mbart.mk
              echo "\tSTUDY=grid-mbart BATCH=${BATCH} SMOOTH=${SMOOTH} TUNING=${TUNING} BETA2=${BETA2} LR=${LR} DECAY=${DECAY} zsh scripts/mbart.sh" >>scripts/mbart.mk
              echo "" >>scripts/mbart.mk
              targets+=("exp${counter}")
              counter+=1

            done

          done
        done
      done
    done
  done
done

echo "all: ${targets}" >>scripts/mbart.mk
echo "" >>scripts/mbart.mk
