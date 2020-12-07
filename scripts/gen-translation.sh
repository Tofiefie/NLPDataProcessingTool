
for LR in 3e-4 7e-4; do
  for SMOOTH in 0.0; do
    for ALGO in "dot" ; do

      qsub -q full-gpu -l ngpus=8 -l ncpus=2 -v "STUDY=tran-go4,LR=${LR},SMOOTH=${SMOOTH},ALGO=${ALGO}" scripts/translation.sh

    done
  done
done