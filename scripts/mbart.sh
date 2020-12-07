
cd /home/MLTL/yiran-wang/PycharmProjects/torchglyph

NUM_DEVICES=1 /home/MLTL/yiran-wang/miniconda3/bin/python3 -m examples tune_mbart_translator \
  --study ${STUDY:-"demo"} --batch-size ${BATCH:-"1024"} --label-smoothing ${SMOOTH:-"0.1"} \
  --lr ${LR:-"5e-4"} --beta2 ${BETA2:-"0.98"} \
  --weight-decay ${DECAY:-"1e-2"} --seed ${SEED:-"42"} --tuning ${TUNING:-"qof"} \
  --num-training-steps ${TRAINING:-"100000"} --num-warmup-steps ${WARMUP:-"4000"}