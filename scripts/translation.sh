cd /home/MLTL/yiran-wang/PycharmProjects/torchglyph

NUM_DEVICES=8 /home/MLTL/yiran-wang/miniconda3/bin/python3 -m examples train_translator \
  --study ${STUDY:-"demo"} --lr ${LR:-"7e-4"} --label-smoothing ${SMOOTH:-"0.1"} \
  --enc-self-algo 