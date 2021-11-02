CUDA_VISIBLE_DEVICES=0 python -W ignore main_binary.py --note 2048-div-10-no-lr-schedule-data-july-2021-binary-orig --lamb 0.1 --temp 14 --n_epoch 200 --kth 101
CUDA_VISIBLE_DEVICES=0 python -W ignore main_binary.py --note 2048-div-10-no-lr-schedule-data-july-2021-binary-orig --lamb 0.1 --temp 8 --n_epoch 200 --kth 84
CUDA_VISIBLE_DEVICES=1 python -W ignore main_binary.py --note 2048-div-10-no-lr-schedule-data-july-2021-binary-orig --lamb 0.1 --temp 8 --n_epoch 200 --kth 82
CUDA_VISIBLE_DEVICES=0 python -W ignore main_three.py --note 2048-div-10-no-lr-schedule-data-july-2021-binary-orig --lamb 0.1 --temp 8 --n_epoch 200 --kth 82
binary
CUDA_VISIBLE_DEVICES=0 python -W ignore main_binary.py --note 2048-div-10-no-lr-schedule-data-july-2021-binary-orig --lamb 0.1 --temp 2 --n_epoch 200 --kth 101
test results: 951.0    31.0    96.84
test results: 3.0    81.0    96.43
96.64
three
CUDA_VISIBLE_DEVICES=0 python -W ignore main_three.py --note 2048-div-10-no-lr-schedule-data-july-2021-binary-orig --lamb 0.1 --temp 10 --n_epoch 200 --kth 82 74025
CUDA_VISIBLE_DEVICES=1 python -W ignore main_three.py --note 2048-div-10-no-lr-schedule-data-july-2021-binary-orig --lamb 0.2 --temp 3 --n_epoch 200 --kth 80 

test results: 654.0    187.0    8.0    77.03
test results: 47.0    80.0    6.0    60.15
test results: 3.0    7.0    74.0    88.1
test Epoch [147/200], Acc: 75.80, Best Acc: 86.87, Avg Acc Per Class: 75.09, Best Avg Acc Per Class: 75.09.