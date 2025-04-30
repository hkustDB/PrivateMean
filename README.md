# Ziyue Huang, Yuting Liang, and Ke Yi. Instance-optimal mean estimation under differential privacy (NeurIPS 2021).


| Folder          | Description                                                                                                |
| ----------------| ---------------------------------------------------------------------------------------------------------- |
| data                 | contains MNIST data                                                                                        |
| lpme                 | implementation for the methods in [locally private mean estimation](https://proceedings.mlr.press/v89/gaboardi19a.html) |
| coinpress            | a copy of the code from https://github.com/twistedcubic/coin-press, containing implementation of [coinpress](https://proceedings.neurips.cc/paper/2020/hash/a684eceee76fc522773286a895bc8436-Abstract.html) |
|quantile_binary_search| implementation of our methods                                                                         |

## Dependencies
numpy v1.23.5<br>
scipy v1.9.3<br>
torch v2.1.0+cpu<br>
joblib v1.3.2

The [joblib](https://joblib.readthedocs.io/en/stable/) library is used for computing some functions on different coordinates of the data in parallel (para='0'). Alternatively, the [multiprocessing](https://docs.python.org/3/library/multiprocessing.html) library can be used (para='1'), or sequential computations can be used (para='2').

## Evaluation
To reproduce the experiments in the central model (Fig. 1-8), run:
```test1
python central_tests.py
```
To reproduce the experiments in the local model (Fig. 10-12), run:
```test2
python local_tests.py
```
To reproduce the study on the clipping threshold (Fig. 9), run:
```test3
python run_syn_qt.py
```
