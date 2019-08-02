This repo contains MRC OSI code related to BERT quantization.

Reference NLP task is combined Intent-Slot classification, as in <https://paperswithcode.com/paper/bert-for-joint-intent-classification-and-slot>

References to interesting papers are collected in [./doc/QuantizationInNLP.md](./doc/QuantizationInNLP.md)


To run the Docker environment shell, run:

```
$ ./rundocker.sh docker/murygin-bert.docker

```

and then `cd murygin/bert`. Pass `--no-map-sockets` in case of socket binding
conflict.

To run the experiment, see [./murygin/bert/README.md](./murygin/bert/README.md).


