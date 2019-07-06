#!/bin/bash

spark-submit \
  --master local[*] \
  --class com.rossmann.store.sales.RossmannStoreSales \
  rossmannstoresales_2.11-0.1.jar \
  /data/Sales/train.csv /data/Sales/test.csv /data/Sales/result.csv
