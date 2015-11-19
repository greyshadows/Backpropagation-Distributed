# Vinay Chandragiri 120101018

Distributed implementation of Neural Networks in pyspark

sbin/start-master.sh  to start master
SPARK_WORKER_INSTANCES=4 SPARK_WORKER_CORES=4 ./sbin/start-slaves.sh  to start 4 workers
sbin/stop-all.sh   to stop all workers

./bin/pyspark /Users/chvinay/Desktop/final_distributed.py  to run the file

The key idea is that the entire dataset is split into clusters and the all the outputs and the deltas are calculated in the map part. In the reduce part, the weights are updated using the delta values.


The Dataset is divided into both training and testing set in the program.

Thanks. :)