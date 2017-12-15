#!/bin/bash
### Merge the stdout et stderr in a single file
#$ -j y

### set array job indices 'min-max:interval'
#$ -t 1-59:1

#$ -l ct=1:00:00
#$ -l vmem=4G
#$ -l fsize=20G
#$ -l irods=1
#$ -l sps=1
#$ -o /sps/km3net/users/tgal/analysis/k40calib/logs
#$ -e /sps/km3net/users/tgal/analysis/k40calib/logs
#$ -P P_km3net

### set the name
#$ -N k40calib
#set -e

JPP_PATH="/sps/km3net/users/tgal/apps/jpp/trunk"
CSV_PATH="/sps/km3net/users/tgal/analysis/k40calib"
DET_ID=14

echo "Task id = $SGE_TASK_ID"

source $KM3NET_THRONG_DIR/src/python/pyenv.sh
source ${JPP_PATH}/setenv.sh ${JPP_PATH}

PWD_=$(pwd)
cd /usr/local/root/v5.34.23
source bin/thisroot.sh
cd "$PWD_"

km3pipe detx ${DET_ID} -o detector.detx

N=$(( $SGE_TASK_ID - 1 ))
for RUN in $(seq $(( $N * 100 + 1 )) $(( $N * 100 + 100))); do
    echo "========================================="
    echo "  Calibrating run $RUN"
    echo "========================================="

    RUN_FILE="${DET_ID}_${RUN}.root"
    CSV_FILE="${DET_ID}_${RUN}.csv"
    MONITOR_FILE="${DET_ID}_${RUN}_monitor.root"

    km3pipe retrieve ${DET_ID} ${RUN} -o ${RUN_FILE}

    [ ! -f ${RUN_FILE} ] && continue

    JMonitorK40 -a detector.detx -f ${RUN_FILE} -o ${MONITOR_FILE}
    k40calib ${MONITOR_FILE} -o ${CSV_FILE}

    cp ${CSV_FILE} ${CSV_PATH}

    rm -f ${RUN_FILE}
    rm -f ${MONITOR_FILE}
done
