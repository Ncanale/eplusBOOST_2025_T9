#!/bin/bash
export JUPYTER_CONFIG_DIR=/tmp/my_jupyter_config
mkdir -p $JUPYTER_CONFIG_DIR

targetdir="/eos/project/c/crystal-ferrara/www/"
currRunJson="single_run_number.json"
sourcefile="check_signal_epBOOST"
sourcefile2="compare_epBOOST"

sleepsec=180

while true; do

    # Detect current Run
    filepath=$(find /eos/project/i/insulab-como/testBeam/TB_2025_06_T9_epBOOST/ASCII_MICHELA/ascii_daq_sshfs/ -name 'run*.dat' -printf '%T@ %p\n' | sort -n | tail -1 | awk '{print $2}')

    run_number="${filepath##*/}"
    run_number="${run_number#run}"
    run_number="${run_number%%_*}"

    echo $run_number

    # Update json
    sed -i "s/\"run_number\": *[0-9]\+/\"run_number\": $run_number/" $currRunJson

    # Execute jupyter
    currFile="run${run_number}.html"
    echo $currFile
    jupyter nbconvert --execute --to html ${sourcefile}.ipynb
    jupyter nbconvert --execute --to html ${sourcefile2}.ipynb
    cp ${sourcefile}.html $targetdir/$currFile
    cp ${sourcefile}.html $targetdir/current.html
    cp ${sourcefile2}.html $targetdir/compare.html

    sleep ${sleepsec}s

done
