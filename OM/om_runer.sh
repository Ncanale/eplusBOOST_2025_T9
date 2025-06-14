#!/bin/bash
export JUPYTER_CONFIG_DIR=/tmp/my_jupyter_config
mkdir -p $JUPYTER_CONFIG_DIR


targetdir="/eos/project/c/crystal-ferrara/www/"
currRunJson="single_run_number.json"
sourcefile="check_signal_epBOOST"
sourcefile2="commpare_epBOOST"


sleepsec=10git

while true; do

        # Detect current Run
        filepath="$(ls -tr /eos/project/i/insulab-como/testBeam/TB_2025_06_T9_epBOOST/ASCII_MICHELA/ascii_daq_sshfs/*.dat  | tail -1)
        run_number="${filepath##*/}"
        run_number="${run_number#run}"
        run_number="${run_number%%_*}"

        echo $run_number

        # Update json
        sed -i "s/\"run_number\": *[0-9]\+/\"run_number\": $run_number/" $currRunJsongi

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