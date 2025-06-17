#!/bin/bash
export JUPYTER_CONFIG_DIR=/tmp/my_jupyter_config
mkdir -p $JUPYTER_CONFIG_DIR

targetdir="/eos/project/c/crystal-ferrara/www/"
currRunJson="single_run_number.json"
sourcefile="check_signal_epBOOST"
sourcefile2="compare_epBOOST"

sleepsec=180

# Definisci run iniziale
start_run=730223

while true; do
    # Trova il run più recente disponibile
    filepath="$(ls -tr /eos/project/i/insulab-como/testBeam/TB_2025_06_T9_epBOOST/ASCII_MICHELA/ascii_daq_sshfs/*.dat | tail -1)"
    latest_run="${filepath##*/}"
    latest_run="${latest_run#run}"
    latest_run="${latest_run%%_*}"

    echo "Latest available run: $latest_run"

    # Itera da start_run a latest_run
    for ((run_number=start_run; run_number<=latest_run; run_number++)); do
        echo "Processing run $run_number"

        # Aggiorna JSON
        sed -i "s/\"run_number\": *[0-9]\+/\"run_number\": $run_number/" $currRunJson

        # Esegui notebook
        currFile="run${run_number}.html"
        jupyter nbconvert --execute --to html ${sourcefile}.ipynb
        jupyter nbconvert --execute --to html ${sourcefile2}.ipynb
        cp ${sourcefile}.html "$targetdir/$currFile"
        cp ${sourcefile}.html "$targetdir/current.html"
        cp ${sourcefile2}.html "$targetdir/compare.html"

        # Aggiorna run iniziale per evitare ripetizioni
        start_run=$((run_number + 1))
    done

    echo "Sleeping for ${sleepsec}s..."
    sleep ${sleepsec}s
done
