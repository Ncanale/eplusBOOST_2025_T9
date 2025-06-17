#!/bin/bash
export JUPYTER_CONFIG_DIR=/tmp/my_jupyter_config
mkdir -p "$JUPYTER_CONFIG_DIR"

targetdir="/eos/project/c/crystal-ferrara/www/"
currRunJson="single_run_number.json"
sourcefile="check_signal_epBOOST"

sleepsec=180
start_run=730223

while true; do
    # Trova il run più recente disponibile
    filepath=$(find /eos/project/i/insulab-como/testBeam/TB_2025_06_T9_epBOOST/ASCII_MICHELA/ascii_daq_sshfs/ -name 'run*.dat' -printf '%T@ %p\n' | sort -n | tail -1 | awk '{print $2}')
    
    if [[ -z "$filepath" ]]; then
        echo "Nessun file trovato. Aspetto 5 secondi..."
        sleep 5
        continue
    fi

    latest_run="${filepath##*/}"
    latest_run="${latest_run#run}"
    latest_run="${latest_run%%_*}"

    echo "Latest available run: $latest_run"

    # Itera da start_run a latest_run
    for ((run_number=start_run; run_number<=latest_run; run_number++)); do
        echo "Processing run $run_number"

        # Verifica che il run sia numerico
        if [[ "$run_number" =~ ^[0-9]+$ ]]; then
            echo "{ \"run_number\": $run_number }" > "tmp_$currRunJson" && mv "tmp_$currRunJson" "$currRunJson"
        else
            echo "Errore: run_number non valido: '$run_number'" >&2
            continue
        fi

        # Esegui notebook
        currFile="run${run_number}.html"
        jupyter nbconvert --execute --to html "${sourcefile}.ipynb"

        # Copia i risultati
        cp "${sourcefile}.html" "$targetdir/$currFile"
        cp "${sourcefile}.html" "$targetdir/current.html"
    done

    # Aggiorna il run iniziale per il prossimo ciclo
    start_run=$((latest_run + 1))

    echo "Finished up to run $latest_run"
    echo "Sleeping for ${sleepsec}s..."
    sleep "${sleepsec}"
done
