for snr in $(seq 10.00 -10.00 -10.00)
do
    for a in $(seq 1 1 10)
    do
        for delta in $(seq 0.50 -0.10 0.00)
        do
            regex="epoch([0-9]+).ckpt"
            for f in $DATA_PATH/rnn_exc/model/simple_n100_a${a}_delta${delta}_snr${snr}/*
            do
                n="$(basename $f)"
                if [[ $n =~ $regex ]]
                then
                    python model/rnn_exc/test_simple_baseline.py --n 100 --a $a --delta $delta --snr $snr --epochs ${BASH_REMATCH[1]}
                fi
            done
        done
    done
done