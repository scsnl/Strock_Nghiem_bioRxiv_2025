for snr in $(seq 10.00 -10.00 -10.00)
do
    for a in $(seq 1 1 10)
    do
        for delta in $(seq 0.50 -0.10 0.00)
        do
            python model/rnn_exc/train_simple.py --n 100 --a $a --delta $delta --snr $snr
        done
    done
done