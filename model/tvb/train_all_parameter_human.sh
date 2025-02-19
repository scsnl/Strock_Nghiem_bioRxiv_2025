for snr in $(seq 10.00 -10.00 -10.00)
do
    for region in PCC PCC_Pcun PCC_Pcun_Ang
    do
        for qiasd in $(seq 4.5 0.1 5.0)
        do
            python model/tvb/train_simple.py --region $region --noise 1e-4 --qiasd $qiasd --qint 5.0 --snr $snr # --redo
        done
    done
done