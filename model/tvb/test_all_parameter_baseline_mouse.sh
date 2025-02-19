for snr in $(seq 10.00 -10.00 -10.00)
do
    for region in RSC RSC_Cg RSC_Cg_PrL
    do
        for qiasd in $(seq 4.5 0.1 5.0)
        do
            regex="epoch([0-9]+).ckpt"
            for f in $DATA_PATH/tvb/model/tvb_${region}_asdQi_${qiasd}_ntQi_5.0_noise1.0e-04_snr${snr}/*
            do
                n="$(basename $f)"
                if [[ $n =~ $regex ]]
                then
                    python model/tvb/test_simple_baseline.py --region $region --noise 1e-4 --qiasd $qiasd --qint 5.0 --snr $snr --epochs ${BASH_REMATCH[1]} --redo
                fi
            done
        done
    done
done