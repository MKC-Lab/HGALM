dataset=PubMed
architecture=kdcross
cuda=1
kd_freq=7000

python3 main_specter_kd.py --bert_model allenai/specter2_base/ \
                --train_dir ${dataset}_input/ --test_file ${dataset}_input/test.txt \
                --output_dir ${dataset}_output/ --architecture ${architecture} --use_pretrain --cuda ${cuda} --kd_freq ${kd_freq} 
python3 main_specter_kd.py --bert_model allenai/specter2_base/ \
                --train_dir ${dataset}_input/ --test_file ${dataset}_input/test.txt \
                --output_dir ${dataset}_output/ --architecture ${architecture} --use_pretrain --eval --cuda ${cuda} --kd_freq ${kd_freq} 
python3 postprocess.py --dataset ${dataset} --output_dir ${dataset}_output/ --architecture ${architecture} --cuda ${cuda}
python3 patk.py --dataset ${dataset} --output_dir ${dataset}_output/ --architecture ${architecture} --cuda ${cuda}



