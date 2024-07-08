dataset=MAG
architecture=cross
cuda=0

python3 main.py --bert_model scibert_scivocab_uncased/ \
                --train_dir ${dataset}_input/ --test_file ${dataset}_input/test.txt \
                --output_dir ${dataset}_output/ --architecture ${architecture} --use_pretrain --cuda ${cuda}
python3 main.py --bert_model scibert_scivocab_uncased/ \
                --train_dir ${dataset}_input/ --test_file ${dataset}_input/test.txt \
                --output_dir ${dataset}_output/ --architecture ${architecture} --use_pretrain --eval --cuda ${cuda}

python3 postprocess.py --dataset ${dataset} --output_dir ${dataset}_output/ --architecture ${architecture} --cuda ${cuda}
python3 patk.py --dataset ${dataset} --output_dir ${dataset}_output/ --architecture ${architecture} --cuda ${cuda}
