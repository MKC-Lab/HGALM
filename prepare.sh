dataset=MAG
metagraph=PRP

green=`tput setaf 2`
reset=`tput sgr0`

echo "${green}=====Step 1: Preparing Testing Data=====${reset}"
python3 prepare_test.py --dataset ${dataset}

echo "${green}=====Step 2: Generating Training Data=====${reset}"
python3 prepare_train.py --dataset ${dataset} --metagraph ${metagraph}
#
head -200000 ${dataset}_input/dataset.txt > ${dataset}_input/train.txt
sed -n '200001,220000p' ${dataset}_input/dataset.txt | awk 'NR % 4 == 1 || NR % 4 == 0' > ${dataset}_input/dev.txt

