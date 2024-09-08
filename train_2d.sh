
# nohup ./train_2d.sh &

dsets=("chest" "path" "derma" "oct" "pneumonia" "retina" "breast" "blood" "tissue" "organa" "organc" "organs")
dsets=("pneumonia" "retina" "breast" "blood" "tissue" "organa" "organc" "organs")

for dset in "${dsets[@]}"; do
    for i in {0..4}; do
        python MedMNIST2D/train_and_eval_pytorch.py \
            --output_root ./output \
            --num_epochs 10 \
            --size 28 \
            --gpu_ids 0 \
            --model_flag resnet50 \
            --run $i \
            --data_flag "${dset}mnist" > "output/${dset}_2d_${i}.log" 2>&1
    done
done

        python MedMNIST2D/train_and_eval_pytorch.py \
            --output_root ./output \
            --num_epochs 10 \
            --size 28 \
            --gpu_ids 0 \
            --model_flag resnet50 \
            --run $i \
            --data_flag "${dset}mnist" > "output/${dset}_2d_${i}.log" 2>&1