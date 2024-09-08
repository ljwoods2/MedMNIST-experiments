
# nohup ./train_3d.sh &

dsets=("organ" "nodule" "fracture" "adrenal" "vessel" "synapse")

for dset in "${dsets[@]}"; do
    for i in {0..4}; do
        python MedMNIST3D/train_and_eval_pytorch.py \
            --output_root ./output \
            --num_epochs 10 \
            --size 28 \
            --gpu_ids 0 \
            --model_flag resnet50 \
            --run $i \
            --data_flag ${dset}mnist3d > output/${dset}_3d_${i}.log 2>&1
    done
done


# python MedMNIST3D/train_and_eval_pytorch.py \
#     --output_root ./output/test_3d \
#     --num_epochs 10 \
#     --size 28 \
#     --gpu_ids 0 \
#     --model_flag resnet50 \
#     --run 1 \
#     --data_flag organmnist3d > output/organ_3d_2.log 2>&1