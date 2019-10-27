mode=$1
cuda=$2

if [ $mode = 'train' ]; then
    CUDA_VISIBLE_DEVICES="$cuda" python train.py \
        --raw \
        --batch_size 4
elif [ $mode = 'test' ]; then
    CUDA_VISIBLE_DEVICES="$cuda" python generate.py \
        --length=100 \
        --nsamples=4 \
        --prefix='今天' \
        --fast_pattern \
        --save_samples \
        --save_samples_path=./sample/test.txt
else
    echo "[!] wrong mode to run the script"
fi
