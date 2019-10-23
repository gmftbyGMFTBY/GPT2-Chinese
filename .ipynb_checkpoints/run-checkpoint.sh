mode=$1

if [ $mode = 'train' ]; then
    CUDA_VISIBLE_DEVICES="4,7" python train.py \
        --raw \
        --batch_size 4
elif [ $mode = 'test' ]; then
    CUDA_VISIBLE_DEVICES="4,7" python ./generate.py \
        --length=50 \
        --nsamples=4 \
        --prefix=今天 \
        --fast_pattern \
        --save_samples \
        --save_samples_path=./sample/test.txt
else
    echo "[!] wrong mode to run the script"
fi
