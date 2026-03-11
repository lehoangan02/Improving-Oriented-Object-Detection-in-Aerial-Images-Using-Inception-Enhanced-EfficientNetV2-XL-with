PROJECT_DIR=/Users/lehoangan/Documents/GitHub/BBAV/Improving-Oriented-Object-Detection-in-Aerial-Images-Using-Inception-Enhanced-EfficientNetV2-XL-with
DEVKIT_DIR=$PROJECT_DIR/datasets/DOTA_devkit

cd $PROJECT_DIR
source myenv/bin/activate

export PYTHONPATH=$PROJECT_DIR:$DEVKIT_DIR:${PYTHONPATH:-}

echo "Python path: $(which python)"
echo "PYTHONPATH: $PYTHONPATH"
echo "CUDA available check:"
python -c "import torch; print(torch.cuda.is_available())"

if [ ! -f "$DEVKIT_DIR/polyiou.cpython-*.so" ]; then
    echo "Building polyiou..."
    cd $DEVKIT_DIR
    swig -c++ -python polyiou.i
    python setup.py build_ext --inplace
    cd $PROJECT_DIR
fi

python main.py \
  --data_dir /Users/lehoangan/Documents/GitHub/BBAV/DATA/Mock \
  --num_epoch 50 \
  --batch_size 1 \
  --dataset dota \
  --phase train \
  --conf_thresh 0.1

end_time=$(date +%s)

echo "========================================"
echo "End time: $(date)"
echo "Total runtime: $((end_time - start_time)) seconds"
echo "========================================"
