# 百度网盘AI大赛——图像处理挑战赛：手写文字擦除第10名方案

This repository is the implementation of PaddlePaddle-based handwriting removal model.

## File Structure

├── dataloader.py\
├── Dataset\
├── generate_files.py\
├── Log\
├── loss.py\
├── main.py\
├── model.py\
├── README.md\
├── requirement.txt\
├── run.sh\
└── utils.py

## Environment

The environment can be seen in `requirement.txt` file.

## Generate file list

Running the `generate_files.py` to generate file list in dataset directory.

## Training

Once the data is well prepared, you can begin training:
```
python work/main.py\
    --train 1\
    --arch segformer_b2\
    --dataRoot data/dehw_train_dataset\
```

## Generate

If you want to predict the results, run:

```
python train_segformer.py\
    --loadSize 512\
    --train 0\
    --arch segformer_b2\
    --testDataRoot 'Your Path'\
    --modelLog 'Your Path'
```