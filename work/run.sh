# generate train and test file list
# python generate_files.py

# train
# python train_segformer.py\
#     --train 1\
#     --arch segformer_b2\
#     --dataRoot Dataset/Baidu_Dewords/dehw_train_dataset\

# get the result
python train_segformer.py\
    --loadSize 512\
    --train 0\
    --arch segformer_b2\
    --testDataRoot Dataset/Baidu_Dewords/dehw_testB_dataset\
    --modelLog Log/segformer_b2/01202247.pdparams