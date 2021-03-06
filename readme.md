download original images from: http://vis-www.cs.umass.edu/fddb/originalPics.tar.gz

unzip:
tar -xvzf originalPics.tar.gz images

unzip pretrained model

tar -xvzf training/pretrained/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz

total training sample: 2306/40 => 58 steps per epoch | 25000/58=> 431 epochs
total training sample: 2306/40 => 58 steps per epoch | 14 epochs=> 2900 steps
total eval samples: 539
# ************* training ******************
# training samples: 2306
# batch size: 21
# steps per epoch: 110
# required epochs: 35
# total training steps: 3850

# ************* Eval ******************
# eval samples: 539


# rm -rf models/ssd_resnet50/run-batch-18-steps-2750-lrwsteps-200/*

#for eval we dont want to use gpu

export model_dir=models/ssd_resnet50/run-batch-21-steps-3850
mkdir $model_dir

export pipeline_config_path=training/pretrained/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/pipeline.config
#kill existing eval process
ps aux|grep python\ model_main_tf2.py

export CUDA_VISIBLE_DEVICES=-1

nohup python model_main_tf2.py\
  --model_dir=$model_dir\
  --pipeline_config_path=$pipeline_config_path\
  --checkpoint_dir=$model_dir &

# we definaltey wanna use gpu duh!
export CUDA_VISIBLE_DEVICES=0

nohup python model_main_tf2.py\
  --model_dir=$model_dir\
  --pipeline_config_path=$pipeline_config_path\
  --num_train_steps=3850\
  --checkpoint_every_n=500 &

#start tensorboard via vscode to export 6006 to local

tensorboard --logdir=/datadrive/experiments/face-detection/models/ssd_resnet50

#export model

export checkpoint_dir=/datadrive/experiments/face-detection/models/ssd_resnet50/run-batch-21-steps-2200-lrwsteps-200
export export_dir=/datadrive/experiments/face-detection/models/ssd_resnet50/run-batch-21-steps-2200-lrwsteps-200/exported

python exporter_main_v2.py\
  --input_type image_tensor\
  --pipeline_config_path $pipeline_config_path\
  --trained_checkpoint_dir $checkpoint_dir\
  --output_directory $export_dir