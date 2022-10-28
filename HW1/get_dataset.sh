# Download dataset
gdown 1LMIaOY8NSKWmGtbvTsjXcHVaYKnNN_9u -O hw1_data.zip

# Unzip the downloaded zip file
mkdir hw1_data
unzip ./hw1_data.zip -d hw1_data
gdown 1HZxEPVrKkTwhod6BjwKUn-Ync0z0mKzU -O checkpoints/inception_10_10_best_epoch20.pt 

gdown 13h_9bz3epcdLayl_T_U8rH-GnUgezChB -O checkpoints/epoch_32.pt 