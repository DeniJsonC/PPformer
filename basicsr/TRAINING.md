## Training
#### Training on LoL/FiveK/SID datasets
```
1. Download LOL/FiveK/SID training and testing data, see [datasets/README.md](datasets/README.md).  
```
```
2. cd PPformer
./train.sh Enhancement/Options/PPformer_*.yml 
##########
eg. ./train.sh Enhancement/Options/PPformer_LOL.yml
##########
*You may change the 'batch_size' in the config depends on your GPU mode and numbers.*
```