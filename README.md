# Hindsight Goal Generation (HGG)

This is a TensorFlow implementation for our paper [Exploration via Hindsight Goal Generation](http://arxiv.org/abs/1906.04279) accepted by NeurIPS 2019.


## Requirements
1. Python 3.6.9
2. MuJoCo == 1.50.1.68
3. TensorFlow >= 1.8.0
4. BeautifulTable == 0.7.0
5. gym < 0.22

## Running Commands

Run the following commands to reproduce our main results shown in section 5.1.

```bash
python train.py --tag='HGG_fetch_push' --env=FetchPush-v1
python train.py --tag='HGG_fetch_pick' --env=FetchPickAndPlace-v1
python train.py --tag='HGG_hand_block' --env=HandManipulateBlock-v0
python train.py --tag='HGG_hand_egg' --env=HandManipulateEgg-v0

python train.py --tag='HGG_fetch_push_with_obstacle' --env=FetchPush-v1 --goal=obstacle
```
