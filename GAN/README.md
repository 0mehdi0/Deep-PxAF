# Pulse2Pulse - DeepFake ECG Generator

## Pre-trained DeepFake ECG generator can be found here: - [ECG generator](https://drive.google.com/file/d/1xdoUlkWU7YqtZQmYsdx7CC4CU7tzob8r/view?usp=share_link)

* First download the model and then paste it in `GAN/output/test_exp_1/checkpoints`. 
* Use `Test_models.ipynb` to plot result and save data. 

To train new generator model, we present how to train the Pulse2Pulse generative adversarial network which can generate DeepFake ECGs as discussed in our original paper.

## Trainning Pulse2Pulse GAN to generate DeepFake ECGs. 
## Installation
First, clone the repository.  

```pyhton
git clone https://github.com/0mehdi0/ECG-NAS.git && cd ECG-NAS && cd GAN
```

And then use instraction presented below.

```python
# To train (check the pulse2pulse_train.py file for more information)
python pulse2pulse_train.py train \ # Three options: train, retrain, inference, check
    --exp_name "test_exp_1" \ # A name to the experiment
    --data_dirs 'sample_ecg_data' 'sample_ecg_data' \ # data directories (check sample_ecg_data directory for the format)
    --checkpoint_interval 100 \
    --num_epochs 4000 \
    --start_epoch 0 \
    --bs 10 \
    --lr 0.0001 \
    --b1 0.5 \
    --b2 0.9 \
```
```python
# To retrain the above experiment from a checkpoint at epoch 100
python pulse2pulse_train.py retrain --exp_name "test_exp_1" \
    --num_epochs 3000 \
    --start_epoch 100 \ #start epoch number is 100 assuming that the checkpoint used to restart the trianing is 100
    --checkpoint_path ".\checkpoint\chck_100.pt"
```
More parameters are in the pulse2pulse_train.py file. For example, output directory, tensorboard directory etc. can be changed in this file. 

---

## Database
The original database is downloaded from the PhysioNet PAF prediction challenge through the following link: https://physionet.org/content/afpdb/1.0.0/




## Contributors

* [Mohammad Loni](https://vsehwag.github.io/)
* [Mehdi Asadi](https://ir.linkedin.com/in/mehdi-asadi-966a1b242?trk=)

The code in this repository is based on the following amazing work.

* https://github.com/vlbthambawita/deepfake-ecg


