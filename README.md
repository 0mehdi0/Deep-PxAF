# ECG-NAS
NAS-PxAF


First, you need to download the datasets (PxAF.pkl): [link](https://drive.google.com/file/d/1G5uFIGllmJIk05G1Acp2IItjK159XQhC/view?usp=sharing) , (Selected_GAN.pt): [link](https://drive.google.com/file/d/1j1wuQjeUR02wKyAllhOwo_dE0MjF0Oop/view?usp=sharing) , (Selected_PAF.csv): [link](https://drive.google.com/file/d/1vAn5PieATTsYW7TCHYrU38zWtpIPc8R9/view?usp=sharing) and (GAN_Data.pt): [link](https://drive.google.com/file/d/1-Tz5bikmHLaK8ds2r8D1Uzlw89XMD-pW/view?usp=sharing)
Next, move them to the datasets folder.<br /> .Then, install requirements by running: <br />
```pyhton
pip install -r requirements.txt
```
To evaluate the signal processing step:<br />

```python
python signalprocess_1ch.py
```
To search for the best CNN architecture for the processed 2D images with randomseed 100 and synthetic data:<br />
```python
python search_1ch_3class.py --seed 100 --GAN_flag" 1
```
To fine-tune the best designed architecture:<br />
```python
python retrain3class.py --seed 100 --GAN_flag" 1
```
