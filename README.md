# ECG-NAS
NAS-PxAF


First, you need to download the datasets (PxAF.pkl): [link](https://drive.google.com/file/d/1G5uFIGllmJIk05G1Acp2IItjK159XQhC/view?usp=sharing) 
Next, move it to the datasets folder.<br /> .Then, install requirements by running: <br />
```pyhton
pip install -r requirements.txt
```
To evaluate the signal processing step:<br />

```python
python signalprocess_1ch.py
```
To search for the best CNN architecture for the processed 2D images:<br />
```python
python search_1ch_3class.py
```
To fine-tune the best designed architecture:<br />
```python
python retrain3class.py
```
