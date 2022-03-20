# ECG-NAS
ECG-NAS


Firstly you should download dataset_PAF.pkl from [link](https://drive.google.com/file/d/1G5uFIGllmJIk05G1Acp2IItjK159XQhC/view?usp=sharing) 
then move it to datasets folder.<br /> And then install requirements by run : .<br />
```pyhton
pip install -r requirements.txt
```
After that you have to run signalprocess by :<br />

```python
python signalprocess_1ch.py
```
Then you should search for the beast architecture by :<br />
```python
python search_1ch_3class.py
```
In final step retrain the best architecture by :<br />
```python
python retrain3class.py
```
