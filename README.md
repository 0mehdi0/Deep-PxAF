# ECG-NAS
ECG-NAS


Firstly you should download dataset_PAF.pkl from [link](https://drive.google.com/file/d/1G5uFIGllmJIk05G1Acp2IItjK159XQhC/view?usp=sharing) then move it to datasets folder.<br />
After that you have to run signalprocess by :<br />
'''
python signalprocess_1ch.py
'''
Then you should run the search for the beast architecture by <br />
'''
python search_1ch_3class.py
'''
In final step retrain the best architecture by <br />
'''
python retrain3class.py
'''
