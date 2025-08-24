# UDC Mamba
Thank you MambaIR
This is our publication to CVIP 2024.

## Dataset
Download data from kaggle
```
!kaggle datasets download -d aniruthsundararajan/udc-cvpr
```
and place it inside datasets folder

Then run mat_to_data python file
```
!python mat_to_data.py
```
to convert the mat files to images

## Running

Training
```
!python -m resmambair.train_resmamba
```

Testing
```
!python -m resmambair.testing.tester.py
```
