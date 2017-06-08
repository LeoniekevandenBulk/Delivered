setlocal enableDelayedExpansion

set MYDIR = C:\Users\WoutervanderWeel\Documents\AI\Master AI\Intelligent Systems in Medical Imaging\Final project\Delivered\data\Training_Batch

FOR /F %%x IN ('dir /B/D %MYDIR%') DO med2image.py -i vol.nii -d out -o volume.jpg -s -1