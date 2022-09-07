# A Hybrid Framework with Transfer Learning Technique for Multiclass Classification of Alzheimer Diseases Using MRI Images

This repository hosts the code source guidelines for reproducible experiments on A Hybrid Framework with Transfer Learning Technique for Multiclass Classification of
Alzheimer Diseases Using MRI Images. It allows training for the 1)- VGG16 model with fine tuning, 2)- with Principal Component Analysis (PCA), 3)- with a transfer learning through VGG16, 4)- with some additional layer and frozen blocks in the VGG16 network.  Any updates regarding this manuscript will be mentioned on this repositpry in the future.

This is the new method which can handle 5-way classification of NC, EMCI, MCI, LMCI, and AD.

Dataset:

The datasets used in this work are obtained from the ADNI database: 
http://adni.loni.usc.edu

These images are processes with the help of SPM12:
https://www.fil.ion.ucl.ac.uk/spm/software/spm12/
The results for training and loss in terms of accuracy, true positives, true negatives, false negative, false positive and resuting confusion matrices are shown. 
<img src="https://github.com/imrizvankhan/Hybrid5way/blob/main/Images/processing.jpg"  height = 400 width = 400>
<img src="https://github.com/imrizvankhan/Hybrid5way/blob/main/Images/networks.jpg" height = 400 width = 600>
<img src="https://github.com/imrizvankhan/Hybrid5way/blob/main/Images/vgg%2Bpca.jpg"  height = 300 width = 400>
<img src="https://github.com/imrizvankhan/Hybrid5way/blob/main/Images/cf.jpg"  height = 300 width = 600>
<img src="https://github.com/imrizvankhan/Hybrid5way/blob/main/Images/cf%2Bpca.jpg"  height = 600 width =800>
<img src="https://github.com/imrizvankhan/Hybrid5way/blob/main/Images/adl%2Bfl%2Btable-other.jpg"  height = 800 width =800>
<img src="https://github.com/imrizvankhan/Hybrid5way/blob/main/Images/pixels.jpg"  height = 350 width =450>
<img src="https://github.com/imrizvankhan/Hybrid5way/blob/main/Images/table%2Bpca%2Brf.jpg" height = 150 width = 700>
<img src="https://github.com/imrizvankhan/Hybrid5way/blob/main/Images/adnidata.jpg" height = 150 width = 400>

