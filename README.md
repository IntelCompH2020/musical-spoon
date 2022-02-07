# musical-spoon
A hierarchical topic modeling graphical user interface for training and visualization. The GUI is implemented utilizing PyQT5 and the algorithm behind contemplates two different alternatives for the construction of hierarchical topic models oriented towards the use of domain experts:
- HTM-WS: Hierarchical Topic Model with word selection 
- HTM-DS Hierarchical Topic Model with document selection 
which rely on LDA, but rather than modifying its underlying generative process, construct the hierarchy structure through nested executions of LDA-Mallet, under distinct conditions.

To run the application:
```
python3 main.py
```

After a couple of seconds, the GUI’s starting window is shown, through which the user can select the project folder to save the hierarchical topic models, the training corpus, and the directory in which he has Mallet locally located. These three constitute the required input parameters of the application.

![](https://github.com/Nemesis1303/MusicalSpoonV3/blob/main/gui/Images/mainWindow.png?raw=true)

After selecting the required parameters and clicking the START button, the user is redirected to the main window of the application, which is composed of the following subwindows:

Configuration view         |  Select / train root model view
:----------------------------------:|:----------------------------------:
![](https://github.com/Nemesis1303/MusicalSpoonV3/blob/main/gui/Images/configuration.JPG?raw=true) |  ![](https://github.com/Nemesis1303/MusicalSpoonV3/blob/main/gui/Images/train_select.png?raw=true)

Train / edit submodels view         |  See topic's description view
:----------------------------------:|:----------------------------------:
![](https://github.com/Nemesis1303/MusicalSpoonV3/blob/main/gui/Images/edit_model.JPG?raw=true) |  ![](https://github.com/Nemesis1303/MusicalSpoonV3/blob/main/gui/Images/see_topic_desc.JPG?raw=true)

