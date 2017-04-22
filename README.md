Classifier
===
Introduction
---
This program can train a model for classification task with neural and sparse feature.</br>

How to compile this project in windows
---
* Step 0: Open cmd, and change directory to project directory. </br> Use this command `cd /your/project/path/NNClassifier`. </br>
* Step 1: Create a new directory in NNClassifier. Use this command `mkdir build` </br>
* Step 2: Change your directory. Use this command `cd build`. </br>
* Step 3: Build project. Use this command </br> `cmake .. -DEIGEN3_INCLUDE_DIR=/your/eigen/path -DN3L_INCLUDE_DIR=/your/LibN3L-2.0/path`. </br>
* Step 4: Then you can double click "NNClassifier.sln" to open this project. </br>
* Step 5: Now you can compile this project by Visual Studio. </br>
* Step 6: If you want to run this project.Please open project properties and add this argument. </br>
`-train /your/training/corpus -dev /your/development/corpus -test /your/test/corpus -option /your/option/file -l` </br>

How to compile this project in Linux
---
The steps are same as in windows.</br>
* Step 0: Open terminal, and change directory to project directory. </br> Use this command `cd /your/project/path/NNClassifier`. </br>
* Step 1: Create a new directory in NNClassifier. Use this command `mkdir build` </br>
* Step 2: Change your directory. Use this command `cd build`. </br>
* Step 3: Build project. Use this command </br> `cmake .. -DEIGEN3_INCLUDE_DIR=/your/eigen/path -DN3L_INCLUDE_DIR=/your/LibN3L-2.0/path`. </br>
* Step 4: Now you can compile this project with this command `make` </br>


Training and save model
---
If you want to train and save a classification model in windows.Please open project properties and add this argument. </br>
`-train /your/training/corpus -dev /your/development/corpus -test /your/test/corpus -option /your/option/file -l -model /your/model/file` </br>

And if in Linux, just use following command. </br>
`./NNCNNLabeler -train /your/training/corpus -dev /your/development/corpus -test /your/test/corpus -option /your/option/file -l -model /your/model/file` </br>

Test your model
---
If you want to test your model in windows.Please open project properties and add this argument.</br>
`-test /your/test/corpus -output /test/output/file -model /your/model/file` </br>
And in Linux, the command is `./NNCNNLabeler -test /your/test/corpus -output /test/output/file -model /your/model/file` </br>


NOTE
---
Make sure you have eigen ,LibN3L-2.0, cmake in both Linux and windows</br>
* Eigen:http://eigen.tuxfamily.org/index.php?title=Main_Page </br>
* LibN3L-2.0:https://github.com/zhangmeishan/LibN3L-2.0 </br>
* cmake:https://cmake.org/</br>
And visual studio 2013 version (or newer) is required in windows.
* visual studio:https://www.visualstudio.com/zh-hans/free-developer-offers/
