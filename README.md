Classifier
===
Introduction
---
This project can train a model for classification task with neural and sparse feature.</br>

How to compile this project
---
**Windows**</br>
* Step 0: Open cmd, and change directory to project directory. </br> Use this command `cd /your/project/path/NNClassifier`. </br>
* Step 1: Create a new directory in NNClassifier. Use this command `mkdir build` </br>
* Step 2: Change your directory. Use this command `cd build`. </br>
* Step 3: Build project. Use this command </br>
`cmake .. -DEIGEN3_INCLUDE_DIR=/your/eigen/path -DN3L_INCLUDE_DIR=/your/LibN3L-2.0/path`. </br>
* Step 4: Then you can double click "NNClassifier.sln" to open this project. </br>
* Step 5: Now you can compile this project by Visual Studio. </br>

**Linux** </br>
* Step 0: Open terminal, and change directory to project directory. </br> Use this command `cd /your/project/path/NNClassifier`. </br>
* Step 1: Create a new directory in NNClassifier. Use this command `mkdir build` </br>
* Step 2: Change your directory. Use this command `cd build`. </br>
* Step 3: Build project. Use this command </br> 
`cmake .. -DEIGEN3_INCLUDE_DIR=/your/eigen/path -DN3L_INCLUDE_DIR=/your/LibN3L-2.0/path`. </br>
* Step 4: Now you can compile this project with this command `make` </br>

The format of corpus
---
Label \t The words of sentences , and split your words with space .
[s]The#sparse#feature [s]split#with [s]space#too


Training and save model
---
**Windows**</br>
If you want to train and save a classification model in windows.Please open project properties and add this argument. </br>
`-train training/corpus -dev development/corpus -test test/corpus -option option/file -l -model  model/file` </br>
**Linux**</br>
And if in Linux, the argument is same as windows </br>

Test your model
---
**Windows**</br>
If you want to test your model in Windows.Please open project properties and add this argument.</br>
`-test test/corpus -output output/file -model model/file` </br>
**Linux** </br>
And in Linux, the command is </br>
`./NNCNNLabeler -test test/corpus -output output/file -model model/file` </br>


NOTE
---
Make sure you have eigen ,LibN3L-2.0, cmake in both Linux and Windows.</br>
And Visual Studio 2013 version (or newer) is required in Windows.</br>

* Eigen:http://eigen.tuxfamily.org/index.php?title=Main_Page </br>
* LibN3L-2.0:https://github.com/zhangmeishan/LibN3L-2.0 </br>
* cmake:https://cmake.org/</br>
* visual studio:https://www.visualstudio.com/zh-hans/free-developer-offers/ </br>
