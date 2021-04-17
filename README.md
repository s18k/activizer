<h1 align="center"> Activizer </h1>


<h2 align="center"> Interface for Active Learning </h2>


</p>

<h3 align="center"> About Active Learning</h3>

<p > Active learning is the process by which your model chooses the training data it will learn the most from, 
    with the idea being that your model will predict better on your test set with less data if it’s encouraged
     to pick the samples it wants to learn from. </p> <br>


<p>Interface for Active Learning supports Image dataset where the user can upload data in Zip or RAR format. It supports 3 classifiers and 7 query strategies.<p>
<h3> Classifiers </h3>

- KNN Classifier

- Random Forest Classifier

- Decision Tree Classifier
<h3> Query By Committee </h3>

- Uncertainty Sampling

- Random Sampling

- Entropy Sampling

- Query By Committee(Uncertainty Sampling)

- Query By Committee(Vote Entropy Sampling)

- Query By Committee(Max Disagreement Sampling)

- Query By Committee(Consensus Entropy Sampling)


This project is implemented with the Active Learning package [modAL](https://github.com/modAL-python/modAL)

<h3 align="center"> How to Run</h3>
<h3> Classifiers </h3>
```
pip install activizer 
```
```
function test() {
  console.log("notice the blank line before this function?");
}
```

- Select the Classifier Algorithm, the Query Strategy and give the number of samples you wish to label. Then select the training / testing dataset in Zip format

<p align="center">
<img src="images/Screenshot (490).png" align="center" width="900"> <br>
</p>

- For each iteration an image will be shown and a dropdown to label that image. Below them will be shown a graph with current accuracy.

<p align="center">
<img src="images/Screenshot (489).png" align="center" width="900"> <br>
</p>

- After all the iterations are over, Final accuracy with graph will be shown

<p align="center">
<img src="images/Screenshot (491).png"> <br>
</p>

- The user can see the images along with the labels provided by the algorithm selected during training. The trained model can be downloaded in pickle format (.pkl file) and can be used for prediction by unpickling it.

- The Interface can be used for prediction by uploading the validation dataset in Zip format. 

<p align="center">
<img src="images/Screenshot (492).png"> <br>
</p>

- The result will be shown in a table consisting of image name and label predicted by the model.

<p align="center">
<img src="images/Screenshot (493).png"> <br>
</p>
