<p align="center">
<img src="https://zenodo.org/badge/358878863.svg">
</p>
<h1 align="center"> Activizer </h1>


<h2 align="center"> Interface for Active Learning </h2>


</p>

<h3> About Active Learning</h3>

<p > Active learning is the process by which a learning algorithm can query a user
interactively to label data points which are close to the decision boundary formed during classification </p> <br>

<p>The primary objective of this project is to build an Interface for Active learning which
simplifies the process of chosing algorithms,query strategies and labels This eliminates the task of writing programs for each task  
The interface helps annotators of various domains to label data in an interactive manner and also provides features of saving the final model and results
</p> <br>


<p>Interface for Active Learning supports Image dataset where the user can upload data in Zip format. It supports 3 classifiers and 7 query strategies.<p>
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
<p>This project requires python 3.x installed on your machine </p>
<h3> Installation </h3>
The package can be installed usin the command : pip install activizer

<h3>Open Python console and run the following commands</h3>

- from activizer import app
- app.run()
- <img src="images/Screenshot (498).png" align="center" width="900"> <br>

- Copy the url in the browser
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
