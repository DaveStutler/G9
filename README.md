### ECS 171: Machine Learning (Group 9)

# Analysis on the Particle Collider Dataset


# Introduction
We have two datasets obtained from different particle colliders, `Output_File_2023_02_15.root` which includes data about the linear collisions and `yieldHistos_7p7GeV_Pion_2022_08_31.root` which contains data about circular. The project idea is to analyze the datasets on the shape of explosions from a particle collider in order to determine what aspects of the particles used and conditions are common between collider and detector types. These findings would be significant because they can help researchers determine the collider type they should use when looking to collide heavy ion particles.

Two problems we intend to solve using supervised learning techniques include:
1. classifying linear and spherical collisions 
    1. using SVM and binary classification models
1. predicting count of decayed particles based on 3 features (stopping power, position, and collision type)
    1. using a regression model

For unsupervised learning, we will experiment with generators where our goal will be to generate synthetic data that
matches our real data distribution and is indistinguishable from our real data. The application for this would be to provide more data samples for physicists who do not have access to expensive real-world particle collider data to learn from and analyze.

# Methods
## Data Exploration Results
### Description of Data
We compiled the relevant columns from both the linear collisions dataset and circular collisions dataset into a
mutable dataframe. Although this dataframe appears to only have 2 rows, contained within those rows contain are
1000x1000 matrices representing data from over 91,000 collisions. This way of packaging the data allows physicists
to work with larger quantities of data then would be compatible with a csv file. The original .root files had over
200 different features, but the the features that we choose to include in our dataframe and thus to analyse were
dEdx_PionPlus_Isolated;1, dEdx_PionMinus_Isolated;1, dEdx_KaonPlus_Isolated;1, dEdx_KaonMinus_Isolated;1,
dEdx_ProtonPlus_Isolated;1, dEdx_ProtonMinus_Isolated;1, dEdx_DeuteronPlus_Isolated;1, dEdx_DeuteronMinus_Isolated
1, dEdx_TritonPlus_Isolated;1, dEdx_HelionPlus_Isolated;1, dEdx_HelionMinus_Isolated
1, dEdx_AlphaPlus_Isolated;1, and dEdx_AlphaMinus_Isolated;1. We decided to use these datapoints because each
feature isolates stopping power data per composite particle and these features coincide in both files.

### Descriptions of Graphs 
The y axis on our plots represents stopping power per particle. The x axis represents mass energy of a particle. We
decided to leave the y axis inverted when graphing in order to make the y axis stand out more from the x axis to
help with our analysis. Some interesting trends we noticed on these graphs were a small variance along the axis for
pions and deuteron and proton minus distributions and logarithmic correlations with other distributions.

Plot of counts collected for ProtonMinus at various stopping powers and positions in linear collisions

<img src="ProtonMinus-Linear.png" width="450" />

Plot of counts collected for ProtonMinus at various stopping powers and positions in circular collisions

<img src="ProtonMinus-Circular.png" width="450" />



## Data Preprocessing (steps)
1. Analyze original linear and circular `.root` files to determine which features coincide in both files
2. Convert 13 1000x1000 matrices into tabular data form
    1. One-hot encode our `linear` and `circular` features
    2. Iterate through each matrix in order to find all indexes where at least one feature has data
    3. Create dataframes for each matrix consisting of 'position' 'stopping power' and the count of that feature such that all dataframes contain the same values for 'position' and 'stopping power'
    4. Concatenate all individial feature dataframes together based on 'position' and 'stopping power' index. We call this master dataframe 'holygrail'
    5. Download 'holygrail' as a csv
3. Normalizing and splitting 
    1. Split `train`, `validation` and `test` sets with a split ratio of 70:10:20, respectively
    2. Normalize `train`, `validation` and `test` sets separately
    3. Within the normalizer function: normalize `circular` and `linear` elements separately and ensure and equal number of each is in each training, testing, and val set
    4. Within the normalizer function: Threshold very small values and delete rows with no significant data


## Model 1: Binary Classification - Linear vs. Circular Collider Types
### Description
In hopes of predicting the type of collider, either circular or linear, we use a 10-layer ANN that takes in the feature counts, positions, and stopping powers and classifies the data as one of our one-hot-encodered collider types
* Architecture: we use Relu activation functions for efficent runtime. We also use Sigmoid activation function in our output layer to classify the 2 groups, and use the binary_crossentropy function to update our model weights and bias. We split our dataset into 90:10 of propotion, with linear and circular columns as our target and every other columns as our features.
* We incorportated EarlyStopping and ModelCheckpoint to ensure we were getting the best weights possible and not spending more time training than the model needed.
* We also incorporated validation data into our training in order to find the best models for ModelCheckpoint.
* The validation, training, and testing data were all normalized seprately and there was no possibility of data leakage
* There was also an equal number of both classes in all sets in order to avoid bais
* We tested the model on unseen values and report the accurary_score, classification report, and a confusion matrix as well as the epoch at which we found our best model based on ModelCheckpoint and EarlyStopping
* ### K-folds
* In order to test the models consistancy accross unseen data, we tested it on 5 folds in which a random section of the training data was sectioned off as validation data and normalized seperately.
* The goal was to try to get a high mean and low sd to indicate that our model has low variation and is consistantly accurate at predicting unseen data

## Model 2: Binary Classification Using SVM - Linear vs. Circular Collisions
### Description
To further classifying the Linear vs Circular Collisions, we attempt at using a SVM model to illustrate the boundary between 2 features. We hypothesize that the `position` and the `stopping power` greatly influence the classification problem. To test this hypothesis, we pick 2 features: `position`, and `stopping power`. These variables has the correlation 0.57, and 0.059 respectively.  
* Architecture: In our svm model, we implement with Radial Basis Function Kernel (RBF) function to as our kernel.
* Our dataset were splited into training and testing 80:20. 
* The validation, training, testing data were all normalized.
* We check the accurary_score for performance, as well as illustrating classification report and the confusion matrix for model performance.
* We tested our model on both the training data and testing data, reporting classification report and confusion matrix for both data.

## Model 3: Regression - Predicting the Count of Each Decayed Particle
### Description
To increase the complexity of our project, we decided to make an attempt at predicting the count of decayed
particles detected at a certain stopping power and position under either a circular or linear collision. To do
this, we use build a regression ANN model. 
* Architecture: In our 1st hidden layer we use a linear activation function than then feeds its outputs into the
2nd hidden layer which uses a LeakyRelu activation functions (the 3rd and 4th hidden
layers follow this same architecture). The output layer makes use of a linear activation function. To update our
model weights and bias, we  use the MSE loss function.
* Once again, we used EarlyStopping and ModelCheckpoint to ensure we were getting the best weights possible and not spending more time training than the model needed.
* We also incorporated validation data into our training in order to find the best models for ModelCheckpoint.
* The validation, training, and testing data were all normalized seprately and there was no possibility of data leakage
* We tested our model on unseen values and reported the MSE as well as the epoch at which we found our best model based on ModelCheckpoint and EarlyStopping
* ### K-folds
* In order to test the models consistancy accross unseen data, we tested it on 5 folds in which a random section of the training data was sectioned off as validation data and normalized seperately.
* The goal was to try to get a high mean and low sd to indicate that our model has low variation and is consistantly accurate at predicting unseen data

## Model 4: Data Generator using encoding and decoding
### Description
Running experiments in nuclear colliders is expensive, so in order to help scientist who want to generate synthetic data to model this procress, we tried to make a generator using an encoding and decoding VAE model.
* Preprocessing: to decrease the complexity of our data, we decided to only look at one feature, namely 'dEdx_PionPlus_Isolated;1'.
* we created a new df that contained only position, stopping power, and this element at indexes where its counts were non zero and we normalized all columns 
* Architecture: Most of the architecture was built through hours of trial an error. The final model includes 6 encoder layers and 5 decoder layers with a latent space of 10 dimensions.
* 'MSE' was used to measure loss.
### Testing 
* Because this is an unsupervized learning method, testing was not as straight forward as usual.
* We used visual means of assessing the fit by plotting our generated sample in a 3d scattor plot with the original preprocessed data
* We also used statistical tests to see how likely it was that these two samples came from the same population
* Finally, we implemented DBSCAN clustering on a dataset that contained the original data as well as our generated samples to see if the generated samples would be classified as either belonging to another cluster, or anomalies.

# Results
## Data Preprocessing 
The results of our data preprocessing phase include:

1. Extraction of 13 features shared from our circular and spherical `.root` files
1. Conversion of 13 1000x1000 matrices into a singular tabular pandas dataframe
    1. columns consisting of position, stopping power, and other linear and circular features
1. 'holygrail'.csv file which contains all relivant data points as a dataframe

## Model 1: Binary Classification - Linear vs. Circular Collider types

The testing accuracy was .96. For k-folds, there was a mean accuracy of .95 with a sd of .019

### Fitting of the data
The training accuracy was .98 and the testing accuracy was only very slightly lower. Also, we determined the best model based on the validation scores to prevent overfitting. Based on that, I would say overfitting was minimal

Confusion Matrix

<img src="/PictureFolder/confusionMatrix_model1.png" alt= "Confusion Matrix" width="450" />

Training and Testing graph

<img src="/PictureFolder/model1_training&testing.png" alt= "Training and Testing graph" width="450" />

K-fold Average Accuracy

<img src="/PictureFolder/model1_AverageAccuracy.png" alt= "K-fold Average Accuracy" width="450" />

K-fold Accuracy Scores

<img src="/PictureFolder/model1_score.png" alt= "K-fold Accuracy Scores" width="450" />


## Model 2: Binary Classification Using SVM - Linear vs. Circular Collisions
The accuracy for test .62. The accuracy for train was .59. We do not think a svm model is able to represent this data as well as neural nets 

Confusion Matrix for XTest

<img src="/PictureFolder/confusionMatrix_Model2_XTest.png" alt= "Confusion Matrix for XTest" width="450" />

Confusion Matrix for XTest

<img src="/PictureFolder/confusionMatrix_Model2_XTrain.png" alt= "Confusion Matrix for XTest" width="450" />


## Model 3: Regression - Predicting the Count of Each Decayed Particle
The testing MSE was 0.006 and the training MSE was 0.004. When we ran K-folds validation on it to test its consistancy, we got a mean loss of 0.01 with a standard deviation of 0.0006
### Fitting of the data 
Based on the similar testing and training MSEs and the very low variation between K-Folds loss, it is unlikely that our model is significantly overfitting the data

K-Fold classification result

<img src="/PictureFolder/model3_kfoldResult.png" alt= "K-Fold classification result" width="450" />

Mean Square Error

<img src="/PictureFolder/model3_MSE.png" alt= "Mean Square Error" width="450" />

## Model 4: 
Visially, the sample generated seems to match the dataset fairly well

When preforming statistical tests to determine whether of not the 2 datapoints were likely to come from the same population distribution, we found that 2 of the columns were not able to reject the HO that the 2 datapoints came from the same population, however, the count column had a p-value low enough to reject the null hypothesis
based on our small sample size of just 500 though comparied to the dataset which contains >100,000 rows, we think the fact that it visiually matches the data and got a high p-value for 2 of 3 columns is pretty good
When preforming DBSCAN clustering with the generated samples and original data, both the samples and the data were put into the same cluster and there were no anomalies detected, which means the data we generated ewas similar enough to not be classified as an outlier


Generated Data (Blue) with Sample Data (Red)

<img src="/PictureFolder/model4_2ndGraph.png" alt= "Generated Data (Blue) with Sample Data (Red)" width="450" />

Generated Data (Blue) with Sample Data (Red) tranformed with np.log

<img src="/PictureFolder/model4_3rdgraph.png" alt= "Generated Data (Blue) with Sample Data (Red) tranformed with np.log" width="450" />

P-values

<img src="/PictureFolder/Model4_pvalues.png" alt="p-values" width="450" />

DB scan clustering 

<img src="/PictureFolder/model4_4thGraph.png" alt= "fill in" width="450" />

### Results summary: Overall, our two neural nets preformed the best with very impressive accuracy and loss scores. Our SVM model got very poor scores. We think that our data is too complex for this type of model. The generator was hard to assess in terms of accuracy, however based on how it looks in comparison to the sampled data, it seems to be generating reasonable samples 

# Discussion
## Data Exploration
The goal in our Data Exploration phase was 3-fold:

1. figure out a way to access the data in our two `.root` files
1. figure out what features were common in both files
1. discover meaning in the data and their respective graphs

After figuring out how to extract meaningful data from our root files and the interpretation of this data, we ended up with 16 10,000 by 10,000 matrices such that:

1. rows correspond to position
1. columns correspond to change in energy over distance (equates to Stopping power)
1. values correspond to the number of decayed particles that were detected at that Stopping Power and Position

### Challenges and Shortcomings
The biggest challenge we came across with this Data Exploration milestone was figuring out how to extract the data
from the two `.root` files we are working with. Initially, the plan was to convert these `.root` files into `.csv`
files outside of Colab by first converting them into a dictionary and then a pandas dataframe and finally saving as
a `.csv`, and then import the `.csv` files into Colab and work with those. However, for some reason, the `.csv`
file conversion process was causing us to loose important data, so we decided to directly work with the `.root`
files in Colab. We were able to copy and paste our old code which converted these `.root` files into a dictionary
and then a pandas dataframe. Figuring out how to work with `.root` files without installing root in general was
another challenge.

## Preprocessing
While our preprocessing plan initially involved making our matrices dense (i.e. "zooming into meaningful
data") for the purpose of making our neural net run faster, we instead decided to go with an alternative approach.
From our Data Exploration milestone, we were able to extract 13 1000x1000 matrices.
 
After careful consideration, we believed it to be more beneficial for us to convert these matrices into tabular
data form as it is something we are all more familiar and comfortable working with. 

Before creating the `holygrail.csv` files, we first one-hot encoded
our `linear` and `circular` features. After doing so, we  then actually began the process of creating the dataframe
columns (i.e. converting the matrices), which consisted of the positions, stopping power, the detection number for
different particle types, and the one hot encoded columns representing circular and linear collider types. This
process involved the challenge of trying to extract only the most useful parts of each matrix so that the size did
not get too overwhelmingly large, while still keeping the rows and columns between matrices exactly the same
indexes so that we could merge them into a dataframe smoothly based on the position and stopping power columns. 

After each matrix was converted into a tabular dataform, we then merged each of them into a single tabular
dataframe, after which we normalized all features before beginning training and testing of our model. We normalized
the circular data and the linear data seperately because the circular data on average had many more detections then
linear, and we did not want this to bias our model. The circular data came from a differenrt root file where its
likely more events were run, causing the circular data to have higher counts. A model that trained on this data
would not have generalized to the real world, so we scaled all columns seperately for circular and linear collider
types. 

After separately normalizing our circular and linear collider types, we then split and normalized our train,
validation and test sets separately to avoid any potential data leakage. The reason for including both X and y
values in our normalization function is to ensure that our data is normalized separately based on the different
collider shapes (to account for differences in the sample sizes for linear and circular collisions). It's worth
mentioning that before normalizing our training, testing, and validation data sets, we did make sure to have the
same amount of linear and circular instances to avoid any bias in our model. 
 
### Challenges and Shortcomings
The biggest challenge was figuring how to convert 13 1000x1000 matrices into a nice readable dataframe. We had to
decide how we wanted to represent this data in our dataframe, and once we decided that we wanted to merge all the
particles from different matrices into the same rows based on stopping power and position, it became a big issue to
try to make sure that we are extracting the data at the exact same places in the matrices for all features, and
that the places we are extracting data would contain the least amount of useless data (all zeros would be
considered useless). 

In retrospect, it would have been easier to do this based on looking at areas across all graphs where most data
seemed to be concentrated. However, the approach we choose involved interating through matrices to find the maximum
and minimum row and column where there is meaningful data, using those values to find the global minimum and
maximum, evaluating at those points and creating a dataframe for each feature, recursively merging all dataframes
into one, and then normalizing and removing noise and all rows of all 0s.

## Model 1: Binary Classification - Linear vs. Circular Collisions
Seeing as our data consists of two different types of particle collisions, attempting to create a model that
classified between circular and spherical collisions was a no-brainer. As mentioned in the [description of this
model](#model-1-binary-classification---linear-vs-circular-collisions), we created a 10-layer ANN. Using 12 features
in our dataframe (excluding `circular` and `spherical` labels), we used this model to predict whether the
datapoints were consistent with that of a linear collision or a spherical collision. 


### Challenges and Shortcomings
In the first draft of the model, we were originally getting very high testing accuracy after just one epoch. We were very suspicious and spent a long time trying to figure out if our model was amazing, or incredibly bias. We decided to normalize the training, testing, and validation sets seprately in order to prevent any leakage of distributions. 
Another big thing we noticed is that the preprocessed data contained a lot more instances of circular collider types than linear. This would have caused bais because the model would have learnt that predicting 'circular' more often than 'linear' would result in higher accuracy, however, that wouldn't have generalized well to real life data from other colliders. In order to combat this, we included a sampler in our normalizer function to ensure that all testing, training, and validation sets had the samenumber of linear and circular collision types in each. 
After doing this, our accuracy decreased, but it is still quite high. However, we are now confident that our model does not include bias. We ran paired t-tests on the columns across both collider types. 
Our p-value's from these tests showed that there were several features with a p-value of zero (not exactly zero but rounded to zero in colab) meaning there
are very strong statistical differences between our linear and circular datapoints. These differences would help our model quickly learn our data. With this knowledge, we were able to conclude that there was no issue with our preprocessing or model. Our classification problem was simply easy. Since our data is significantly different for each collision type, our model was able to hone in on these trends and them to make accurate predictions

## Model 2: Binary Classification Using SVM - Linear vs. Circular Collisions
### Overview about SVMs
The purpose of the data is to predict if the particle collisions is either circular or linear. In our model 2, we are using SVM to classify if they are any of these two. We begin using Label Encoding to manipulate the dataframe and make it suitable for passing it as SVMs. SVM Model is one of the best model for predicting binary classification, and the target variables allow it to be modeled as a Binary Classification, choosing this model we enhancing our understanding in binary classification problem, and provide a different perspective on how predicting works in SVM.

### Details about the SVMs
In the SVM model, we are using a kernel called RBF, it is a non-linear function that use 4 independent variables, `position`, `stopping power`, `dEdx_KaonPlus_Isolated;1` and `dEdx_PionPlus_Isolated;1`, and it helps a prediction of `circular` and `linear` collision.

### Challenges about SVMs
It is pretty hard to make it accurate, for instance, originally, we are only using `dEdx_KaonPlus_Isolated;1` and `dEdx_PionPlus_Isolated;1` to predict the model, the result is not as what we want based on a lower accuracy, precision and recall. With our strong effort on brainstorming which variables are highly correlated, we realized that `position` and `stopping power` provides a better result of accuracy, recall, and precision. In addition, the model of SVM provides a good understanding of time complexity for all of us. On average, we take 10-15 minutes to run 68k of row of data just to predit the SVM model and the time complexity of the SVM model is around O(n^2) to O(n^3), in which we understand that it will take a significantly longer time to predict a model for SVM. It provides a great understanding for all of us to know the important of time complexity and how it would effect the efficiency in machine learning field.

By repetitively running and debugging the SVM Model and keep improving it, we get a better and better result in performance metrics over and over time. We have learned a lot from building this model although it takes the most of our time to run compared to other model. 

## Model 3: Regression - Predicting the Count of Each Decayed Particle
The way our dataframe reads is that at a specific stopping power and position (under either a circular or linear
collision), it gives the number of all decayed particles detected. As such, since our classification problem appears to be simple, with our second model we wanted to explore answering the question of how we could predict the number of decayed particles in different scenarios depending on stopping power, position, and collision type (linear or circular). 

To split our X and y training, testing, and validation sets, our X uses the linear, circular, stopping power, and position features while our y is the count of the 13 different decayed particles [mentioned here in our abstract](#abstract).

As our results show, our regression model was able to reach its best model after epoch 2 and our testing
MSE came out to be around 0.006, which indicates to us that our model did pretty well with its predictions. 

### Challenges and Shortcomings
With this model we did not come across any huge challenges or shortcomings. Our main concern, again, with receiving such high accuracy and low MSE scores is that our preprocessing steps introduced bias. Since we reiteratively reviewed our preprocessing steps and did everything we could to eliminate any bias such as:

1. normalizing our circular and linear data sets separately (since circular data initially had more counts)
1. splitting and normalizing our train, validation and test data sets separately to avoid any potential data leakage
1. making sure our training, testing, and validation data sets had equal samples of linear and circular collisions
1. Using K-folds validation to test consistancy 

   One thing to note is that the more epochs it does, the more the model overfits and preforms worse in testing. Because we have such a large dataset, after just 2 epochs the model has already apdated it weights a significant number of times. Thus we think only training for 2 epochs lead to the best testing MSE because it did not overfit the model as running it for longer epochs did 

we remain confident that our models and the training, validation, and testing sets are robust and as unbiased as possible.

## Model 4: 
I really wanted to practice unsupervized learning methods and I thought a data generator would be a cool thing to implement since it has real world applications such as helping scientists that do not have access to their own nuclear colliders to run experiments. In this model we used a encoder decoder VAE to generate 500 random samples of data. Based on the plots, the sampled data seems to closely tend towards the denser parts of the dataset scatter plot, which indicates that our model is doing what its suppose to do
### Challenges and Shortcomings
I initially spent a long time trying to figure out cool trends in the data using DBSCAN clustering. My original goal was to classify each of the particles into broader groups based on identified clusters. After spending days preprocessing the data for this task and futzing with parameters, I concluded that there are no interesting trends or clusters to be observed.
That being said, I wish I would have spent less time on that so that I could have spent more time on perfecting the generator. The generator was a lot of work because I came in with no background knowledge and had to figure out everything. Optimizing the parameters was very difficult and I still do not think my parameters are very good. It was very frustering to see how smooth the dataset distribution is, and yet the generated samples were not able to match it very closely no matter what I did. Another issue was that I really didn't know how to test it. I do not think the tests I was running are industry standard; they just made the most sense to me as a way to see how closely the sample matched the data

# Conclusion
Overall, we all learned a lot from this project and had a lot of fun working with this unique set of data. The main thing we wish we could have done differently is to explore more ways to enhanse our generator to generate more accurate synthetic data. We feel this would have a very attractive real world application because it would allow for scientists to have access to more particle collider data which is very expensive and hard to obtain. This is a direction we would like to go in the future when we continue this project on our own. Another thing we would have liked to do differently for this submission is to remove our svm model because its accuracy is very low and the model does not add anything to our project. However, we felt the need to include it because two of our group members worked hard on it and we wanted to make sure they would get credit for their contributions. In conclusion, we are very happy with what we were able to achieve with our binary classifier and regression neural net, however, if we had more time, we would like to improve our generator in model 4. 

# Collaboration
* Darian Lee: extracted the data from the root files; preprocessed and analysied all data; built models 1, 3, and 4, collaborated with Kayla in implementing the K-fold validation, helped write about models 1, 3, and 4 in final report
* Huy Nguyen: collaborated with Kahee in implementing the SVM model classification(model 2); detail description and result for model 2.
* Vincent Serracino: Introduced the particle collider dataset, knowledgeable about particle collider, our team result's interpreter in term of high level physics
* Kayla M. Araiza: collaborated with Darian in implementing K-Fold validation, restructor the README file, responsible for final report.
* Mujun Zhang: organize and editing README file for Preprocessing & First Model Building milestone.
* Kahee Chan: collaborated with Huy in implementing the SVM model classification(model 2); detail description, discussion, and result for model 2; responsible for visual report on the model

File description of data:
`Collider_circular.root`: titled `yieldHistos_7p7GeV_Pion_2022_08_31.root`
`Collider_linear.root`: titled `Output_File_2023_02_15.root`

***Group Member emails*** 
* [Darian Lee](deee@ucdavis.edu)
* [Huy Nguyen](hxnguyen@ucdavis.edu)
* [Vincent Serracino](vpserracino@ucdavis.edu)
* [Kayla M. Araiza](kmaraiza@ucdavis.edu)
* [Mujun Zhang](mjuzhang@ucdavis.edu)
* [Kahee Chan](kahchan@ucdavis.edu)

# Link To Google Colab Code 
https://colab.research.google.com/drive/1ZcecHvvYBBgO4CEE2mIW6B_TaWX5CkqI?usp=sharing
