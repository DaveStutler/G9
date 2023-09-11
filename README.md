### ECS 171: Machine Learning (Group 9)

# Analysis on the Particle Collider Dataset

# Abstract
We have two datasets, `Output_File_2023_02_15.root` which includes data about the linear collisions and `yieldHistos_7p7GeV_Pion_2022_08_31.root` which contains data about circular. The project idea is to analyze the datasets on the shape of explosions from a particle collider in order to determine what aspects of the particles used and conditions are common between collider and detector types. These findings would be significant because they can help researchers determine the collider type they should use when looking to collide heavy ion particles. We are not fully certain on what machine learning model we will use, but are considering unsupervised learning by applying a Convolutional neural network (CNN) to the data to identify correlations and connections between data points. 

# Introduction

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
1, dEdx_TritonPlus_Isolated;1, dEdx_TritonMinus_Isolated;1, dEdx_HelionPlus_Isolated;1, dEdx_HelionMinus_Isolated
1, dEdx_AlphaPlus_Isolated;1, and dEdx_AlphaMinus_Isolated;1. We decided to use these datapoints because each
feature isolates stopping power data per composite particle and these features coincide in both files.

### Descriptions of Graphs 
The y axis on our plots represents stopping power per particle. The x axis represents mass energy of a particle. We
decided to leave the y axis inverted when graphing in order to make the y axis stand out more from the x axis to
help with our analysis. Some interesting trends we noticed on these graphs were a small variance along the axis for
pions and deuteron and proton minus distributions and logarithmic correlations with other distributions.

## Data Preprocessing (steps)
1. Analyze original linear and circular `.root` files to determine which features coincide in both files
1. Convert 14 1000x1000 matrices into tabular data form
    1. One-hot encode our `linear` and `circular` features
    1. Create single dataframe columns for each of the 14 1000x1000 matrices 
    1. Concatenate single columns into 1 tabular dataframe
        1. columns consist of position, stopping power, and then all other different features
1. Generate normalized `.csv` files 
    1. Normalize `circular` and `linear` elements separately
    1. Split `train` and `test` sets 80:20 ratio
    1. Normalize `train` and `test` sets separately
    1. Create `X_train`, `y_train`, `X_test`, `y_test` splits and create `.csv` files for each. 

## Model 1: Binary Classification - Linear vs. Circular Collisions
### Description
In hopes of predicting the type of collider, either circular or linear, we use a 6-layers ANN to predict the type
that we one-hot encoded during our pre-processing phase. This feed forward neural net base on the selected
features of the particles to classify.
* Architecture: we use Relu activation functions for efficent runtime. We also use Sigmoid activation function in our output layer to classify the 2 groups, and use Binary Logarithmic Loss function to update our model weights and bias. We split our dataset into 90:10 of propotion, with linear and circular columns as our target and every other columns as our features.
* We check the MSE and accurary_score for performance, as well as illustrating the classification report on our
result.

## Model 2: Linear Regression - Predicting the Stopping Power and Position
### Description
To increase the complexity of our project, we decided to make an attempt at using counts of particles detected to
predict stopping power and position. To do this, we use build a 5-layer linear regression ANN model. 
* Architecture: In our 1st hidden layer we use a linear activation function than then feeds its outputs into the
2nd hidden layer which uses a LeakyRelu activation functions with a learning rate of 0.01 (the 3rd and 4th hidden
layers follow this same architecture). The output layer makes use of a linear activation function. To update our
model weights and bias, we  use the MSE loss function.
* We check the MSE and accurary_score for performance, as well as illustrating the classification report on our
result.

## Model 3: SVM - 
### Description

## Model 4:
### Description

# Results
## Data Preprocessing 
(wip)

## Model 1: Binary Classification - Linear vs. Circular Collisions
In our project, using a 4-layer ANN to process complex data is a reasonable choice. The Relu activation function can speed up training because it does not involve exponential operations, while the Sigmoid activation function is suitable for classification problems. It is also appropriate to use a sigmoid output layer for binary classification. Choosing an appropriate activation function and number of network layers can improve the performance of the model.
* Data Preprocessing: In the data preprocessing stage, the types are one-hot encoded, which is a common practice for multi-class classification problems. It is also standard practice to split the dataset into 90:10 train and test sets to evaluate the performance of the model.
* Performance Metrics: This experiment evaluates the performance of the model using mean squared error (MSE) and accuracy. These two metrics provide important information about the performance of the model on the training and test data. MSE is used to measure the prediction error of continuous output, while accuracy (Accuracy) is used to evaluate the performance of classification models. The use of these two metrics is appropriate because they provide different aspects of performance information.
* Classification Report: Generating a classification report is mentioned, which is a good practice. Classification reports usually include indicators such as precision, recall, F1 score, etc., which are very helpful for understanding the performance of the model on each category. This can help determine if the model is performing better or worse on certain categories.
* Tuning and Improvement: While some descriptions of the model are provided, no hyperparameter tuning or other steps for further improvement are mentioned in the lab report. In practical applications, multiple trials and adjustments are usually required to optimize model performance.
* For the four cross-validation folds (Folds), the MSE of each fold is very close, all around 0.018.
The average MSE value is 0.018, which means that the average prediction error of the model is relatively small. MSE measures the difference between the predicted value of the model and the actual value, and a smaller MSE indicates a better predictive performance of the model.
* R2 Score Analysis:
For the four cross-validation folds, the R2 scores for each fold range from 0.413 to 0.416.
The average R2 score is 0.414, which is a relatively stable value. The R2 score measures how well the model fits the observed data, and the closer to 1, the better the fit.
* Overall average MSE and R2:
The overall mean MSE was 0.018 and the overall mean R2 was 0.414. This shows that the performance of the model on the entire data set is also stable, and the fitting effect on the data is better.
* Collapse of best MSE and R2 scores:
First-fold cross-validation achieves the best performance in terms of MSE and R2. This means that at this compromise, the model has the smallest prediction error and the best fit to the data.
* Overall, this project appears to be a reasonable modeling of a classification problem, using an appropriate neural network architecture and performance evaluation metrics. However, more detailed information, such as the choice of hyperparameters and the results of the training process, as well as a broader model performance report, would provide a more complete understanding of the quality of experiments and model performance

## Model 2: Linear Regression - Predicting the Stopping Power and Position

## Model 3: SVM - 

## Model 4: 

# Discussion
## Data Exploration
The goal in our Data Exploration phase was 3-fold:
1. figure out a way to access the data in our two `.root` files
1. figure out what features were common in both files
1. discover meaning in the data and their respective graphs

After extracting data from the `.root` files and making pandas dataframes out of them is how we were able to figure
out that our dataframe had only 2 rows (one for circular and the other for spherical collisions) each of which
contained 1000x1000 matrices representing data from over 91,000 collisions.

Through careful analysis of the graphs these matrices made, in addition to previous knowledge and experience one of our group members had with similar datasets, we were able to deduce what the rows, columns, and values of the matrices represented:
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
From our Data Exploration milestone, we were able to extract 14 1000x1000 matrices.
 
After careful consideration, we believed it to be more beneficial for us to convert these matrices into tabular
data form as it is something we are all more familiar and comfortable working with. 

Before creating the 4 normalized X_train, y_train, X_test, y_test `.csv` files, we first one-hot encoded our `linear` and `circular` features. After doing so, we  then actually began the process of creating the dataframe columns (i.e. converting the matrices), which consisted of the positions, stopping power, the detection number for different particle types, and the one hot encoded columns representing circular and linear collider types. This process involved the challenge of trying to extract only the most useful parts of each matrix so that the size did not get too overwhelmingly large, while still keeping the rows and columns between matrices exactly the same indexes so that we could merge them into a dataframe smoothly based on the position and stopping power columns. 

After each matrix was converted into a tabular dataform, we then merged each of them into a single tabular
dataframe, after which we normalized all features before beginning training and testing of our model. We normalized
the circular data and the linear data seperately because the circular data on average had many more detections then
linear, and we did not want this to bias our model. The circular data came from a differenrt root file where its
likely more events were run, causing the circular data to have higher counts. A model that trained on this data
would not have generalized to the real world, so we scaled all columns seperately for circular and linear collider
types. 

After separately normalizing our circular and linear collider types, we then split and normalized our train and
test sets separately to avoid any potential data leakage. The reason for including both X and y values in our
normalization function is to ensure that our data is normalized separately based on the different collider shapes
(to account for differences in the sample sizes for linear and circular collisions). After normalization of the X
train and X test splits, we then remove the `linear` and `circular` labels to prevent our models from memorizing. 

### Challenges and Shortcomings
The biggest challenge was figuring how to convert 26 1000x1000 matrices into a nice readable dataframe. We had to
decide how we wanted to represent this data in our dataframe, and once we decided that we wanted to merge all the
particles from different matrices into the same rows based on stopping power and position, it became a big issue to
try to make sure that we are extracting the data at the exact same places in the matrices for all features, and
that the places we are extracting data would contain the least amount of useless data (all zeros would be
considered useless). 

In retrospect, it would have been easier to do this based on looking at areas across all graphs where nost data
seemed to be concentrated. However, the approach we choose involved interating through matrices to find the maximum
and minimum row and column where there is meaningful data, using those values to find the global minimum and
maximum, evaluating at those points and creating a dataframe for each feature, recursively merging all dataframes
into one, and then normalizing and removing noise and all rows of all 0s 

## Model 1: Binary Classification - Linear vs. Circular Collisions

## Model 2: Linear Regression - Predicting the Stopping Power and Position

## Model 3: SVM -

## Model 4: 

# Conclusion

# Collaboration

File description of data:
`Collider_circular.root`: titled `yieldHistos_7p7GeV_Pion_2022_08_31.root`
`Collider_linear.root`: titled `Output_File_2023_02_15.root`

***Group Member emails*** 
[Darian Lee](deee@ucdavis.edu)
[Huy Nguyen](hxnguyen@ucdavis.edu)
[Vincent Serracino](vpserracino@ucdavis.edu)
[Kayla M. Araiza](kmaraiza@ucdavis.edu)
[Mujun Zhang](mjuzhang@ucdavis.edu)
[Kahee Chan](kahchan@ucdavis.edu)

# Link To Google Colab Code 
https://colab.research.google.com/drive/1ZcecHvvYBBgO4CEE2mIW6B_TaWX5CkqI?usp=sharing
