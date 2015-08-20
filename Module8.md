# Machine Learning: Module 8

###
##Title: "Human Activity Recognition"
###
## 1. 		Executive Summary	
###

Data set related to exercise activity obtained from: http://groupware.les.inf.puc-rio.br/har was explored, analyse and model was builded and identify in order to help to predict how well activities (A,B,C,D,E) were performed. "A" is activity perform based on standard specification. Wherease "B","C","D" and "E" are activities which are intentionally performed "badly".

By using the Random Tree model, we can expect 100% accurary if run the model against 20 samples.

###
## 2. Environment preparation
###	



This is optional code to ustilise the mulitcore capability

```r
#optional: set multicore
library(cluster)
library(parallel)
library(doSNOW)

coreNumber=max(detectCores(),1)
cluster=makeCluster(coreNumber, type = "SOCK",outfile="")
registerDoSNOW(cluster)
```

Library and files are read in preparation of data exploratory analysis

```r
library(caret)
library(dplyr)
trainRaw <- read.csv("./pml-training.csv")
testRaw <- read.csv("./pml-testing.csv")
```

###
## 2. Exploratorty Data Analysis
###	

To avoid overfitting, we will identify predictors with the following guidelines:

1. Based on our knowledge and understanding, predictors contribute to the outcome (classe) are identified and included as predictors
2. Predictors with "NA" as values in the ALL column are identified and removed.
3. Preictors with zero variance are identified and removed.
 
Since by performing the activities (classe), the values are gathered through the devices and are stored in the column with words "belt", "forearm", "arm", and "dumbell". Therefore these predictors (together with classe) are extracted. 


```r
mystr = grep("belt|forearm|arm|dumbell|classe", colnames(trainRaw), value = TRUE)
trainRaw= trainRaw[,mystr]
```

Predictors with with all values = NA are removed since this will not add value to the prediction. 


```r
# colnames are identified
coln=colnames(trainRaw [,colSums(is.na( trainRaw )) != 0])
coli=match(coln,colnames(trainRaw))
trainClean = trainRaw [,-coli]
```

Predictors with near Zero Variance are removed since this predictors with these property does not contribute to the prediction. 34 are identified and removed.

```r
temp=nearZeroVar(trainClean)
trainClean=trainClean[,-temp]
```

###
## 3. Dataset Slicing
###	

The dataset originated from trainRaw are divided into 3 parts. 


1. trainData: To train the model
2. valData: To cross validate and out of sample estimation on the quality of the model and therefore from there select the best model.
3. testData: To assess the performance of the final model selected.
 
The model will be train using trainData and testing (testData). The validation set will be from the testRaw and this will be used to compute the out of sample error 

```r
# Training set & Testing set are preparated using slicing
set.seed(1234) 
trainSet = createDataPartition(y=trainClean$classe, p=0.7, list=F)
trainData = trainClean[trainSet,]
testData = trainClean[-trainSet,]
valClean = trainClean[-trainSet,]
valSet = createDataPartition(y=valClean$classe, p=0.5, list=F) 
valData = valClean[-valSet,]
testData = valClean[-valSet,]
```

###
## 3. Fit & Strategy for Model Selection
## 	a. Non-linear modeling (Decision Tree)
###	

The outcome (classe) is observed to be categorise (non-linear), therefore non-linear regression modeling is recommended. Two models (Decision Tree and Random Forest) will be built.



```r
#Decision tree (rpart). rpart function will bootstrap apply
modelCART=train(classe ~., data=trainData, method="rpart")
#modelCART$finalModel
predictCART=predict(modelCART, valData)
CART=confusionMatrix(valData$classe, predictCART)
```

###
## 	b. Non-linear modeling (Random Tree)
###	

The Random Tree model is robust to correlated covariates & outliners.


```r
# 2 fold cross validation is apply
controlRF = trainControl(method="cv", 2)
modelRF = train(classe ~ ., data=trainData, method="rf", trControl=controlRF, ntree=250)
# valData are used to x-validate the model and out-of-sample measurement on the model trained.
predictRF=predict(modelRF, valData)
RF=confusionMatrix(valData$classe, predictRF)
```

Optional: Others model tried.

```r
# Randown forest with PCA --- option 
# resource intensive
# random forest with pca will apply center and scale by default
# error using 0.8 with muticore
#ctrl = trainControl(preProcOptions = list(thresh = 0.9))
#modelRFPCA = train(classe ~ ., data = trainData, preProcess="pca", trControl = ctrl, method="rf")
#predictRFPCA=predict(modelRFPCA, testData)
#RFPCA=confusionMatrix(testData$classe, predictRFPCA)
```
###
## 3. Model Selection
###	

```r
# Display the accurarcy
CART
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 614   2  40 174   7
##          B 127 109 193 140   0
##          C 125  13 190 185   0
##          D  58   5  41 322  56
##          E  55   7 109  60 310
## 
## Overall Statistics
##                                           
##                Accuracy : 0.5252          
##                  95% CI : (0.5069, 0.5433)
##     No Information Rate : 0.3328          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.399           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.6272  0.80147  0.33159   0.3655   0.8311
## Specificity            0.8864  0.83607  0.86366   0.9224   0.9101
## Pos Pred Value         0.7336  0.19156  0.37037   0.6680   0.5730
## Neg Pred Value         0.8266  0.98862  0.84232   0.7728   0.9738
## Prevalence             0.3328  0.04623  0.19477   0.2995   0.1268
## Detection Rate         0.2087  0.03705  0.06458   0.1094   0.1054
## Detection Prevalence   0.2845  0.19341  0.17437   0.1638   0.1839
## Balanced Accuracy      0.7568  0.81877  0.59762   0.6439   0.8706
```

```r
RF
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 834   0   0   3   0
##          B   1 567   1   0   0
##          C   1   4 504   4   0
##          D   1   0   4 477   0
##          E   0   0   1   2 538
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9925          
##                  95% CI : (0.9887, 0.9953)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9905          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9964   0.9930   0.9882   0.9815   1.0000
## Specificity            0.9986   0.9992   0.9963   0.9980   0.9988
## Pos Pred Value         0.9964   0.9965   0.9825   0.9896   0.9945
## Neg Pred Value         0.9986   0.9983   0.9975   0.9963   1.0000
## Prevalence             0.2845   0.1941   0.1734   0.1652   0.1829
## Detection Rate         0.2835   0.1927   0.1713   0.1621   0.1829
## Detection Prevalence   0.2845   0.1934   0.1744   0.1638   0.1839
## Balanced Accuracy      0.9975   0.9961   0.9923   0.9897   0.9994
```

To select best model with categorise outcome, we will compare accurracy, sensitivity and specificity. It is not suprise that random forest is the best model   
with accurarcy = 99.3%, sensitivity of at least = 98.1% and specificity = 99.6%.

###
## 4. Conclusion for Model selected
###	

```r
predictModel=predict(modelRF, testData)
#RF=confusionMatrix(valData$classe, predictRF)
postResample(predictModel,testData$classe)
```

```
##  Accuracy     Kappa 
## 0.9925221 0.9905415
```

The accurarcy from test set is still 99.3%. Therefore given 20 samples, we can expect the selected model is able to provide 100% accurarcy on the prediction.

From the test set, the 

```r
# inspect the predictors that was selected.
varImp(modelRF)
```

```
## rf variable importance
## 
##   only 20 most important variables shown (out of 39)
## 
##                  Overall
## roll_belt         100.00
## yaw_belt           85.70
## pitch_forearm      68.39
## pitch_belt         66.95
## roll_forearm       56.14
## magnet_belt_z      42.99
## accel_belt_z       40.98
## roll_arm           39.97
## magnet_belt_y      36.55
## accel_forearm_x    35.81
## magnet_forearm_z   31.20
## magnet_forearm_y   29.66
## magnet_forearm_x   28.33
## magnet_arm_y       28.00
## accel_arm_x        27.68
## accel_forearm_z    27.22
## magnet_arm_x       27.16
## gyros_belt_z       24.83
## yaw_forearm        24.40
## magnet_belt_x      24.27
```
