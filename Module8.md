# Machine Learning: Module 8

###
##Title: "Human Activity Recognition"
###
## 1. 		Executive Summary	
###

Data set related to exercise activity obtained from: http://groupware.les.inf.puc-rio.br/har was explored, analyse and model was builded and identify in order to help to predict how well activities (A,B,C,D,E) were performed. "A" is activity perform based on standard specification. Wherease "B","C","D" and "E" are activities which are intentionally performed "badly".



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

Since by performing the activities (classe), the values are gathered through the devices and are stored in the column with words "belt", "forearm", "arm", and "dumbell". Therefore these predictors (together with classe) are extracted. 


```r
mystr = grep("belt|forearm|arm|dumbell|classe", colnames(trainRaw), value = TRUE)
trainRaw= trainRaw[,mystr]

#remove those columns with all value = NA since this will not add value to the prediction. 

# colnames are identified
coln=colnames(trainRaw [,colSums(is.na( trainRaw )) != 0])
coli=match(coln,colnames(trainRaw))
trainClean = trainRaw [,-coli]

# remove those predictors with near Zero Variance since this predictors with these property does not contribute to the prediction. 34 are identified and removed.
temp=nearZeroVar(trainClean)
trainClean=trainClean[,-temp]
```

###
## 3. Dataset Slicing
###	

The dataset originated from trainRaw are divided into 2 parts. For modelling (trainData) and testing (testData). The validation set will be from the testRaw and this will be used to compute the out of sample error 

```r
# Training set & Testing set are preparated using slicing
set.seed(1234) 
trainSet = createDataPartition(y=trainClean$classe, p=0.7, list=F)
trainData = trainClean[trainSet,]
testData = trainClean[-trainSet,]
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
predictCART=predict(modelCART, testData)
CART=confusionMatrix(testData$classe, predictCART)
accurarcyCART=postResample(predictCART,testData$classe)
```

###
## 	b. Non-linear modeling (Random Tree)
###	

The Random Tree model is robust to correlated covariates & outliners.


```r
# 2 fold cross validation is apply
controlRF = trainControl(method="cv", 2)
modelRF = train(classe ~ ., data=trainData, method="rf", trControl=controlRF, ntree=250)
predictRF=predict(modelRF, testData)
RF=confusionMatrix(testData$classe, predictRF)
accurarcyRF=postResample(predictRF,testData$classe)
```


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
## 3. Conclusion for model
###	

```r
# Display the accurarcy
#accurarcyCART
#accurarcyRF
CART
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1244    3   90  328    9
##          B  265  218  366  288    2
##          C  243   22  404  357    0
##          D  133   10   72  645  104
##          E   97   10  221  128  626
## 
## Overall Statistics
##                                           
##                Accuracy : 0.5331          
##                  95% CI : (0.5202, 0.5459)
##     No Information Rate : 0.3368          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.4087          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.6276  0.82890  0.35039   0.3694   0.8448
## Specificity            0.8898  0.83618  0.86855   0.9229   0.9114
## Pos Pred Value         0.7431  0.19140  0.39376   0.6691   0.5786
## Neg Pred Value         0.8247  0.99052  0.84585   0.7763   0.9761
## Prevalence             0.3368  0.04469  0.19592   0.2967   0.1259
## Detection Rate         0.2114  0.03704  0.06865   0.1096   0.1064
## Detection Prevalence   0.2845  0.19354  0.17434   0.1638   0.1839
## Balanced Accuracy      0.7587  0.83254  0.60947   0.6462   0.8781
```

```r
RF
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    1    0    0    0
##          B    4 1132    3    0    0
##          C    0    9 1010    7    0
##          D    0    2   11  951    0
##          E    0    1    0    3 1078
## 
## Overall Statistics
##                                          
##                Accuracy : 0.993          
##                  95% CI : (0.9906, 0.995)
##     No Information Rate : 0.285          
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9912         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9976   0.9886   0.9863   0.9896   1.0000
## Specificity            0.9998   0.9985   0.9967   0.9974   0.9992
## Pos Pred Value         0.9994   0.9939   0.9844   0.9865   0.9963
## Neg Pred Value         0.9991   0.9973   0.9971   0.9980   1.0000
## Prevalence             0.2850   0.1946   0.1740   0.1633   0.1832
## Detection Rate         0.2843   0.1924   0.1716   0.1616   0.1832
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9987   0.9936   0.9915   0.9935   0.9996
```

To select best model with categorise outcome, we will compare accurracy, sensitivity and specificity. It is not suprise that random forest is the best model   
with accurarcy = 99.3%, sensitivity of at least = 98.6% and specificity = 99.6%.


```r
# inspect the predictors that was selected.
varImp(modelCART)
```

```
## rpart variable importance
## 
##   only 20 most important variables shown (out of 39)
## 
##                     Overall
## pitch_forearm        100.00
## roll_belt             97.70
## roll_forearm          90.66
## yaw_belt              48.21
## magnet_belt_y         40.91
## accel_forearm_x       37.31
## accel_belt_z          35.12
## pitch_belt            33.64
## total_accel_belt      29.91
## gyros_belt_z          22.05
## roll_arm              21.68
## magnet_arm_x          20.72
## accel_arm_x           20.69
## yaw_arm               18.40
## gyros_arm_y            0.00
## gyros_arm_z            0.00
## magnet_forearm_y       0.00
## total_accel_forearm    0.00
## magnet_arm_z           0.00
## gyros_belt_x           0.00
```

```r
varImp(modelRF)
```

```
## rf variable importance
## 
##   only 20 most important variables shown (out of 39)
## 
##                  Overall
## roll_belt        100.000
## yaw_belt          75.284
## pitch_forearm     70.284
## pitch_belt        62.096
## roll_forearm      49.291
## magnet_forearm_z  21.812
## accel_forearm_z   20.005
## accel_forearm_x   18.998
## magnet_belt_z     17.013
## roll_arm          16.854
## magnet_belt_x     14.413
## yaw_arm           14.251
## accel_belt_z      14.001
## magnet_belt_y     13.216
## yaw_forearm       11.921
## magnet_forearm_y  11.751
## gyros_arm_y        8.406
## accel_forearm_y    8.382
## magnet_arm_x       8.346
## magnet_arm_y       8.191
```
