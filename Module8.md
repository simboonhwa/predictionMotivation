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

Since based the outcome (classe), the values gathered through the devices are stored in the column with words "belt", "forearm", "arm", and "dumbell", therefore these predictors (together with classe) are extracted. 


```r
mystr = grep("belt|forearm|arm|dumbell|classe", colnames(trainRaw), value = TRUE)
trainRaw= trainRaw[,mystr]

#remove those columns with all value = NA since this will not add value to the prediction. ??? are identified with on NA value and removed.

# colnames are identified
coln=colnames(trainRaw [,colSums(is.na( trainRaw )) != 0])

# column index are identify to apply to the test set.
coli=match(coln,colnames(trainRaw))
trainClean = trainRaw [,-coli]

# remove those predictors with near Zero Variance since this predictors with thsee property does not contribute to prediction. 34 are identified and removed.
temp=nearZeroVar(trainClean)
trainClean=trainClean[,-temp]

#nzv=nearZeroVar(trainClean, saveMetrics= TRUE)
#nzv[nzv$nzv,]

#append column index for "near zero var" predicators into coli.
#coli=c(coli,temp) %>% unique %>% sort


#col_names <- names(trainRaw)
#lapply(trainRaw[,col_names] , factor)
#temp=lapply(trainRaw[,colnames(trainClean)] , is.factor)
#str(temp)
```

###
## 3. Dataset Slicing
###	


```r
# Training set & Testing set are preparated using slicing
set.seed(1234) 
trainSet = createDataPartition(y=trainClean$classe, p=0.7, list=F)
trainData = trainClean[trainSet,]
testData = trainClean[-trainSet,]
```


###
## 3. Dataset Slicing
###	
temp=dummyVars( ~ ., data = trainData)
trainlmData=predict(temp, newdata = trainData))


You can also embed plots, for example:



