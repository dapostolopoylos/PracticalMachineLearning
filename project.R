setwd("D:\\Machine Learning\\CourseProject")

## Read the train and test datasets that have already been downloaded.
train <- read.csv("data/pml-training.csv",sep=",",header = TRUE,stringsAsFactors = FALSE,na.strings = c("NA","#DIV/0!", ""))
test <- read.csv("data/pml-testing.csv",sep=",",header = TRUE,stringsAsFactors = FALSE,na.strings = c("NA","#DIV/0!", ""))

## Do some data cleansing.
test<-test[,colSums(is.na(test)) == 0]
test<-test[,-c(1,2)]
train<-train[,colSums(is.na(train)) == 0]
train<-train[,-c(1,2)]

## Start processing and load libraries
set.seed(12345)
suppressMessages(library(caret))
library(rpart)
suppressMessages(library(randomForest))
suppressMessages(library(doParallel))

inTrain <- createDataPartition(train$classe,p=0.6,list=FALSE)
myTrain <- train[inTrain,] 
myTest <- train[-inTrain,]

## Classification Tree without Cross validation
fitCL <- train(myTrain$classe~.,data=myTrain,method="rpart")
predCL <- predict(fitCL,myTest)
confusionMatrix(predCL,myTest$classe) ## Accuracy: 0.4906

## Classification Tree with Cross validation
train_control <- trainControl(method="cv", number=5)
fitCL_CV <- train(myTrain$classe~.,data=myTrain,method="rpart",trControl=train_control)
predCL_CV <- predict(fitCL_CV,myTest)
confusionMatrix(predCL_CV,myTest$classe) ## Accuracy: 0.4906

## Classification Tree with Cross validation and Preprocessing
fitCL_CV_PR <- train(myTrain$classe~.,data=myTrain,method="rpart",trControl=train_control,preProcess=c("center", "scale"))
predCL_CV_PR <- predict(fitCL_CV_PR,myTest)
confusionMatrix(predCL_CV_PR,myTest$classe) ## Accuracy: 0.4906

# number of cores to use
cl <- makeCluster(20)
registerDoParallel(cl)

## Random Forest
fitRF <- train(myTrain$classe~.,data=myTrain,method="rf",allowParallel = TRUE)
 
stopCluster(cl)	

predRF <- predict(fitRF,myTest) 
confusionMatrix(predRF,myTest$classe) ## Accuracy : 0.9996 

cl <- makeCluster(20)
registerDoParallel(cl)

## Random Forest with Cross Validation
fitRF_CV <- train(myTrain$classe~.,data=myTrain,trControl=train_control,method="rf",allowParallel = TRUE)
 
stopCluster(cl)

predRF_CV <- predict(fitRF_CV,myTest)
confusionMatrix(predRF_CV,myTest$classe) ## Accuracy : 0.9989 

finalPRED<-predict(fitRF,test)

