#Download the required libraries
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(reshape2)) install.packages("reshape2", repos = "http://cran.us.r-project.org")
library(tidyverse)
library(caret)
library(data.table)
library(ggplot2)
library(reshape2)

#Open 'Biomechanical features of orthopedic patients' data-set from my local machine. 
setwd("/Users/arsankar/Desktop/Knowledge/Data Science/Module 9 - Capstone/CYO")
#Please note that I have already downloaded the data-set from Kaggle to my local directory.
#Hence,the above line of code wouldnt work on another PC as this is specific to my local machine.
#So, for you to be able to run my overall code, request you to download the data-set from
#my GitHub Repository #https://github.com/AravindAmazon/CYO---Orthopedic-BioMechanical-features
#and update your local machine directory path above.



#Read CSV file and store in data-frame
datasetnew <- read.csv("column_3C_weka.csv", sep = "|", header = TRUE)
#Understand the data-set. Firstly, see a high-level view of all the fields
glimpse(datasetnew)

#Understand data-types and min-max values for each field
summary(datasetnew)
#Count the number of empty cells in the data-set
sum(!complete.cases(datasetnew))
#See a boxplot view of class vs. each feature
datasetnewbox<-melt(datasetnew, id.var="class")
bp<-ggplot(data=datasetnewbox, aes(x=variable, y=value)) + geom_boxplot(aes(fill=class))
bp+facet_wrap(~variable, scales = "free")
#Deploy Model 1 - Multinomial Logistic Regression
#Partition training and test sets in 80-20 ratio
test_index <- createDataPartition(y = datasetnew$class, times = 1, p = 0.2, list = FALSE)
training_set <- datasetnew[-test_index,]
test_set <- datasetnew[test_index,]

#Build the model on the training set
library(nnet)
my_model <- multinom(class~., data = training_set)
#Validate against the test set
my_model_results <- predict(my_model,test_set)

#Plot confusion matrix
confusionMatrix(data = my_model_results, reference = test_set$class)

#Start tabling accuracy results for each iteration
resulttable1 <- table(observed = test_set$class, predicted = my_model_results)
accuracy1 <- sum(diag(resulttable1))/length(my_model_results)
#Accuracy from Multinomial Logistic Regression model is:
print(accuracy1)
#Now, let us understand if a tree-structure can be a good alternative model fit for the data
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(rpart.plot)) install.packages("rpart.plot", repos = "http://cran.us.r-project.org")
library (rpart)
library(rpart.plot)
ctree_temp <- rpart(class~., data = training_set, method = 'class')
rpart.plot(ctree_temp)

# With all the predictors being continuous variables within a set range, a tree-structure can not only
#be a good fit but also show a decision-tree based visualization of the model that is
#easily interpretable. Aggregation of many such decision trees can help with a variety of
#options to tune and enhance accuracy. Hence, a RandomForest model can be deployed in this regard.
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
library(randomForest)
rfmodel <- randomForest(class~., data = training_set, ntree=10)
#See model summary
rfmodel
#Validate on the test set
rfp <- predict(rfmodel, test_set)
#See overall accuracy
resulttable2a <- table(observed = test_set$class, predicted = rfp)
resulttable2a
accuracy2a <- sum(diag(resulttable2a))/length(rfp)
#Accuracy from the baseline model of RandomForest is:
print(accuracy2a)
# Tune for optimum mtry
mtry <- tuneRF(training_set[-7],training_set$class,ntreeTry=100, stepFactor = 0.5, improve = 0.01, trace = TRUE, plot = TRUE)
best.m <- mtry[mtry[,2]==min(mtry[,2]),1]
mtry
best.m
# Applying optimum mtry value and also increasing tree-size to 100
rfmodel_tuned <- randomForest(class~., data = training_set, ntree=100, mtry = best.m)
rfp_tuned <- predict(rfmodel_tuned, test_set)
resulttable2b <-table(observed = test_set$class, predicted = rfp_tuned)
resulttable2b
accuracy2b <- sum(diag(resulttable2b))/length(rfp_tuned)
# Accuracy after optimizing mTry for least error estimate is :
print(accuracy2b)
Model <- c("Multinomial Logistic Regression","Random Forest","Random Forest(tuned to optimize mtry)")
Accuracy <- c(accuracy1, accuracy2a, accuracy2b)
data.frame(Model, Accuracy)