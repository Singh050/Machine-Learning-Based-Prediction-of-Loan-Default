# Group 5
# Members - Bristi Bose, Barjinder Singh, Hemapriya Kangala, Abdul Aziz

# Load necessary libraries
library(caret)
library(adabag)
library(class)
library(glmnet)
library(randomForest)
library(xgboost)
library(nnet) # for neural nets
library(e1071) # for SVM
library(DMwR) # for SMOTE
library(dplyr)
library(pROC)
library(ggplot2)
library(gbm)
library(catboost)
library(rpart)
library(doParallel)

# Set up parallel processing with 2 cores
cl <- makeCluster(2)
registerDoParallel(cl)


# Load the dataset
loan <- read.csv("Loan_default.csv")

# Drop LoanID and preprocess the data
loan <- loan %>% select(-LoanID)


boxplot(loan$Income, main = "Income Boxplot", horizontal = TRUE, col = "lightblue")
boxplot(loan$LoanAmount, main = "Loan Amount Boxplot", horizontal = TRUE, col = "lightgreen")
boxplot(loan$CreditScore, main = "Credit Score Boxplot", horizontal = TRUE, col = "lightpink")


# Normalize numerical columns (remove attributes afterward)
loan$Income <- as.numeric(scale(loan$Income))
loan$LoanAmount <- as.numeric(scale(loan$LoanAmount))
loan$CreditScore <- as.numeric(scale(loan$CreditScore))

# Convert binary categories to 0/1
loan$HasMortgage <- ifelse(loan$HasMortgage == "Yes", 1, 0)
loan$HasDependents <- ifelse(loan$HasDependents == "Yes", 1, 0)
loan$HasCoSigner <- ifelse(loan$HasCoSigner == "Yes", 1, 0)

# One-hot encode categorical variables
dummy_vars <- dummyVars(" ~ Education + EmploymentType + MaritalStatus + LoanPurpose", data = loan)
loan_dummy <- data.frame(predict(dummy_vars, newdata = loan))
loan <- cbind(loan, loan_dummy)

# Remove original categorical columns
loan <- loan %>% select(-Education, -EmploymentType, -MaritalStatus, -LoanPurpose)

# Ensure Default is a factor
loan$Default <- as.factor(loan$Default)

# Handle class imbalance using SMOTE
set.seed(123)
smote_data <- SMOTE(Default ~ ., data = loan, perc.over = 100, perc.under = 200)


# Train-test split
set.seed(123)
# Before SMOTE
barplot(table(loan$Default), main = "Class Distribution Before SMOTE", col = c("blue", "red"))
train_index <- createDataPartition(smote_data$Default, p = 0.8, list = FALSE)
# After SMOTE
barplot(table(smote_data$Default), main = "Class Distribution After SMOTE", col = c("blue", "red"))

train_data <- smote_data[train_index, ]
test_data <- smote_data[-train_index, ]

# Create a smaller subset of train_data for SVM
set.seed(123)
small_train_data <- train_data %>% group_by(Default) %>% sample_frac(0.2)



# Density plot for Income by Default
ggplot(smote_data, aes(x = Income, fill = Default)) +
  geom_density(alpha = 0.5) +
  labs(title = "Density Plot of Income by Default", x = "Income", y = "Density") +
  theme_minimal()

# Density plot for Loan Amount by Default
ggplot(smote_data, aes(x = LoanAmount, fill = Default)) +
  geom_density(alpha = 0.5) +
  labs(title = "Density Plot of Loan Amount by Default", x = "Loan Amount", y = "Density") +
  theme_minimal()


# Modeling

# 1. KNN
# Scale data
train_scaled <- scale(train_data[, -which(names(train_data) == "Default")])
test_scaled <- scale(test_data[, -which(names(test_data) == "Default")])
# Train KNN
knn_pred <- knn(train_scaled, test_scaled, cl = train_data$Default, k = 5)


# 2. Train Naive Bayes
nb_model <- naiveBayes(Default ~ ., data = train_data)
nb_pred <- predict(nb_model, test_data, type = "raw")[, 2]


# 3. Penalized Logistic Regression (glmnet)
x_train <- model.matrix(Default ~ ., train_data)[, -1]
y_train <- as.numeric(train_data$Default) - 1

cv_glmnet <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 0.5)
glmnet_model <- glmnet(x_train, y_train, family = "binomial", alpha = 0.5, lambda = cv_glmnet$lambda.min)
x_test <- model.matrix(Default ~ ., test_data)[, -1]
glmnet_pred <- predict(glmnet_model, newx = x_test, type = "response")
glmnet_pred_class <- ifelse(glmnet_pred > 0.5, 1, 0)

# 4. Random Forest
rf_tuned <- train(
  Default ~ ., data = train_data, method = "rf",
  tuneGrid = expand.grid(.mtry = c(2, 3, 4)),
  trControl = trainControl(
    method = "cv", number = 5, allowParallel = TRUE
  )
)
rf_pred <- predict(rf_tuned, test_data)

# 5. XGBoost
dtrain <- xgb.DMatrix(data = x_train, label = y_train)
dtest <- xgb.DMatrix(data = x_test)

xgb_params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 6,
  eta = 0.1,
  subsample = 0.8,
  colsample_bytree = 0.8
)
xgb_tuned <- xgb.cv(
  params = xgb_params, data = dtrain, nrounds = 100, nfold = 5,
  early_stopping_rounds = 10, verbose = 0
)
xgb_model <- xgboost(params = xgb_params, data = dtrain, nrounds = xgb_tuned$best_iteration)
xgb_pred <- predict(xgb_model, dtest)
xgb_pred_class <- ifelse(xgb_pred > 0.5, 1, 0)

# 6. Neural Network
nn_tuned <- train(
  Default ~ ., data = train_data, method = "nnet",
  tuneGrid = expand.grid(.size = c(5, 10), .decay = c(0.1, 0.5)),
  trControl = trainControl(method = "cv", number = 5, allowParallel = TRUE),
  trace = FALSE
)
nn_pred <- predict(nn_tuned, test_data)


# 7. Support Vector Machine (SVM) with smaller data and reduced grid
svm_tuned <- train(
  Default ~ ., data = small_train_data, method = "svmRadial",
  tuneGrid = expand.grid(.C = c(1, 10), .sigma = c(0.1, 1)),
  trControl = trainControl(method = "cv", number = 5, allowParallel = TRUE)
)
svm_pred <- predict(svm_tuned, test_data)


# 8. Gradient Boosting Machine (GBM)
gbm_tuned <- train(
  Default ~ ., data = train_data, method = "gbm",
  tuneGrid = expand.grid(
    n.trees = c(100, 200), 
    interaction.depth = c(3, 5), 
    shrinkage = c(0.01, 0.1),
    n.minobsinnode = c(10, 20)
  ),
  trControl = trainControl(method = "cv", number = 5, allowParallel = TRUE),
  verbose = FALSE
)
gbm_pred <- predict(gbm_tuned, test_data)


# 9. Decision Tree
decision_tree_model <- train(
  Default ~ ., data = train_data, method = "rpart",
  tuneGrid = expand.grid(.cp = seq(0.01, 0.1, by = 0.01)),
  trControl = trainControl(method = "cv", number = 5, allowParallel = TRUE)
)
decision_tree_pred <- predict(decision_tree_model, test_data)


# 10. Elastic Net
x_train <- model.matrix(Default ~ ., train_data)[, -1]
y_train <- as.numeric(train_data$Default) - 1

cv_glmnet <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 0.5)
elastic_net_model <- glmnet(x_train, y_train, family = "binomial", alpha = 0.5, lambda = cv_glmnet$lambda.min)
x_test <- model.matrix(Default ~ ., test_data)[, -1]
elastic_net_pred <- predict(elastic_net_model, newx = x_test, type = "response")
elastic_net_pred_class <- ifelse(elastic_net_pred > 0.5, 1, 0)


### Evaluate Performance
###KNN
roc_knn <- roc(as.numeric(test_data$Default), as.numeric(knn_pred))
auc_knn <- auc(roc_knn)


# Logistic Regression
roc_glmnet <- roc(as.numeric(test_data$Default), as.numeric(glmnet_pred_class))
auc_glmnet <- auc(roc_glmnet)

# Random Forest
roc_rf <- roc(as.numeric(test_data$Default), as.numeric(rf_pred))
auc_rf <- auc(roc_rf)

# XGBoost
roc_xgb <- roc(as.numeric(test_data$Default), as.numeric(xgb_pred_class))
auc_xgb <- auc(roc_xgb)

# Neural Net
roc_nn <- roc(as.numeric(test_data$Default), as.numeric(nn_pred))
auc_nn <- auc(roc_nn)

# SVM
roc_svm <- roc(as.numeric(test_data$Default), as.numeric(svm_pred))
auc_svm <- auc(roc_svm)

# Evaluate GBM
roc_gbm <- roc(as.numeric(test_data$Default), as.numeric(gbm_pred))
auc_gbm <- auc(roc_gbm)


# Evaluate Decision Tree
roc_decision_tree <- roc(as.numeric(test_data$Default), as.numeric(decision_tree_pred))
auc_decision_tree <- auc(roc_decision_tree)

# Evaluate Elastic Net
roc_elastic_net <- roc(as.numeric(test_data$Default), as.numeric(elastic_net_pred_class))
auc_elastic_net <- auc(roc_elastic_net)

roc_nb <- roc(as.numeric(test_data$Default), nb_pred)
auc_nb <- auc(roc_nb)


# Plot ROC curves for all models
plot(roc_knn, col = "yellow", main = "ROC Curves for Models")
plot(roc_glmnet, col = "red", add = TRUE)
plot(roc_rf, col = "blue", add = TRUE)
plot(roc_xgb, col = "green", add = TRUE)
plot(roc_nn, col = "purple", add = TRUE)
plot(roc_svm, col = "orange", add = TRUE)
plot(roc_gbm, col = "pink", add = TRUE)
plot(roc_decision_tree, col = "brown", add = TRUE)
plot(roc_elastic_net, col = "darkgreen", add = TRUE)
plot(roc_nb, col = "cyan", add = TRUE) # Added Naive Bayes

# Update legend to include Naive Bayes 
legend("bottomright", legend = c(
  "KNN", "Logistic Regression", "Random Forest", "XGBoost", "Neural Net", 
  "SVM", "GBM", "Decision Tree", "Elastic Net", "Naive Bayes"
), col = c("yellow", "red", "blue", "green", "purple", 
           "orange", "pink", "brown", "darkgreen", "cyan"), lwd = 2)



# Print AUCs for all models
cat("KNN AUC:", auc_knn, "\n")
cat("Penalized Logistic Regression AUC:", auc_glmnet, "\n")
cat("Random Forest AUC:", auc_rf, "\n")
cat("XGBoost AUC:", auc_xgb, "\n")
cat("Neural Net AUC:", auc_nn, "\n")
cat("SVM AUC:", auc_svm, "\n")
cat("Gradient Boosting Machine (GBM) AUC:", auc_gbm, "\n")
cat("Decision Tree AUC:", auc_decision_tree, "\n")
cat("Elastic Net AUC:", auc_elastic_net, "\n")
cat("Naive Bayes AUC:", auc_nb, "\n") # Added Naive Bayes AUC




# Collect AUC values for all 10 models
auc_values <- data.frame(
  Model = c(
    "KNN", "Penalized Logistic Regression", "Random Forest", 
    "XGBoost", "Neural Net", "SVM", "Gradient Boosting Machine (GBM)", 
    "Decision Tree", "Elastic Net", "Naive Bayes"
  ),
  AUC = c(
    auc_knn, auc_glmnet, auc_rf, auc_xgb, auc_nn, 
    auc_svm, auc_gbm, auc_decision_tree, auc_elastic_net, auc_nb
  )
)

# Bar Plot for AUC Comparison
barplot(
  auc_values$AUC, 
  names.arg = auc_values$Model, 
  col = "skyblue", 
  las = 2, # Rotate x-axis labels
  main = "Model AUC Comparison", 
  ylim = c(0, 1), 
  ylab = "AUC"
)

# Add text to the bar plot
text(
  x = seq_along(auc_values$AUC), 
  y = auc_values$AUC, 
  labels = round(auc_values$AUC, 3), 
  pos = 3, 
  cex = 0.8, 
  col = "black"
)


# Stop parallel processing
stopCluster(cl)

# Memory management
rm(loan, smote_data, train_data, test_data, dtrain, dtest, small_train_data)
gc()

