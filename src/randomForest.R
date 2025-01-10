library(tidyverse)
library(randomForest)
library(caret)
library(fastDummies)

house_data <- sampled_data.csv

set.seed(123)

# Define regions
northeast <- c("Connecticut", "Maine", "Massachusetts", "New Hampshire",
               "Rhode Island", "Vermont", "New Jersey", "New York",
               "Pennsylvania")
midwest <- c("Illinois", "Indiana", "Michigan", "Ohio", "Wisconsin",
             "Iowa", "Kansas", "Minnesota", "Missouri",
             "Nebraska", "North Dakota", "South Dakota")
west <- c("Arizona", "Colorado", "Idaho", "Montana", "Nevada",
          "New Mexico", "Utah", "Wyoming", "Alaska", "California",
          "Hawaii", "Oregon", "Washington")
south <- c("Delaware", "Florida", "Georgia", "Maryland",
           "North Carolina", "South Carolina", "Virginia",
           "West Virginia", "District of Columbia",
           "Alabama", "Kentucky", "Mississippi",
           "Tennessee", "Arkansas", "Louisiana",
           "Oklahoma", "Texas")

# Add region column
house_data <- house_data %>%
  mutate(
    region = case_when(
      state %in% northeast ~ "Northeast",
      state %in% midwest ~ "Midwest",
      state %in% west ~ "West",
      state %in% south ~ "South",
      TRUE ~ "Other"
    )
  )

# One-hot encode region and status
house_data <- house_data %>%
  dummy_cols(select_columns = c("region", "status"), remove_selected_columns = TRUE)

# Frequency encode high-cardinality features
city_freq <- house_data %>%
  count(city, name = "city_freq")
house_data <- house_data %>%
  left_join(city_freq, by = "city") %>%
  mutate(city = NULL)  # Drop original column

# Select relevant features
features <- house_data %>%
  select(-street, -state, -brokered_by, -zip_code, -h_id, -X)

# Train-test split
trainIndex <- createDataPartition(features$price, p = 0.8, list = FALSE)
train_data <- features[trainIndex, ]
test_data <- features[-trainIndex, ]

# Define tuning grid
tune_grid <- expand.grid(
  mtry = c(2, 4, 6),       # Number of variables randomly sampled at each split
  splitrule = "variance",  
  min.node.size = c(1, 5, 10)
)

# Train Random Forest model with cross-validation and grid search
control <- trainControl(method = "cv", number = 10, search = "grid")

rf_model <- train(price ~ ., data = train_data,
                  method = "ranger",
                  trControl = control,
                  tuneGrid = tune_grid,
                  num.trees = 1000,
                  importance = "impurity")

# Feature importance
importance <- varImp(rf_model, scale = FALSE)
print(importance)
plot(importance, top = 5, main = "Feature Importance")

# Evaluate on test data
predictions <- predict(rf_model, newdata = test_data)
rmse <- sqrt(mean((test_data$price - predictions)^2))
mae <- mean(abs(predictions - test_data$price))
r_squared <- 1 - sum((test_data$price - predictions)^2) / sum((test_data$price - mean(test_data$price))^2)

results <- data.frame(
  RMSE = rmse,
  MAE = mae,
  R_Squared = r_squared
)
print(results)

rf2 <- train(price ~ ., data = train_data, 
             method = "rf", 
             trControl = control, 
             tuneGrid = expand.grid(mtry = c(2, 4, 6)),
             num.trees = 1000,
             verbose = FALSE)
rfPred2 <- predict(rf2, test_data)
rfRMSE2 <- RMSE(rfPred2, test_data$price)

rmse2 <- sqrt(mean((test_data$price - rfPred2)^2))
mse2 <- rmse*rmse
mae2 <- mean(abs(rfPred2 - test_data$price))
r_squared2 <- 1 - sum((test_data$price - rfPred2)^2) / sum((test_data$price - mean(test_data$price))^2)

results2 <- rbind(
  data.frame(
    RMSE = rmse2,
    MAE = mae2,
    MSE = mse2,
    R_Squared = r_squared2
  )
)

