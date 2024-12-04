library(randomForest)
library(caret)


house_data_clean <- readRDS("data_processed/house_data_clean.rds")

##Target encoding: Replacing by mean

# Calculate mean price for each city
city_encoding <- house_data_clean %>%
  group_by(city) %>%
  summarize(city_mean_price = mean(price, na.rm = TRUE))

house_data_clean <- house_data_clean %>%
  left_join(city_encoding, by = "city")

# Calculate mean price for each state
state_encoding <- house_data_clean %>%
  group_by(state) %>%
  summarize(state_mean_price = mean(price, na.rm = TRUE))

house_data_clean <- house_data_clean %>%
  left_join(state_encoding, by = "state") %>% 
  select(-city, -state)

##Processing
set.seed(123)  

house_data <- house_data_clean[, c("price", "house_size", "bed", "bath", 
                                   "land_size", "city_mean_price", "state_mean_price")] 

# Split into training and testing sets
trainIndex <- createDataPartition(house_data$price, p = 0.8, list = FALSE)
train_data <- house_data[trainIndex, ]
test_data <- house_data[-trainIndex, ]

# Define tuning grid 
tune_grid <- expand.grid(
  mtry = c(2, 4, 6),       # Number of variables randomly sampled at each split
  splitrule = "variance",  
  min.node.size = c(1, 5, 10)  
)

# Train the Random Forest model with cross-validation and grid search
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
mse <- rmse*rmse
mae <- mean(abs(predictions - test_data$price))
r_squared <- 1 - sum((test_data$price - predictions)^2) / sum((test_data$price - mean(test_data$price))^2)

results <- rbind(
  data.frame(
    RMSE = rmse,
    MAE = mae,
    MSE = mse,
    R_Squared = r_squared
  )
)


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


}