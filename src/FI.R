library(tidyverse)
library(caret)
library(corrplot)
library(GGally)
library(scales)
library(plotly)

# Load the data
house_data <- readRDS("data_processed/house_data.rds")

# Apply log transformations
house_data <- house_data %>%
  mutate(
    log_price = log1p(price),       # Log transformation for price
    log_house_size = log1p(house_size),  # Log transformation for house size
    log_land_size = log1p(land_size)  # Log transformation for land size
  )

# Remove rows where land_size is 0 or land_size is not greater than house_size
house_data <- house_data %>%
  filter(land_size > house_size & price > house_size & city!="")

sum(is.na(house_data$city))

# Check the cleaned data
summary(house_data)


# 1. Handling Missing Values
# Check for missing values in the dataset
sum(is.na(house_data))  # Number of missing values
# For simplicity, let's impute missing values using median for numerical columns
house_data <- house_data %>%
  mutate(across(where(is.numeric), ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))

# 2. Scaling Numerical Features
# Log transformations were already applied for price, house_size, and land_size.
# Now, let's scale the numerical variables for algorithms that require it (e.g., linear regression)
scaled_data <- house_data %>%
  mutate(
    house_size = scale(log_house_size), 
    price = scale(log_price),  # scaling log-transformed price
    land_size = scale(log_land_size)
  )

# 3. Encoding Categorical Variables
# If you have categorical variables, you can encode them:
# One-hot encoding for a categorical variable (e.g., state)
house_data <- house_data %>%
  mutate(across(where(is.factor), as.character)) %>%
  mutate(across(where(is.character), as.factor))

# One-hot encode 'state' using dummyVars from caret package
# Create dummy variables for 'state'
dummies <- dummyVars(~ state, data = house_data)

# Predict the dummy variables
state_dummies <- predict(dummies, newdata = house_data)

# Add the dummy variables to the original dataset without overwriting
house_data <- cbind(house_data, state_dummies)


# 4. Creating New Features
# Create a new feature: Price per Square Foot 
house_data <- house_data %>%
  mutate(price_per_sqft = price / house_size)

# 5. Feature Interactions
# Interaction between house_size and land_size
house_data <- house_data %>%
  mutate(house_land_interaction = log_house_size * log_land_size)

# Check the structure of the dataset after adding the new features
str(house_data)


# 6. Detecting and Removing Outliers
# For example, let's detect and remove outliers using the IQR method for 'price'
Q1 <- quantile(house_data$log_price, 0.25, na.rm = TRUE)
Q3 <- quantile(house_data$log_price, 0.75, na.rm = TRUE)
IQR <- Q3 - Q1
outlier_limit <- 1.5 * IQR
house_data <- house_data %>%
  filter(log_price >= (Q1 - outlier_limit) & log_price <= (Q3 + outlier_limit))

# Inspecting the transformed data
summary(house_data)

# Check for any missing values after transformations
sum(is.na(house_data))  # Make sure no missing values are left

# Visualize any new features
ggplot(house_data, aes(x = price_per_sqft)) +
  geom_histogram(bins = 30, fill = "blue", color = "black") +
  labs(title = "Distribution of Price per Square Foot")


# Select numeric columns for correlation
numeric_cols <- house_data %>%
  select_if(is.numeric) %>%                     
  mutate_if(is.integer, as.numeric) %>%
  select(-c(h_id, street, zip_code)) # Exclude any non-numeric variables like h_id, street, etc.

# Include the new features in the correlation matrix
numeric_cols <- numeric_cols %>%
  select(log_price, log_house_size, log_land_size, price_per_sqft, house_land_interaction, bath, bed, price)

# Compute correlation matrix
cor_matrix <- cor(numeric_cols, use = "pairwise.complete.obs")

# Plot correlation heatmap
corrplot(cor_matrix, method = "color", tl.cex = 0.6, addCoef.col = "white", tl.col = "black")

# Train a linear regression model with log-transformed variables
log_model <- lm(log_price ~ log_house_size + log_land_size + bath + bed + price_per_sqft, data = house_data)

# Summarize the log-transformed model
summary(log_model)

# Predict the log-transformed price
log_predictions <- predict(log_model, house_data)

# Calculate RMSE (Root Mean Squared Error)
log_residuals <- log_predictions - house_data$log_price
log_rmse <- sqrt(mean(log_residuals^2))
cat("Log Model RMSE:", log_rmse, "\n")

# Calculate R-squared
log_r_squared <- summary(log_model)$r.squared
cat("Log Model R-squared:", log_r_squared, "\n")

# Load necessary library
library(glmnet)

# Prepare the data
x <- as.matrix(house_data %>%
                 select(log_house_size, log_land_size, bath, bed, price_per_sqft))
y <- house_data$log_price

# Fit the Lasso regression model
lasso_model <- cv.glmnet(x, y, alpha = 1)

# View the best lambda from cross-validation
best_lambda <- lasso_model$lambda.min
cat("Best lambda for Lasso:", best_lambda, "\n")

# Make predictions
lasso_predictions <- predict(lasso_model, s = "lambda.min", newx = x)

# Calculate RMSE for Lasso model
lasso_residuals <- lasso_predictions - y
lasso_rmse <- sqrt(mean(lasso_residuals^2))
cat("Lasso Model RMSE:", lasso_rmse, "\n")

# Calculate R-squared for Lasso model
lasso_r_squared <- 1 - sum(lasso_residuals^2) / sum((y - mean(y))^2)
cat("Lasso Model R-squared:", lasso_r_squared, "\n")

# Load necessary library for Random Forest
library(randomForest)

# Prepare the data (using original price, not log-transformed)
x_rf <- house_data %>%
  select(house_size, land_size, bath, bed, price_per_sqft)

y_rf <- house_data$price  # Using raw price, not log-transformed

# Train the Random Forest model
rf_model <- randomForest(x = x_rf, y = y_rf, importance = TRUE, ntree = 500)

# Print the summary of the Random Forest model
print(rf_model)

# Make predictions
rf_predictions <- predict(rf_model, newdata = x_rf)

# Calculate RMSE for Random Forest model
rf_residuals <- rf_predictions - y_rf
rf_rmse <- sqrt(mean(rf_residuals^2))
cat("Random Forest Model RMSE:", rf_rmse, "\n")

# Calculate R-squared for Random Forest model
rf_r_squared <- 1 - sum(rf_residuals^2) / sum((y_rf - mean(y_rf))^2)
cat("Random Forest Model R-squared:", rf_r_squared, "\n")



