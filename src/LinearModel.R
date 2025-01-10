
library(tidyverse) 
library(dplyr)
library(naniar)
library(ggplot2)
library(fastDummies)
library(corrplot)
library(formattable)
library(caret)
library(MASS)
library(lmtest)
library(lubridate)
library(kableExtra)
library(sandwich)



# Reading sampled data
stratified_sample <- read.csv("sampled_data.csv")%>%
  dplyr::select(-X)
view(stratified_sample)

#########
# Model Building & Feature Selection
#########

# Checking Correlation between Numerical Variables

# Correlation Matrix of Numerical Data
cor_mat_data <- stratified_sample %>% select_if(., is.numeric) # Dataset of numerical variables to be used for the calculation of correlations
cor_mat_data <- cor_mat_data %>% mutate_if(is.integer, as.numeric) # Converting integers to be numeric
str(cor_mat_data) # Only 9 variables included
cor_mat <- cor(cor_mat_data[, !colnames(cor_mat_data) %in% c("h_id", "brokered_by", "street", "zip_code")], use = "pairwise.complete.obs") # take out unique id and other numerically encoded categorical variables from here


# Finding higher correlations within the correlation matrix of numeric data (i.e., absolute correlation >= 0.95)
high_cor <- data.frame(
  row = rownames(cor_mat)[row(cor_mat)[upper.tri(cor_mat)]],
  col = colnames(cor_mat)[col(cor_mat)[upper.tri(cor_mat)]],
  corr = cor_mat[upper.tri(cor_mat)]
) # Correlation matrix transformed into vector form
high_cor$corr <- as.numeric(high_cor$corr) # Correlations need to be numeric, not character
order_cor <- order(abs(high_cor$corr), decreasing = TRUE) # Order rows to have the highest absolute values at the top
high_cor <- high_cor[order_cor, ]
high_cor <- high_cor %>% filter(abs(corr) >= 0.95) # No variable is perfectly correlated that is has a correlation of 100%

# Making a heat map of correlations between numeric variable(change label format)
corrplot(cor_mat, method = "color",addCoef.col = "black", tl.cex = 0.6, tl.col = "black")

# Checking the Distribution of Numeric Variables after sampling
ggplot(data = stratified_sample, aes(x = price)) +
  geom_histogram(binwidth = 50, fill = "black", color = "black", alpha = 0.7) +
  ggtitle("Distribution of Target Variable") #Right tail
ggplot(data = stratified_sample, aes(x = bath)) +
  geom_bar(fill = "darkblue", color = "black") +
  ggtitle("Distribution of baths")
ggplot(data = stratified_sample, aes(x = house_size)) +
  geom_histogram(fill = "darkblue", color = "black") +
  ggtitle("Distribution of house size")
ggplot(data = stratified_sample, aes(x = land_size)) +
  geom_histogram(fill = "darkblue", color = "black") +
  ggtitle("Distribution of land size")


#######
# One-Hot Encoding of Categorical Variables
#######

# Summarising unique values of each variable in the dataset
col_no_unique <- data.frame(stratified_sample[, -1] %>% summarise(across(everything(), n_distinct)) %>% pivot_longer(everything()))

# Creating dummy variables for categorical variables
# Dummy variables for status column
status <- unique(stratified_sample$status) # Two unique variables "sold" and "for_sale"
status_dummies <- dummy_cols(stratified_sample, select_columns = "status")[, (ncol(stratified_sample) + 1):(ncol(stratified_sample) + length(unique(stratified_sample$status)))]
status_dummies <- status_dummies[, -ncol(status_dummies)] # Reducing one column to avoid multicollinearity

# Dummy Variable for state column
state <- unique(stratified_sample$state) # 51 unique values
state_dummies <- dummy_cols(stratified_sample, select_columns = "state")[, (ncol(stratified_sample) + 1):(ncol(stratified_sample) + length(unique(stratified_sample$state)))]
state_dummies <- state_dummies[, -ncol(state_dummies)] # Reducing one column to avoid multicollinearity

# Alternative method - Grouping states by region then converting into dummy
#Grouping states by region, make a model where regions are one hot encoded instead of states 
#and see how that performs.

northeast <-c("Connecticut","Maine","Massachusetts","New Hampshire",
              "Rhode Island","Vermont", "New Jersey","New York",
              "Pennsylvania")
midwest <- c("Illinois","Indiana","Michigan","Ohio","Wisconsin",
             "Iowa","Kansas","Minnesota","Missouri",
             "Nebraska","North Dakota","South Dakota")
west <- c("Arizona","Colorado","Idaho","Montana", "Nevada",
          "New Mexico","Utah","Wyoming","Alaska","California",
          "Hawaii","Oregon","Washington")
south <- c("Delaware", "Florida", "Georgia","Maryland",
           "North Carolina","South Carolina","Virginia",
           "West Virginia","District of Columbia",
           "Alabama","Kentucky","Mississippi",
           "Tennessee","Arkansas","Louisiana",
           "Oklahoma", "Texas")

#New dataset with region column instead of state
data_regional <- stratified_sample %>%
  mutate(region = case_when(
    state %in% northeast ~ "Northeast",
    state %in% midwest ~ "Midwest",
    state %in% west ~ "West",
    state %in% south ~ "South"))%>%
  dplyr::select(zip_code, street, city,h_id, house_size,bath,bed,price,sale_frequency,status,brokered_by,land_size, region)

# Dummy Variable for Region column
region <- unique(data_regional$region) # 51 unique values
region_dummies <- dummy_cols(data_regional, select_columns = "region")[, (ncol(data_regional) + 1):(ncol(data_regional) + length(unique(data_regional$region)))]
region_dummies <- region_dummies[, -ncol(region_dummies)]


# Grouping Variables based on frequency for Dimensionality reduction

# Grouping zip code below frequency 11
zip_code_stats <- stratified_sample %>%
  group_by(zip_code) %>%
  summarise(count = n()) %>%
  arrange(desc(count))

# Filter zip code with occurrences <= 11
zip_stats_less_than_11 <- zip_code_stats %>%
  filter(count <= 11) %>%
  pull(zip_code)

# Replace zip codes with 'other' if they occur <= 11 times
stratified_sample$zip_code <- ifelse(stratified_sample$zip_code %in% zip_stats_less_than_11, "other", stratified_sample$zip_code)

# Get the number of unique zip_codes
length(unique(stratified_sample$zip_code))

# Grouping cities Below frequency 70
city_stats <- stratified_sample %>%
  group_by(city) %>%
  summarise(count = n()) %>%
  arrange(desc(count))

# Filter cities with occurrences <= 60
city_stats_less_than_60 <- city_stats %>%
  filter(count <= 60) %>%
  pull(city)

# Replace city names with 'other' if they occur <= 60 times
stratified_sample$city <- ifelse(stratified_sample$city %in% city_stats_less_than_60, "other", stratified_sample$city)

# Get the number of unique locations
length(unique(stratified_sample$city))

# Street below frequency 1
street_stats <- stratified_sample %>%
  group_by(street) %>%
  summarise(count = n()) %>%
  arrange(desc(count))

# Filter streets with occurrences <= 1
street_stats_less_than_1 <- street_stats %>%
  filter(count <= 1) %>%
  pull(street)

# Replace street with 'other' if they occur <= 1 times
stratified_sample$street <- ifelse(
  stratified_sample$street %in% street_stats_less_than_1,
  "other",
  stratified_sample$street
)

# Get the number of unique street
length(unique(stratified_sample$street))

# Grouping Brokered by below frequency 50
broker_stats <- stratified_sample %>%
  group_by(brokered_by) %>%
  summarise(count = n()) %>%
  arrange(desc(count))

# Filter broker with occurrences <= 50
broker_stats_less_than_50 <- broker_stats %>%
  filter(count <= 50) %>%
  pull(brokered_by)

# Replace broker with 'other' if they occur <= 50 times
stratified_sample$brokered_by <- ifelse(
  stratified_sample$brokered_by %in% broker_stats_less_than_50,
  "other",
  stratified_sample$brokered_by
)

# Get the number of unique brokers
length(unique(stratified_sample$brokered_by))

# Dummy Variable for city column
city <- unique(stratified_sample$city) 
city_dummies <- dummy_cols(stratified_sample, select_columns = "city")[, (ncol(stratified_sample) + 1):(ncol(stratified_sample) + length(unique(stratified_sample$city)))]
city_dummies <- city_dummies[, -ncol(city_dummies)] # Reducing one column to avoid multicollinearity

# Dummy Variable for brokered_by
brokered_by <- unique(stratified_sample$brokered_by) 
broker_dummies <- dummy_cols(stratified_sample, select_columns = "brokered_by")[, (ncol(stratified_sample) + 1):(ncol(stratified_sample) + length(unique(stratified_sample$brokered_by)))]
broker_dummies <- broker_dummies[, -ncol(broker_dummies)]

# Dummy Variable for street
street <- unique(stratified_sample$street) 
street_dummies <- dummy_cols(stratified_sample, select_columns = "street")[, (ncol(stratified_sample) + 1):(ncol(stratified_sample) + length(unique(stratified_sample$street)))]
steet_dummies <- street_dummies[, -ncol(street_dummies)] # too large for my laptop to run. We want to keep the analysis state level so we will exclude streets.

# Dummy Variable for zip_code
zip_code <- unique(stratified_sample$zip_code)
zipcode_dummies <- dummy_cols(stratified_sample, select_columns = "zip_code")[, (ncol(stratified_sample) + 1):(ncol(stratified_sample) + length(unique(stratified_sample$zip_code)))]
zipcode_dummies <- zipcode_dummies[, -ncol(zipcode_dummies)]


# Removing the categorical columns and attaching the binary variable columns
data_final <- subset(stratified_sample, select = -c(status, state, city, brokered_by, street, zip_code))
data_final <- cbind(data_final, state_dummies, status_dummies, city_dummies, broker_dummies, street_dummies, zipcode_dummies)


# Data with regional dummies 
data_regional_final <- subset(data_regional,select = -c(status, region, city, brokered_by, street, zip_code))
data_regional_final <- cbind(data_regional_final, region_dummies, status_dummies, city_dummies, broker_dummies, street_dummies, zipcode_dummies)

#####
# Estimating the models
####

# Splitting data set into test, train and validation for data_final
set.seed(124)

# Creating an index to split the dataset into 70-30 split 
Index <- createDataPartition(
  y = data_final$price,
  p = .70, # The percentage of data in the training set
  list = FALSE # The format of the results
)

# splitting into training and test dataset using the Index created above
train_data <- data_final[Index, ]
test_data <- data_final[-Index, ]


######
# Building Linear Regression model
#####

# Excluding unique_ID as it is not needed for estimating the model
train_data <- train_data %>%
  dplyr::select(-"h_id")

# Forward Selection of Regressors

# Setting up a variable 'regressors' that contains all regressors except the dependent variable
names(train_data) <- gsub(" ", "_", names(train_data)) # Replacing spaces in column names with underscores
regressors <- names(train_data)[names(train_data) != "price"]

# Beginning with a null model i.e. with only intercept
current_model <- lm(as.formula(paste("price ~ 1")), data = train_data) 
selected_regressors <- c() 
remaining_regressors <- regressors 
best_model <- current_model
best_rss <- sum(residuals(best_model)^2) # Compute RSS for the intercept-only model

# Forward selection process
for (i in seq_along(regressors)) {
  # Evaluate each remaining predictor
  potential_models <- lapply(remaining_regressors, function(pred) {
    lm(as.formula(paste("price ~", paste(
      c(selected_regressors, pred), collapse = "+"
    ))),
    data = train_data)
  })
  
  # Calculate RSS for each candidate model
  rss_values <- sapply(potential_models, function(model) {
    sum(residuals(model) ^ 2)
  })
  
  # Find the predictor that gives the lowest RSS
  best_model_index <- which.min(rss_values)
  
  # Select the predictor with the lowest RSS if it improves the model
  if (length(rss_values) > 0 &&
      rss_values[best_model_index] < best_rss) {
    selected_regressors <-
      c(selected_regressors, remaining_regressors[best_model_index])
    remaining_regressors <- remaining_regressors[-best_model_index]
    best_model <- potential_models[[best_model_index]]
    best_rss <- rss_values[best_model_index]
    print(paste("Added predictor:", selected_regressors[length(selected_regressors)]))
  } else {
    break 
  }
}

# Final features
feature_selection <- summary(best_model)

final_regressors <- c(
  "house_size",
  "bath",
  "state_California",
  "brokered_by_16829",
  "state_Washington",
  "state_Florida",
  "state_Massachusetts",
  "state_Arizona",
  "state_Colorado",
  "state_Nevada",
  "state_Idaho",
  "state_New_Jersey",
  "city_Sacramento",
  "state_District_of_Columbia",
  "state_Connecticut",
  "state_Rhode_Island", 
  "state_Utah", 
  "city_Dallas", 
  "state_Montana", 
  "bed",   
  "status_for_sale", 
  "brokered_by_22611", 
  "sale_frequency", 
  "city_Portland", 
  "state_New_Hampshire", 
  "zip_code_92336", 
  "zip_code_32404", 
  "zip_code_34491",
  "city_Charlotte", 
  "zip_code_85614",
  "city_Fort_Worth",
  "city_Saint_Louis",
  "city_other",
  "city_Houston",
  "city_Phoenix"
)


# Model 1: linear-linear relationship

# Visualizing the relationship
plot(train_data$house_size, train_data$price,
     xlab = "House Size (sq ft)",
     ylab = "Price ($)",
     main = "Relationship between House Size and Price")

lm1 <- lm(as.formula(paste(
  "price ~", paste(final_regressors, collapse = "+")
)), data = train_data)
summary(lm1)

# Plotting to check residuals
par(mfrow = c(2, 2))
plot(lm1)


# Model 2: Model with Log-linear relationship

# Visualing the relationships between regressors using log-linear scatterplot.
plot((train_data$house_size),
     log(train_data$price),
     xlab = "House Size",
     ylab = "Log of Price",
     main = "Log-Linear Relationship"
) # shows a positive correlation

lm2 <-lm(as.formula(paste(
  "log(price) ~", paste(final_regressors, collapse = "+")
)), data = train_data)
summary(lm2)

#  Plotting to check residuals
par(mfrow = c(2, 2))
plot(lm2)

# Model 3: Model with Log-Log relationship

# Visualing the relationships between regressors using log-log scatterplot.
plot(log(train_data$house_size),
     log(train_data$price),
     xlab = "Log of House Size",
     ylab = "Log of Price",
     main = "Log-Log Relationship"
)


#Final Regressor without Land_size
final_regressors2 <- c(
  "bath",
  "state_California",
  "brokered_by_16829",
  "state_Washington",
  "state_Florida",
  "state_Massachusetts",
  "state_Arizona",
  "state_Colorado",
  "state_Nevada",
  "state_Idaho",
  "state_New_Jersey",
  "city_Sacramento",
  "state_District_of_Columbia",
  "state_Connecticut",
  "state_Rhode_Island", 
  "state_Utah", 
  "city_Dallas", 
  "state_Montana", 
  "bed",   
  "status_for_sale", 
  "brokered_by_22611", 
  "sale_frequency", 
  "city_Portland", 
  "state_New_Hampshire", 
  "zip_code_92336", 
  "zip_code_32404", 
  "zip_code_34491",
  "city_Charlotte", 
  "zip_code_85614",
  "city_Fort_Worth",
  "city_Saint_Louis",
  "city_other",
  "city_Houston",
  "city_Phoenix"
)

lm3 <-lm(as.formula(
  paste(
    "log(price) ~ log(house_size)+",
    paste(final_regressors2, collapse = "+")
  )
), data = train_data)

summary(lm3)

# Plotting to check residuals
par(mfrow = c(2, 2))
plot(lm3)


# Model 4: Multi-linear regressions with house size, land size, baths, beds and sale frequency

lm4 <- lm(log(price) ~ log(house_size) + state_California + bath + bed, train_data)
summary(lm4)

# Plotting to check residuals
par(mfrow = c(2, 2))
plot(lm4)

# Model 5: house size, land size interaction term
lm5 <- lm(as.formula(paste("log(price)~house_size*land_size+", paste(final_regressors2, collapse = "+"))), data = train_data)
summary(lm5)

# Plotting to check residuals
par(mfrow = c(2, 2))
plot(lm5)

# Bp test to check if the Residuals are Homosckedatsic
bptest(lm3) 
lm3_se <- coeftest(lm3, vcov = vcovHC(lm3, type = "HC1"))
print(lm3_se)
# The errors are heterosckedastic 

# Splitting data set into test, train and validation for data_regional_final
set.seed(125)

# Creating an index to split the dataset into 70-30 split 
Index_regional <- createDataPartition(
  y = data_regional_final$price,
  p = .70, # The percentage of data in the training set
  list = FALSE # The format of the results
)

# splitting into training and test dataset using the Index created above
train_regional_data <- data_regional_final[Index_regional, ]
test_regional_data <- data_regional_final[-Index_regional, ]

train_regional_data <- train_regional_data %>%
  dplyr::select(-"h_id")

# Forward Selection of Regressors
# Setting up a variable 'regressors' that contains all regressors except the dependent variable
names(train_regional_data) <- gsub(" ", "_", names(train_regional_data)) # Replacing spaces in column names with underscores
regressors <- names(train_regional_data)[names(train_regional_data) != "price"]

# Beginning with a null model i.e. with only intercept
current_model <- lm(as.formula(paste("price ~ 1")), data = train_regional_data) # intercept-only model
selected_regressors <- c() # Empty list to store the selected predictor
remaining_regressors <- regressors # Regressors that still need to be evaluated
best_model <- current_model
best_rss <- sum(residuals(best_model)^2) # Compute RSS for the intercept-only model

# Forward selection process
for (i in seq_along(regressors)) {
  # Evaluate each remaining predictor
  potential_models <- lapply(remaining_regressors, function(pred) {
    lm(as.formula(paste("price ~", paste(
      c(selected_regressors, pred), collapse = "+"
    ))),
    data = train_regional_data)
  })
  
  # Calculate RSS for each candidate model
  rss_values <- sapply(potential_models, function(model) {
    sum(residuals(model) ^ 2)
  })
  
  # Find the predictor that gives the lowest RSS
  best_model_index <- which.min(rss_values)
  
  # Select the predictor with the lowest RSS if it improves the model
  if (length(rss_values) > 0 &&
      rss_values[best_model_index] < best_rss) {
    selected_regressors <-
      c(selected_regressors, remaining_regressors[best_model_index])
    remaining_regressors <- remaining_regressors[-best_model_index]
    best_model <- potential_models[[best_model_index]]
    best_rss <- rss_values[best_model_index]
    print(paste("Added predictor:", selected_regressors[length(selected_regressors)]))
  } else {
    break # Stops if there is no improvement in RSS
  }
}

# Final features
feature_selection2 <- summary(best_model)

regressors_regional <- c(
  "region_Midwest",
  "region_South",
  "region_Northeast",
  "bath",
  "brokered_by_16829",
  "brokered_by_22611",
  "city_other",
  "bed",
  "zip_code_85614",
  "brokered_by_71243",
  "city_Phoenix",
  "zip_code_85338",
  "zip_code_85375",
  "city_Charlotte",
  "sale_frequency",
  "city_Orlando",
  "zip_code_85710",
  "city_Dallas",
  "city_Saint_Louis",
  "city_Houston",
  "city_Fort_Worth",
  "city_Richmond",
  "city_Portland",
  "city_Sacramento",
  "city_San_Antonio"
)

# Model 6: Replacing states with regional Dummies

lm6 <-lm(as.formula(
  paste(
    "log(price) ~ log(house_size)+",
    paste(regressors_regional, collapse = "+")
  )
), data = train_regional_data)

summary(lm6)

# Bp test to check if the Residuals are Homosckedatsic
bptest(lm6) 
robust_se2 <- coeftest(lm6, vcov = vcovHC(lm6, type = "HC1"))


#####
# Predictions using the fitted model
#####
# Model 3 is the chosen model
names(test_data) <- gsub(" ", "_", names(test_data)) # Replacing spaces in column names with underscores
predicted_price <- exp(predict(lm3, newdata = test_data))

# Adjust intervals using robust variance-covariance
robust_interval <- predict(lm3, newdata = test_data, interval = "confidence", se.fit = TRUE)

robust_predictions <- data.frame(
  fit = robust_interval$fit,
  lwr = robust_interval$fit - 1.96 * sqrt(diag(lm3_se)),
  upr = robust_interval$fit + 1.96 * sqrt(diag(lm3_se))
)

# Accuracy of chosen models
actuals <- test_data$price

# Calculating relevant metrics
MAE <- mean(abs(predicted_price - actuals))
MSE <- mean((predicted_price - actuals)^2)
RMSE <- sqrt(MSE)
R2 <- cor(predicted_price, actuals)^2

cat("MAE:", MAE, "\nMSE:", MSE, "\nRMSE:", RMSE, "\nR2:", R2)


# Plotting actual vs predicted price
results_df <- data.frame(
  Actual = actuals,
  Predicted = predicted_price
)

ggplot(results_df, aes(x = Actual, y = Predicted)) +
  geom_point(aes(color = "Predicted"), alpha = 0.6, size = 3) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed", size = 1) +
  labs(
    title = "Predictions vs Actuals",
    x = "Actual Values",
    y = "Predicted Values"
  ) +
  scale_color_manual(values = c("Predicted" = "blue")) +
  theme_minimal() +
  theme(legend.position = "none")


