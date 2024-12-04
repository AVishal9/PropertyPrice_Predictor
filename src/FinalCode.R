

library(tidyverse) 
library(dplyr)
library(naniar)
library(ggplot2)
library(fastDummies)
library(corrplot)
library(randomForest)
library(formattable)
library(caret)
library(MASS)
library(lmtest)



#### Do we need the status variable "for_sale", "sold". If yes, how is it related


data <- read.csv("Final Project/realtorData.csv") %>% 
  drop_na()


unique(data$state)
#This returns 53 states, next step drop states outside US.

valid_states <- c(
  "Massachusetts", "Connecticut", "New Jersey", "New York",
  "New Hampshire", "Vermont", "Rhode Island", "Wyoming",
  "Maine", "Pennsylvania", "West Virginia", "Delaware",
  "Ohio", "Maryland", "Virginia", "Colorado",
  "District of Columbia", "North Carolina", "Kentucky", "South Carolina",
  "Tennessee", "Georgia", "Alabama", "Florida",
  "Mississippi", "Texas", "Missouri", "Arkansas",
  "Louisiana", "Indiana", "Illinois", "Michigan",
  "Wisconsin", "Iowa", "Minnesota", "South Dakota",
  "Nebraska", "North Dakota", "Montana", "Idaho",
  "Kansas", "Oklahoma", "New Mexico", "Utah",
  "Nevada", "Washington", "Oregon", "Arizona",
  "California", "Hawaii", "Alaska"
)

clean_data <- data %>%
  filter(state %in% valid_states)


#There is no unique ID for house, so create a composite key and remove duplicates
clean_data <- clean_data %>% 
  mutate(
    h_id = paste(zip_code, street, city, house_size, bed, bath, sep = "_")
  )



#Check how many times a house was sold: Count the number of entries for each house 
clean_data <- clean_data %>% 
  group_by(h_id) %>% 
  mutate(sale_frequency = n()) %>% 
  ungroup()

#Remove duplicate entries
# Keep only the most recent sale based on prev_sold_date
# Convert `prev_sold_date` to Date format
clean_data <- clean_data %>% 
  mutate(prev_sold_date = as.Date(prev_sold_date)) %>% 
  group_by(h_id) %>% 
  arrange(desc(prev_sold_date)) %>%  # Sort by most recent date
  slice(1) %>%  # Keep the first (most recent) record
  ungroup() %>% 
  dplyr::select(h_id, zip_code, street, city, state, house_size,
         acre_lot, bed, bath, price, sale_frequency, status, brokered_by)


#Check if still any duplicates exist
duplicates <- clean_data %>%
  group_by(h_id) %>%
  summarise(count = n()) %>%
  filter(count > 1)

#No duplicates

summary(clean_data$house_size)

summary(clean_data$bed)

summary(clean_data$bath)

#75% of the houses are below 2478 square feet. 
#Outliers, ie; very expensive mansions or data entry mistakes skewing the data. 


house_size_threshold <- quantile(clean_data$house_size, 0.99)
baths_threshold <- quantile(clean_data$bath, 0.99)
beds_threshold <- quantile(clean_data$bed, 0.99)


clean_data <- clean_data %>%
  filter(house_size <= house_size_threshold,
         bath <= baths_threshold,
         bed <= beds_threshold)

#Thus we only consider data upto 99th percentile

ggplot(clean_data, aes(x = house_size)) +
  geom_histogram(binwidth = 100, fill = "blue", color = "white", alpha = 0.7) +
  labs(
    title = "House Size Distribution",
    x = "House Size (sq ft)",
    y = "Count"
  ) +
  theme_bw()


#Now categorize houses based on size, number of bedrooms and baths

categorized_data <- clean_data %>%
  mutate(
    house_size_category = case_when(
      house_size < 1500 ~ "Small",
      house_size >= 1500 & house_size <= 2500 ~ "Medium",
      house_size > 2500 & house_size <= 4000 ~ "Large",
      house_size > 4000 ~ "Extra-Large"
    ),
    bed_category = case_when(
      bed <= 2 ~ "Small",
      bed > 2 & bed <= 4 ~ "Medium",
      bed > 4 ~ "Large"
    ),
    bath_category = case_when(
      bath <= 2 ~ "Small",
      bath > 2 & bath <= 3 ~ "Medium",
      bath > 3 ~ "Large"
    )
  )

# Proportional Stratification
set.seed(123)

sample_size <- 15000

stratified_sample <- categorized_data %>%
  group_by(state, house_size_category, bed_category, bath_category) %>%
  # Sample proportionally from each stratum
  sample_n(size = round(n() * (sample_size / nrow(categorized_data))), 
           replace = FALSE) %>%
  ungroup()

#Check if sample data proportionally represents original data
# Proportions in the original dataset
original_proportions <- categorized_data %>%
  group_by(state, house_size_category, bed_category, bath_category) %>%
  summarise(count_original = n(), .groups = "drop") %>%
  mutate(proportion_original = count_original / sum(count_original))

# Proportions in the sample dataset
sample_proportions <- stratified_sample %>%
  group_by(state, house_size_category, bed_category, bath_category) %>%
  summarise(count_sample = n(), .groups = "drop") %>%
  mutate(proportion_sample = count_sample / sum(count_sample))

#Creating unique id for the each composite key
stratified_sample$h_id <- seq_len(nrow(stratified_sample))

#Remove categorical variables
stratified_sample <- stratified_sample %>% 
  dplyr::select(-house_size_category, -bath_category, -bed_category)

# To ensure unit consistency converting acres into square feet
# and changing the feature name to land_size
stratified_sample <- stratified_sample %>% 
  mutate(land_size = acre_lot *43560) %>%
  dplyr::select(-acre_lot)


# Checking the Distribution of Numeric Variables 
ggplot(data = stratified_sample, aes(x = bed)) + geom_histogram() + ggtitle("Distribution of beds")
ggplot(data = stratified_sample, aes(x = bath)) + geom_histogram() + ggtitle("Distribution of baths")
ggplot(data = stratified_sample, aes(x = house_size)) + geom_histogram() + ggtitle("Distribution of house size")
ggplot(data = stratified_sample, aes(x = land_size)) + geom_histogram() + ggtitle("Distribution of land size")


#Checking Correlation between Numerical Variables

#Correlation Matrix of Numerical Data 
cor_mat_data <- stratified_sample %>% select_if(., is.numeric) # Dataset of numerical variables to be used for the calculation of correlations
cor_mat_data <- cor_mat_data %>% mutate_if(is.integer, as.numeric) # Converting integers to be numeric
str(cor_mat_data) #Only 9 variables included 
cor_mat <- cor(cor_mat_data[,!colnames(cor_mat_data) %in% c("h_id", "brokered_by","street","zip_code")], use="pairwise.complete.obs") #take out unique id and other numerically encoded categorical variables from here


# Finding higher correlations within the correlation matrix of numeric data (i.e., absolute correlation >= 0.95)
high_cor <- data.frame(row=rownames(cor_mat)[row(cor_mat)[upper.tri(cor_mat)]], 
                       col=colnames(cor_mat)[col(cor_mat)[upper.tri(cor_mat)]], 
                       corr=cor_mat[upper.tri(cor_mat)]) # Correlation matrix transformed into vector form
high_cor$corr <- as.numeric(high_cor$corr) # Correlations need to be numeric, not character
order_cor <- order(abs(high_cor$corr), decreasing = TRUE) # Order rows to have the highest absolute values at the top
high_cor <- high_cor[order_cor, ]
high_cor <- high_cor %>% filter(abs(corr) >= 0.95) # No variable is perfectly correlated that is has a correlation of 100%

# Making a heat map of correlations between numeric variable(change label format)
corrplot(cor_mat, method="color",tl.cex = 0.6, tl.col = "black") 

####
# Outlier Detection and Elimination 
####

#Although outliers have been removed before sampling, these steps are undertaken just to double check 
#as the histograms still show a right tail.

# Examining the distribution of non binary numeric variables
summary_df <- as.data.frame.matrix(summary(stratified_sample))  #city and state still have NA values

#Presents summary statistics in a table 
kable(summary_df, format = "html") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"))

# boxplots of numerical non-dummy variables
stratified_sample %>% 
  dplyr::select(bath, bed, house_size, land_size, price) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_boxplot()

# Histograms of numerical non-dummy variables
stratified_sample %>% 
  dplyr::select(bath, bed, house_size, land_size, price) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_histogram()

# The variables house_size, price, land_size seem to have extreme values. They have a right tail
# Our target variable price has a long right tail(the mean is higher than the median). The minimum value seems to be $1 which doesn't seem practical
# Even for house size the mean > than the median thus indicating a longer right tail.
# Land size also has a large range, with the minimium being 0.00 acres and the maximum being 933,273,000  sq. ft. 
# The mean exceeds the median by a lot pointing towards an extreme upper value.

###
#Exploratory Analysis - Ratios 
###

# Price:House Size Ratio - To assess how much square footage can be bought for a dollar
ratio <- stratified_sample%>%
  mutate(ratio_hp = price/house_size)

#Summary statistics of this new variable
summary(ratio$ratio_hp) #The range of ratios vary a lot i.e. from 0.000 to 4487.00 

# Visual Inspection
ratio %>% 
  dplyr::select(ratio_hp) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_boxplot()  #Visual inspection shows there is an outlier around 4000 mark

ggplot(ratio, aes(x = ratio_hp)) +
  geom_bar(color = "black") +
  labs(x = "Price Per Square Feet", y = "Count") +
  theme_minimal() #There are properties that are for 4000/sq feet which seem like are luxury properties

# Grouped by state to analyse the trend 
ratio %>%
  dplyr::select(ratio_hp) %>%
  dplyr::group_by(ratio$state) %>%
  summarise(mean_ratio = mean(ratio_hp)) %>%
  arrange(desc(mean_ratio)) %>%
  ungroup()%>%
  print(n = 51)

# Hawaii has the highest mean ratio. It is evident that on average the price/sq foot in hawaii is $825/sq feet.
# From visual inspection we can see there are some outliers with $4000/sq feet properties that are skewing the distribution.
# These seem to be properties in states like California and New York.

#We create a new dataset without these luxury properties 
outliers <- function(df) {
  ratio %>%
    group_by(state,city,zip_code) %>%
    filter(ratio_hp > mean(ratio_hp) - sd(ratio_hp) & 
             ratio_hp <= mean(ratio_hp) + sd(ratio_hp)) %>%
    ungroup()
}

data_outlier <- outliers(stratified_sample)

#Visual inspection after outlier removal 
data_outlier %>% 
  dplyr::select(bath, bed, house_size, land_size, price) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_histogram()

summary(data_outlier)

# The variables still have a right tail, that mean they would need to be log transformed. 
# Therefore, it is unnecessary to take out the outliers since the outliers represents the data as well as real life scenario.

# Bathroom:Bedroom Ratio - To inspect how many bathrooms are there per bedroom. 2 bathrooms more than bedrooms is considered acceptable.

ratiob <- stratified_sample%>%
  mutate(ratio_bb = bath/bed)

#Summary statistics of this new variable
summary(ratiob$ratio_bb) #There are maximum three bathrooms per bedroom which is acceptable

# Visual Inspection
ratiob %>% 
  dplyr::select(ratio_bb) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_bar()  


#######
# One-Hot Encoding of Categorical Variables
#######

#As the dataset has high-cardinality categorical variables, it was difficult to one hot encode them all variables, as the computational power
#wasn't sufficient. 

#Summarising unique values of each variable in the dataset
col_no_unique <- data.frame(stratified_sample[,-1] %>% summarise(across(everything(), n_distinct)) %>% pivot_longer(everything()))

#Creating dummy variables for categorical variables

# Dummy variables for status column 
status <- unique(stratified_sample$status) #Two unique variables "sold" and "for_sale"
status_dummies <- dummy_cols(stratified_sample, select_columns = 'status')[,(ncol(stratified_sample)+1):(ncol(stratified_sample)+length(unique(stratified_sample$status)))]
status_dummies <- status_dummies[,-ncol(status_dummies)] #Reducing one column to avoid multicollinearity 

#Dummy Variable for state column 
state <- unique(stratified_sample$state) #51 unique values
state_dummies <- dummy_cols(stratified_sample, select_columns = 'state')[,(ncol(stratified_sample)+1):(ncol(stratified_sample)+length(unique(stratified_sample$state)))]
state_dummies <- state_dummies[,-ncol(state_dummies)] #Reducing one column to avoid multicollinearity 


#Since city, street, and zip code, brokered_by have extremely high unique values, it is better to frequency encode them to reduce the
#high dimensionality

#Grouping Variables for dimensionality reduction

# Grouping zip code below frequency 10
zip_code_stats <- stratified_sample %>%
  group_by(zip_code) %>%
  summarise(count = n()) %>%
  arrange(desc(count))

# Filter zip code with occurrences <= 9
zip_stats_less_than_11 <- zip_code_stats %>%
  filter(count <= 11) %>%
  pull(zip_code)

# Replace zip codes with 'other' if they occur <= 9 times
stratified_sample$zip_code <- ifelse(stratified_sample$zip_code %in% zip_stats_less_than_11 , 'other', stratified_sample$zip_code)

# Get the number of unique zip_codes
length(unique(stratified_sample$zip_code))

#Grouping cities Below frequency 60
city_stats <- stratified_sample %>%
  group_by(city) %>%
  summarise(count = n()) %>%
  arrange(desc(count))

# Filter cities with occurrences <= 60
city_stats_less_than_60 <- city_stats %>%
  filter(count <= 60) %>%
  pull(city)

# Replace city names with 'other' if they occur <= 60 times
stratified_sample$city <- ifelse(stratified_sample$city %in% city_stats_less_than_60, 'other', stratified_sample$city)

# Get the number of unique locations
length(unique(stratified_sample$city))

# Street below frequency 1
street_stats <- stratified_sample %>%
  group_by(street) %>%
  summarise(count = n()) %>%
  arrange(desc(count))

# Filter streets with occurrences <= 1
street_stats_less_than_1<- street_stats %>%
  filter(count <= 1) %>%
  pull(street)

# Replace street with 'other' if they occur <= 1 times
stratified_sample$street <- ifelse(stratified_sample$street %in% street_stats_less_than_1, 'other', stratified_sample$street)

# Get the number of unique street
length(unique(stratified_sample$street))

# Grouping Brokered by below frequency 50
broker_stats <- stratified_sample %>%
  group_by(brokered_by) %>%
  summarise(count = n()) %>%
  arrange(desc(count))

# Filter broker with occurrences <= 50
broker_stats_less_than_50<- broker_stats %>%
  filter(count <= 50) %>%
  pull(brokered_by)

# Replace broker with 'other' if they occur <= 50 times
stratified_sample$brokered_by <- ifelse(stratified_sample$brokered_by %in% broker_stats_less_than_50, 'other', stratified_sample$brokered_by)

# Get the number of unique brokers
length(unique(stratified_sample$brokered_by))

# Dummy Variable for city column 
city <- unique(stratified_sample$city) #4000 unique Variables before grouping.After grouping 14
city_dummies <- dummy_cols(stratified_sample, select_columns = 'city')[,(ncol(stratified_sample)+1):(ncol(stratified_sample)+length(unique(stratified_sample$city)))]
city_dummies <- city_dummies[,-ncol(city_dummies)] #Reducing one column to avoid multicollinearity 

#Dummy Variable for brokered_by 
brokered_by <- unique(stratified_sample$brokered_by) #13816 unique values. After grouping 5 
broker_dummies <- dummy_cols(stratified_sample, select_columns = 'brokered_by')[,(ncol(stratified_sample)+1):(ncol(stratified_sample)+length(unique(stratified_sample$brokered_by)))]
broker_dummies <- broker_dummies[,-ncol(broker_dummies)] 

#Dummy Variable for street
street <- unique(stratified_sample$street) #28412 unique values. 4 after grouping 
street_dummies <- dummy_cols(stratified_sample, select_columns = 'street')[,(ncol(stratified_sample)+1):(ncol(stratified_sample)+length(unique(stratified_sample$street)))]
steet_dummies <- street_dummies[,-ncol(street_dummies)]  #too large for my laptop to run. We want to keep the analysis state level so we will exclude streets.

#Dummy Variable for zip_code
zip_code <- unique(stratified_sample$zip_code) #7000 unique values. 9 after grouping 
zipcode_dummies <- dummy_cols(stratified_sample, select_columns = 'zip_code')[,(ncol(stratified_sample)+1):(ncol(stratified_sample)+length(unique(stratified_sample$zip_code)))]
zipcode_dummies <- zipcode_dummies[,-ncol(zipcode_dummies)]


#Removing the categorical columns and attaching the binary variable columns 
data2 <- subset(stratified_sample, select = -c(status, state, city, brokered_by, street, zip_code))
data2 <- cbind(data2, state_dummies, status_dummies,city_dummies, broker_dummies,street_dummies,zipcode_dummies)


###
#Estimating the models 
###

# Splitting data set into test, train and validation
library(caret)

set.seed(124)


# Creating an index to split the dataset into training dataset (70%)
Index <- createDataPartition(
  y = data2$price, 
  p = .70, # The percentage of data in the training set
  list = FALSE # The format of the results
)

#splitting into training and test dataset using the Index created above
train_data <- data2[Index,]
test_data  <- data2[-Index,]


######
#Estimating Linear Regression model
#####

# Estimating the Linear Regression model, to establish a base level performance
# We will begin by building a linear regression model using Forward selection 

#Excluding unique_ID as it is not needed for estimating the model
train_data <- train_data  %>%
  dplyr::select(-"h_id") # customer_ID is excluded for the model estimation, since it is not needed for the model

# Forward Selection of Regressors

# Setting up a variable 'regressors' that contains all regressors except the dependent variable
names(train_data) <- gsub(" ", "_", names(train_data))  # Replacing spaces in column names with underscores
regressors <- names(train_data)[names(train_data)!="price"]

# Beginning with a null model i.e. with only intercept
current_model <- lm(as.formula(paste("price ~ 1")), data = train_data) # intercept-only model
selected_regressors <- c() # Empty list to store the selected predictor
remaining_regressors <- regressors # Regressors that still need to be evaluated
best_model <- current_model
best_rss <- sum(residuals(best_model)^2) # Compute RSS for the intercept-only model

# Forward selection process
for (i in seq_along(regressors)) {
  # Evaluate each remaining predictor
  potential_models <- lapply(remaining_regressors, function(pred) {
    lm(as.formula(paste("price ~", paste(c(selected_regressors, pred), collapse = "+"))), 
       data = train_data)
  })
  
  # Calculate RSS for each candidate model
  rss_values <- sapply(potential_models, function(model) {
    sum(residuals(model)^2)
  })
  
  # Find the predictor that gives the lowest RSS
  best_model_index <- which.min(rss_values)
  
  # Select the predictor with the lowest RSS if it improves the model
  if (length(rss_values) > 0 && rss_values[best_model_index] < best_rss) {
    selected_regressors <- c(selected_regressors, remaining_regressors[best_model_index])
    remaining_regressors <- remaining_regressors[-best_model_index]
    best_model <- potential_models[[best_model_index]]
    best_rss <- rss_values[best_model_index]
    print(paste("Added predictor:", selected_regressors[length(selected_regressors)]))
  } else {
    break # Stops if there is no improvement in RSS
  }
}

# Final features
feature_selection <- summary(best_model)

# Final model using balanced dataset 
# The R-squared of the best model is 35%
# The forward selection process found a model with 82 regressors, more than half of which aren't statistically significant
# They also do not add a lot to the model theoretically. Therefore, the regressors which aren't satistically significant will be removed
# As most of those are categorical variables, therefore by removing them the model will be simplified and overfitting reduced.


#  bath                        
#  state_California            
#  house_size                
#  bed                        
#  state_Hawaii               
#  state_Florida                
#  brokered_by_16829
#  state_New_York              
#  state_Washington            
#  land_size                   
#  state_Massachusetts         
#  status_for_sale             
#  state_Arizona              
#  state_Oregon                
#  state_Maine                 
# brokered_by_22611           
#  state_Nevada                
#  state_District_of_Columbia  
#  state_Idaho                 
#  city_Sacramento             
#  zip_code_28461             
#  state_Colorado               
# state_New_Jersey            
# state_Utah                    
# state_Rhode_Island          
# state_Connecticut          
# city_Orlando               
# city_Dallas                 
# zip_code_92584             
# state_New_Hampshire         
# state_North_Carolina        
# state_Montana               
# brokered_by_53016          
# state_Virginia                 
# state_Maryland                  
# state_Pennsylvania             
# state_Texas                   
# state_Iowa                     
# state_Delaware                  
# state_Wisconsin                 
# state_New_Mexico                
# city_Chicago                   
# state_South_Carolina            
# state_Tennessee                 
# state_Arkansas                 
# state_Michigan                 
# city_Phoenix                  
# state_Minnesota             
# zip_code_34473                
# state_Vermont              
#state_Nebraska             
#city_Philadelphia           
#state_Alaska                
#city_Saint_Louis              
#state_Illinois                 
#state_Georgia                 
#street_1212806                 
#state_Oklahoma                
#state_Ohio                   
#state_South_Dakota             
#zip_code_77493                
#state_Kentucky                
#zip_code_23112              
#state_North_Dakota              
#city_other                     
#city_Houston                
#city_Richmond                  
#city_Fort_Worth               
#city_San_Antonio              
#city_Albuquerque            
#state_Mississippi           
#zip_code_78155              
#state_Missouri             
#sale_frequency              
#street_other               
#state_Louisiana            
#state_Indiana               
#state_Kansas                
#state_West_Virginia         
#state_Alabama               
#zip_code_79938             
#zip_code_79928             



final_regressors <-c("house_size", "bath","state_California", 
                     "brokered_by_16829", "state_Washington", "state_Florida", "state_Massachusetts","state_Hawaii", "state_Arizona"               
                     ,"state_Oregon", "state_New_York", "bed", "state_District_of_Columbia","land_size",
                     "state_Idaho","state_New_Jersey","state_Nevada","state_Colorado" ,"city_Sacramento"           
                     ,"status_for_sale ","state_Rhode_Island","city_Dallas","brokered_by_22611")

#Model 1: linear-linear relationship 

#Visualizing the relationship
plot(train_data$house_size, train_data$price)
plot(train_data$land_size, train_data$price) #The plots possibly show an Exponential Relationship between the regressors and price

lm1 <- lm(as.formula(paste("price ~", paste(final_regressors, collapse = "+"))), data = train_data)
summary(lm1)

#The residual standard error is 563,600 on 10426 degrees of freedom. 
#On average, the predictions of lm1 model are off by approximately $563,600 from the actual prices.
#The R squared is 34.92%
#There is still room for improvement

# Plotting to check residuals
par(mfrow=c(2,2))
plot(lm1)

#The Q-Q plot still shows that as the values get larger the variance gets larger as well.

# Model 2: Model with Log-linear relationship 
# Visualing the relationships between regressors using log-linear scatterplot.
plot(log(train_data$house_size), log(train_data$price), 
     xlab = "House Size", ylab = "Log of Price", main = "Log(Price) vs House Size") #shows a positive correlation 
plot(train_data$land_size, log(train_data$price), 
     xlab = "Land Size", ylab = "Log of Price", main = "Log(Price) vs Land Size") #however, weak correlation between land size and price,
                                                                                  #there are significant outliers.

lm2 <- lm(as.formula(paste("log(price) ~",paste(final_regressors, collapse = "+"))), data = train_data)
summary(lm2)

#R-squared is 57.18%.
#A RSE of 0.5237 suggests that the predictions deviate by roughly 52.37% on average from the true price values.

# Plotting to check residuals
par(mfrow=c(2,2))
plot(lm2)  #residuals seem to have some outliers. 
#The red line is close to the dotted line. 
#QQ residual plot suggests that the there is variance on either side 

# Model 3: Model with Log-Log relationship 

# Visualing the relationships between regressors using log-log scatterplot.
plot(log(train_data$house_size), log(train_data$price), 
     xlab = "Log of House Size", ylab = "Log of Price", main = "Log-Log Model")
plot(log(train_data$land_size), log(train_data$price), 
     xlab = "Log of Land Size", ylab = "Log of Price", main = "Log-Log Model") 

# From the scatterplots it is evident that land and price share an multiplicative relationship. House size and price also share
# a multiplicative relationship.

final_regressors2 <-c ("bath","state_California", 
                       "brokered_by_16829", "state_Washington", "state_Florida", "state_Massachusetts","state_Hawaii", "state_Arizona"               
                       ,"state_Oregon", "state_New_York", "bed", "state_District_of_Columbia",
                       "state_Idaho","state_New_Jersey","state_Nevada","state_Colorado","city_Sacramento"           
                       ,"status_for_sale ","state_Rhode_Island","city_Dallas","brokered_by_22611")   

lm3 <- lm(as.formula(paste("log(price) ~ log(house_size) + log(land_size+1)+", paste(final_regressors2, collapse = "+"))), data = train_data)
summary(lm3)


#This model is the best out of the 3 models tested,
#The r squared is 58.38% while the residual standard error is 0.5163

# Plotting to check residuals
par(mfrow=c(2,2))
plot(lm3)

# The residuals  appear randomly scattered around 0,
# which indicates that the model captures the linear relationship between predictors and the response variable fairly well.
#There are noticeable outliers (e.g., observations 14804, 2845, 10124), which deviate significantly from the bulk of residuals. 
#These outliers could influence the model’s coefficients or predictions.
#The variance in the residuals seem constant, which means there is no heterosckedasticity
# The Q-Q plot again shows that model may be impacted incase of by extreme values predictions.


#Model 4: Multi linear regressions with house size, land size, baths, beds and sale frequency
lm4 <- lm(log(price)~log(house_size)+land_size + bath + bed, train_data)
summary(lm4)

#The r squared is 35.29% while the residual standard error is 0.6413
# Plotting to check residuals
par(mfrow=c(2,2))
plot(lm4)

#Model 5: house size, land size interaction term
lm5 <- lm(as.formula(paste("log(price)~house_size*land_size+" , paste(final_regressors2, collapse = "+"))), data = train_data)
summary(lm5)

#The r squared is 57.18% while the residual standard error is 0.5237
# Plotting to check residuals
par(mfrow=c(2,2))
plot(lm5)

#Model 6:  land size polynomial term
final_regressors3 <-c ("bath","house_size","state_California", 
                       "brokered_by_16829", "state_Washington", "state_Florida", "state_Massachusetts","state_Hawaii", "state_Arizona"               
                       ,"state_Oregon", "state_New_York", "bed", "state_District_of_Columbia",
                       "state_Idaho","state_New_Jersey","state_Nevada","state_Colorado","city_Sacramento"           
                       ,"status_for_sale ","state_Rhode_Island","city_Dallas","brokered_by_22611")

lm6 <- lm(as.formula(paste("log(price)~land_size+ I(land_size^2)+" , paste(final_regressors3, collapse = "+"))), data = train_data)
summary(lm6)

# The r squared is 57.19% while the residual standard error is 0.5237
# Plotting to check residuals
par(mfrow=c(2,2))
plot(lm6)


#Model 7: house size, land size interaction term and polynomial 
lm7 <- lm(as.formula(paste("log(price)~house_size*land_size+ I(land_size^2)+" , paste(final_regressors2, collapse = "+"))), data = train_data)
summary(lm7)

# The r squared is 57.27% while the residual standard error is 0.5232
# Plotting to check residuals
par(mfrow=c(2,2))
plot(lm7)

# The best model out of the 7 would be the one with highest rsquared and lowest RSE which is lm3 
# The one with R squared of 58.38% and RSE 51.63

library(sandwich)
bptest(lm3) #Reject the null hypothesis, since the breusch-pagan test still suggest that the errors are heterosckedastic, we will then use robust se.
robust_se <- coeftest(lm3, vcov = vcovHC(lm3, type = "HC1"))
print(robust_se)

#####
#Predictions using the fitted model
#####

# Interpretation 1% increase in house_size results in a 0.5% increase in price as a log-log model is used
names(test_data) <- gsub(" ", "_", names(test_data))  # Replacing spaces in column names with underscores
predicted_price <- exp(predict(lm3, newdata = test_data))

# Accuracy of chosen models
actuals <- test_data$price

# Calculating relevant metrics
MAE <- mean(abs(predicted_price - actuals))
MSE <- mean((predicted_price - actuals)^2)
RMSE <- sqrt(MSE)
R2 <- cor(predicted_price, actuals)^2

cat("MAE:", MAE, "\nMSE:", MSE, "\nRMSE:", RMSE, "\nR2:", R2)

# MAE - On average, the model's prediction are off by $187,846.6 in absolute terms 
# RMSE - On average, the model is $433,944.5  off per prediction.
# MSE - MSE squares the differences between predicted and actual values, which disproportionately penalizes large errors. With such a large MSE, it's clear that there might be some significant prediction errors in the model. T
# The large number suggests that outliers could be skewing the results.
# RMSE > MAE suggests that large prediction errors exist, which may be caused by outliers or skewed data.

#Plotting actual vs predicted price

results_df <- data.frame(
  Actual = actuals,
  Predicted = predicted_price
  )

ggplot(results_df, aes(x = Actual, y = Predicted)) +
  geom_point(aes(color = "Predicted"), alpha = 0.6, size = 3) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed", size = 1) +
  labs(title = "Predictions vs Actuals",
       x = "Actual Values",
       y = "Predicted Values") +
  scale_color_manual(values = c("Predicted" = "blue")) +
  theme_minimal() +
  theme(legend.position = "none") 

# As the property value increases the variance increases 


####
# XG boost
####

library(xgboost)

dataxg <- readRDS("house_data.rds")%>%
          drop_na()


#Creating unique id for the each composite key
dataxg$h_id <- seq_len(nrow(dataxg))

#Remove categorical variables
stratified_sample2 <- dataxg %>% 
  dplyr::select(-house_size_category, -bath_category, -bed_category, -h_id, -sale_frequency) #Excluding unimportant features

# To ensure unit consistency converting acres into square feet
# and changing the feature name to land_size
stratified_sample2 <- stratified_sample2 %>% 
  mutate(land_size = acre_lot *43560) %>%
  dplyr::select(-acre_lot)



# label encode categorical variables 
# Extract column names that are not numeric
categorical <- names(stratified_sample2)[sapply(stratified_sample2, function(x) !is.numeric(x))]

# Convert these columns to factor (equivalent of Pandas 'category')
stratified_sample2[categorical] <- lapply(stratified_sample2[categorical], as.factor)
stratified_sample2[categorical] <- lapply(stratified_sample2[categorical], as.integer) #without this I was getting a warning message NAs introduced by coercion


#creating an index to split the dataset into training dataset (70%)
Index <- createDataPartition(
  y = stratified_sample2$price, 
  p = .80, # The percentage of data in the training set
  list = FALSE # The format of the results
)

#splitting into training and test dataset using the Index created above
TrainData <- stratified_sample2[Index,]
TestData  <- stratified_sample2[-Index,]

# Separate the features and target variables 
x_train <- as.matrix(TrainData %>%
                    dplyr::select(-price)) # Exclude target variable

y_train <- TrainData$price

x_test <- as.matrix(TestData %>%
                      dplyr::select(-price))
y_test <- TestData$price

# Convert data into DMatrix, a special format for XGBoost
dtrain <- xgb.DMatrix(data = x_train, label = y_train)
dtest <- xgb.DMatrix(data = x_test, label = y_test) 

###
#Training the XGBoost model using hyperparameters using grid search & CV
###
set.seed(1)
# Define hyperparameters for XGBoost using grid search 
param_grid <- expand.grid(
  max_depth = c(3, 5, 7),
  eta = c(0.01, 0.1, 0.2),
  subsample = c(0.6, 0.8, 1.0),
  colsample_bytree = c(0.6, 0.8, 1.0)
)


# Variables to store results
best_rmse <- Inf
best_params <- NULL

# Loop through parameter grid
for (i in 1:nrow(param_grid)) {
  params <- list(
    objective = "reg:squarederror",
    eta = param_grid[i, "eta"],
    max_depth = param_grid[i, "max_depth"],
    subsample = param_grid[i, "subsample"],
    colsample_bytree = param_grid[i, "colsample_bytree"]
  )
  
# Performing  cross-validation on training dataset
  cv_results <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = 150,
    nfold = 5,
    metrics = "rmse",
    early_stopping_rounds = 10,
    verbose = 0
  )
  
# Tracking the best parameters
  mean_rmse <- min(cv_results$evaluation_log$test_rmse_mean)
  if (mean_rmse < best_rmse) {
    best_rmse <- mean_rmse
    best_params <- params
  }
}

# Best parameters and RMSE
print(best_params) 
# eta - 0.1, max_depth - 5, subsample - 1, colsample_bytree - 0.6 
#Best params after removing unimportant variables -  eta - 0.1, max_depth - 5, subsample - 1, colsample_bytree - 0.8
#Including state and city but converting them to factors 
#eta - 0.1, max_depth - 7, subsample - 1, colsample_bytree - 0.6 
#eta - 0.2, max_depth - 7, subsample - 0.8, colsample_bytree - 0.6 
#eta - 0.1, max_depth - 7, subsample - 1, colsample_bytree - 0.6 


print(best_rmse) 
#476,116.5
#Best RMSE after removing unimportant variables - 403728.1
#Including state and city but converting them to factors -  423923
#439991.7
#444862.2

####
# Training the model using 5 fold cross-validation after Random search
####

set.seed(42)

# Generate random parameter combinations
param_samples <- data.frame(
  max_depth = sample(3:7, 10, replace = TRUE),
  eta = runif(10, 0.01, 0.3),
  subsample = runif(10, 0.6, 1.0),
  colsample_bytree = runif(10, 0.6, 1.0)
)

best_rmse_RS <- Inf
best_params_RS <- NULL

for (i in 1:nrow(param_samples)) {
  params_RS <- list(
    objective = "reg:squarederror",
    eta = param_samples[i, "eta"],
    max_depth = param_samples[i, "max_depth"],
    subsample = param_samples[i, "subsample"],
    colsample_bytree = param_samples[i, "colsample_bytree"]
  )
  
  cv_results_RS <- xgb.cv(
    params_RS = params_RS,
    data = dtrain,
    nrounds = 150,
    nfold = 5,
    metrics = "rmse",
    early_stopping_rounds = 10,
    verbose = 0
  )
  
  mean_rmse <- min(cv_results_RS$evaluation_log$test_rmse_mean)
  if (mean_rmse < best_rmse_RS) {
    best_rmse_RS <- mean_rmse
    best_params_RS <- params_RS
  }
}

#Printing the best parameter from Random Search
print(best_params_RS) 
#eta = 0.2721691, max_depth = 4, subsample = 0.895, colsample_bytree = 0.7518
#Best params after removing unimportant variables -  eta - 0.1440649, max_depth - 3, subsample - 0.632975, colsample_bytree - 0.6015793
#Including state and city but converting them to factors 
#eta - 0.1440649, max_depth - 3, subsample - 0.632975, colsample_bytree - 0.6015793
#eta - 0.05022595, max_depth - 4, subsample - 0.9244221, colsample_bytree - 0.7743086
print(best_rmse_RS) 
#513,969.8
#Best RMSE after removing unimportant variables - 412468.6
#Including state and city but converting them to factors  - 458507.6
#Including state and city but converting them to factors  - 451387.3

###
# Training on the Final Model
###

# The Grid Search had the lower RMSE so we will evaluate the final model using those parameters 
final_params <- list(objective = "reg:squarederror",
                     eta = 0.1,
                     max_depth = 7,
                     subsample = 1,
                     colsample_bytree = 0.6) 

# Train the model with the 72 boosting rounds
set.seed(17)
xgb_model_cv <- xgb.train(
  params = final_params,
  data = dtrain,
  nrounds = 150
)

# Predict on test data
y_pred_cv <- predict(xgb_model_cv, newdata = dtest)

# Calculate evaluation metrics
mae_cv <- mean(abs(y_pred_cv - TestData$price))
mse_cv <- mean((y_pred_cv - TestData$price)^2)
rmse_cv <- sqrt(mse_cv)
r2_cv <- 1 - (sum((TestData$price - y_pred_cv)^2) / sum((TestData$price - mean(TestData$price))^2))

# Print evaluation metrics
cat("Mean Absolute Error (MAE):", mae_cv, "\n") #179813.3  #3rd:173344.4 
cat("Mean Squared Error (MSE):", mse_cv, "\n")              #3rd:241426162964 
cat("Root Mean Squared Error (RMSE):", rmse_cv, "\n") #413160.5 #3rd:491351.4 
cat("R-squared (R2):", r2_cv, "\n") #53.22%  #3rd:54.47%

#checking the important features in the model 
importance <- xgb.importance(model = xgb_model_cv) #bath, house size, zip_code, street, land_size, bed)


#Final XGBoost model 

#param (eta = 0.1,
#max_depth = 7,
#subsample = 1,
#colsample_bytree = 0.6) #Found using grid search 

#MAE:164636.4 
#MSE:173634154264 
#RMSE:416694.3 
#R^2:0.625686 

#Number of iterations - 150




  







