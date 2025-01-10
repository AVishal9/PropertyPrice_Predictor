
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
library(lubridate)
library(kableExtra)
library(sandwich)
library(xgboost)



# Reading the data file 
data_og <- read.csv("./Final Project/realtorData.csv", sep = ",")

# getting an overview of the data 
View(data_og)
str(data_og)
#There are 5 numeric variables, 4 variables with class character and 3 of integer class.

##########
#Data Cleaning & Exploratory Data Analysis
#########

####
# Dealing with NAs
####

#Getting a summary of features with missing data 
(colSums(is.na(data_og)) / nrow(data_og))*100

# visually representing distribution of NAs among variables 
gg_miss_var(data_og) +
  labs (x = "Variable Names", 
        y = "Number of missing values") +
  scale_y_continuous(labels = scales::label_number(accuracy = 1))+
  theme_bw()

# Our target variable, price, has NAs, therefore dropping all the rows with NA price.
# Since the variables with largest share of NAs are critical features, instead of removing columns we drop the rows with
#Na values 
data_og2 <- data_og %>%
            drop_na()

# Comparing the variable distribution in original data with data without NAs 

ggplot(data = data_og, aes(x = bed)) + geom_histogram() + ggtitle("Before Dropping NAs")
Beds <- ggplot(data = data_og2, aes(x = bed)) + geom_histogram() + ggtitle("After Dropping NAs")


ggplot(data = data_og, aes(x = bath)) + geom_histogram() + ggtitle("Before Dropping NAs")
Bath <- ggplot(data = data_og2, aes(x = bath)) + geom_histogram() + ggtitle("After Dropping NAs")


ggplot(data = data_og, aes(x = house_size)) + geom_histogram() + ggtitle("Before Dropping NAs")
HouseSize <- ggplot(data = data_og2, aes(x = house_size)) + geom_histogram() + ggtitle("After Dropping NAs")


ggplot(data = data_og, aes(x = acre_lot)) + geom_histogram() + ggtitle("Before Dropping NAs")
Acrelot <- ggplot(data = data_og2, aes(x = acre_lot)) + geom_histogram() + ggtitle("After Dropping NAs")

ggplot(data = data_og, aes(x = price)) + geom_histogram() + ggtitle("Before Dropping NAs")
Price <- ggplot(data = data_og2, aes(x = price)) + geom_histogram() + ggtitle("After Dropping NAs")

# The distribution of the variables doesn't seem to change much after dropping the Nas

#####
#Data Cleaning 
####

#checking the number of unique states 
unique(data_og2$state) #53 unique states

# Only retaining states that are in Mainland USA
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

data_og2 <- data_og2 %>%
  filter(state %in% valid_states)

#Creating a composite key as there is no unique ID for house to remove duplicates. 
data_og2 <- data_og2 %>%
  mutate(
    h_id = paste(
      zip_code,
      street,
      city,
      house_size,
      bed,
      bath,
      acre_lot,
      brokered_by,
      status,
      prev_sold_date,
      price,
      state,
      sep = "_"
    )
  )

duplicates <- data_og2[duplicated(data_og2[,"h_id"]), ]
print(duplicates)
#No duplicates were found 

# Updating the composite key - as we need to perform certain aggregation
data_og2 <- data_og2 %>%
  mutate(
    h_id = paste(
      zip_code,
      street,
      city,
      house_size,
      bed,
      bath,
      acre_lot,
      sep = "_"
    )
  )


# Checking the number of times a house was sold: Count the number of entries for each house
clean_data <- data_og2 %>%
  group_by(h_id) %>%
  mutate(sale_frequency = n()) %>%
  ungroup()


# Keep only the most recent sale based on prev_sold_date
# Convert `prev_sold_date` to Date format
clean_data <- clean_data %>%
  mutate(prev_sold_date = as.Date(prev_sold_date)) %>%
  group_by(h_id) %>%
  arrange(desc(prev_sold_date)) %>% # Sort by most recent date
  dplyr::slice(1) %>% # Keep the most recent record
  ungroup() %>%
  dplyr::select(
    h_id, zip_code, street, city, state, house_size,
    acre_lot, bed, bath, price, sale_frequency, status, brokered_by
  )

##############
#Outlier Detection and Removal
#############

# representing distribution of HOUSE_SIZE VARIABLE
summary(clean_data$house_size)

#Histogram for House size 
ggplot(clean_data, aes(x = house_size)) + 
  geom_histogram(bins = 30, fill = "darkgreen", color = "black", alpha = 0.7) +
  labs(
    title = "Distribution of House Sizes",
    x = "Size (sq. feet)",
    y = "Count"
  ) +
  theme_bw() +
  scale_x_continuous(
    limits = c(0, 7000),
    breaks = seq(0, 7000, by = 1000)
  )

#Performing outlier detection using Inter Quartile Range.

# 1st Quartile - 1362
Q1_housesize <- 1362

# 3rd Quartile - 2484
Q3_housesize <- 2484

# 50% of the data lies within the range 
IQR_housesize<- Q3_housesize - Q1_housesize
lowerbound_house <- Q1_housesize - (1.5*IQR_housesize)  # This means no value needs to be eliminated from the lower end. 
upperbound_house <- Q3_housesize + (1.5*IQR_housesize)

# Eliminating the outliers 
clean_data2 <- clean_data%>%
    filter(house_size<= upperbound_house)

# Checking the distribution after outlier removal
summary(clean_data2$house_size) # The difference between mean and median has reduced.

ggplot(clean_data2, aes(x = house_size)) + 
  geom_histogram(bins = 30, fill = "darkgreen", color = "black", alpha = 0.7) +
  labs(
    title = "Distribution of House Sizes after Outlier Removal",
    x = "House Size (sq. feet)",
    y = "Count"
  ) +
  theme_bw() +
  scale_y_continuous(labels =scales::label_number(accuracy = 1) )+
  scale_x_continuous(
    limits = c(0, 5000),
    breaks = seq(0, 5000, by = 500)
  )

# representing distribution of BED VARIABLE
summary(clean_data2$bed) 

clean_data2 %>%
  dplyr::select(bed) %>%
  gather() %>%
  ggplot(aes(value)) +
  geom_boxplot(fill ="lightgreen", color = "darkgreen", outlier.colour = "maroon") +
  labs(
    title = "Boxplot of Bedroom Counts in a Property  ",
    x = "Number of bedrooms",
    y = "Count",
  )+
  theme_bw()

# Performing an outlier detection using Interquartile range 
# 1st Quartile - 3
Q1_bed <- 3

# 3rd Quartile - 4
Q3_bed <- 4

# 50% of the dataset has bedrooms within the range of
IQR_bed<- Q3_bed - Q1_bed
lowerbound_bed <- Q1_bed - (1.5*IQR_bed)
upperbound_bed <- Q3_bed + (1.5*IQR_bed)

# Eliminating the outlier
clean_data2 <- clean_data2%>%
  filter(bed>= lowerbound_bed & bed<= upperbound_bed)

#checking the distribution after outlier elimination 
summary(clean_data2$bed) 

clean_data2 %>%
  dplyr::select(bed) %>%
  gather() %>%
  ggplot(aes(value)) +
  geom_boxplot(fill ="lightgreen", color = "darkgreen", outlier.colour = "maroon")+
  labs(
    title = "Boxplot of Bedroom Counts in a Property After Outlier Removal",
    x = "Number of bedrooms",
    y = "Count",
  )+
  theme_bw()

# representing distribution of BATH VARIABLE
summary(clean_data2$bath)        

clean_data2 %>%
  dplyr::select(bath) %>%
  gather() %>%
  ggplot(aes(value)) +
  geom_boxplot(fill ="lightblue", color = "darkblue", outlier.colour = "maroon") +
  labs(
    title = "Boxplot of Bathroom Counts in a Property  ",
    x = "Number of Bathrooms",
    y = "Count",
  )+
  theme_bw()


# Performing an outlier detection using Interquartile range 
# 1st Quartile - 2
Q1_bath <- 2

# 3rd Quartile - 3
Q3_bath <- 3

# 50% of the dataset has bedrooms within the range of
IQR_bath<- Q3_bath - Q1_bath
lowerbound_bath <- Q1_bath - (1.5*IQR_bath) # No value will be classified as an outlier.
upperbound_bath <- Q3_bath + (1.5*IQR_bath)

# Eliminating the outlier
clean_data2 <- clean_data2%>%
  filter(bath<= upperbound_bath)

#Checking the distribution.
summary(clean_data2$bath)        

clean_data2 %>%
  dplyr::select(bath) %>%
  gather() %>%
  ggplot(aes(value)) +
  geom_boxplot(fill ="lightblue", color = "darkblue", outlier.colour = "maroon") +
  labs(
    title = "Boxplot of Bathroom Counts in a Property After Outlier Removal",
    x = "Number of Bathrooms",
    y = "Count",
  )+
  theme_bw()

# Bathroom:Bedroom Ratio - 
# To inspect number of  bathrooms are per bedroom. 
ratiobb <- clean_data2 %>%
  mutate(ratio_bb = bath / bed)

# Summary statistics of the ratio new variable
summary(ratiobb$ratio_bb) 

# representing distribution of Acre_lot
# Converting acres into square_feet to ensure uniformity of units
clean_data2 <- clean_data2 %>%
  mutate(land_size = acre_lot * 43560) %>%
  dplyr::select(-acre_lot)

# Viewing the summary statistics 
summary(clean_data2$land_size)

ggplot(clean_data2, aes(x = land_size)) + 
  geom_histogram(bins = 30, fill = "darkgreen", color = "black", alpha = 0.7) +
  labs(
    title = "Distribution of Land Size",
    x = "Size (sq. feet)",
    y = "Count"
  ) +
  theme_bw()

# 1st Quartile - 6098
Q1_land <- 6098

# 3rd Quartile - 18300
Q3_land <- 18300

# 50% of the dataset has land size within the range of
IQR_land<- Q3_land - Q1_land

lowerbound_land <- Q1_land - (1.5*IQR_land) # The Lower bound is less than the minimum value
upperbound_land <- Q3_land + (1.5*IQR_land)

# Eliminating the outlier
clean_data2 <- clean_data2%>%
  filter(land_size<= upperbound_land)

#Checking the distribution.
summary(clean_data2$land_size)        

ggplot(clean_data2, aes(x = land_size)) + 
  geom_histogram(bins = 50,fill = "darkgreen", color = "black", alpha = 0.7) +
  labs(
    title = "Distribution of Land Size After Outlier Removal",
    x = "Size (sq. feet)",
    y = "Count"
  ) +
  theme_bw()
# still has a slight right tail

# Filter out invalid records
clean_data2 <- clean_data2 %>%
  filter(land_size > house_size)   

############
# Target Variable - Price
############
summary(clean_data2$price)

ggplot(clean_data2, aes(x = price)) + 
  geom_histogram(bins = 50,fill = "darkgreen", color = "black", alpha = 0.7) +
  labs(
    title = "Distribution of Target Variable",
    x = "Dollars",
    y = "Count"
  ) +
  theme_bw() +
  scale_x_continuous(labels =scales::label_number(accuracy = 1))+
  scale_y_continuous(labels =scales::label_number(accuracy = 1))

# There are houses with prices ranging from
# 1 dollar - 5,000 dollars houses with such low prices are unlikely
# Therefore, those should be eliminated

clean_data2 <- clean_data2%>%
  filter(price> 10000)

# Checking the minimum price each state should have based on the smallest house_size
minimum_house_size_by_state <- clean_data2 %>%
  group_by(state) %>%
  filter(house_size == min(house_size, na.rm = TRUE)) %>%
  dplyr::select(state, house_size, price) %>%
  arrange(state)

# Checking the minimum price each state has
minimum_price_by_state <- clean_data2 %>%
  group_by(state) %>%
  filter(price == min(price, na.rm = TRUE)) %>%
  dplyr::select(state, house_size, price) %>%
  arrange(state)

# Price:House Size Ratio - To assess the price per square footage for different properties.
ratio_hp <- clean_data2 %>%
  mutate(ratio_hp = price / house_size)

# Summary statistics of Ratio
summary(ratio_hp$ratio_hp) # The range of ratios varies. from 0.000 to 4487.00

# Visual Inspection
ggplot(ratio_hp, aes(x = ratio_hp)) +
  geom_histogram(color = "black") +
  labs(x = "Price Per Square Feet", y = "Count") +
  theme_minimal() # There are properties that are for 4000/sq feet which seem like are luxury properties

# Minimum Price per Square Foot Ratio
ratio_min <- ratio_hp %>%
  group_by(state)%>% # Grouped by state to analyse the trend
  filter(ratio_hp==min(ratio_hp)) %>%
  arrange(state) %>%
  dplyr::select(state, ratio_hp)%>%
  print(n=51)%>%
  ungroup()

# Maximum Price per square foot Ratio.
ratio_max <- ratio_hp %>%
  group_by(state)%>%
  filter(ratio_hp==max(ratio_hp)) %>%
  arrange(state) %>%
  dplyr::select(state, ratio_hp)%>%
  ungroup()

# Average Price per square foot Ratio
ratio_avg <- ratio_hp %>%
  group_by(state)%>%
  summarise(ratio_hp = mean(ratio_hp)) %>%
  arrange(state) %>%
  print(n=51)%>%
  ungroup()

# Outlier Removal for Target Variable
lower_percentile <- quantile(clean_data2$price, 0.01)
upper_percentile <- quantile(clean_data2$price, 0.99)

clean_data2 <- clean_data2%>%
  filter(price>= lower_percentile & price<= upper_percentile)

ggplot(clean_data2, aes(x = price)) + 
  geom_histogram(bins = 50,fill = "darkgreen", color = "black", alpha = 0.7) +
  labs(
    title = "Distribution of Target Variable",
    x = "Dollars",
    y = "Count"
  ) +
  theme_bw()
summary(clean_data2$price)

#######
# Data Sampling 
#######

# Categorizing the houses based on size, number of bedrooms and baths

categorized_data <- clean_data2 %>%
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
  sample_n(
    size = round(n() * (sample_size / nrow(categorized_data))),
    replace = FALSE
  ) %>%
  ungroup()

# 14928 observations were sampled 

# Check if sample data proportionally represents original data
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

# Comparing the proportions in both dataset
comparison <- original_proportions %>%
  inner_join(sample_proportions, by = c("state", "house_size_category", "bed_category", "bath_category"))

comparison <- comparison %>%
  mutate(proportion_difference = abs(proportion_original - proportion_sample))  # The sampled data aligns closely with the original data


# Creating unique id for the each composite key
stratified_sample$h_id <- seq_len(nrow(stratified_sample))

# Remove categorical variables
stratified_sample <- stratified_sample %>%
  dplyr::select(-house_size_category, -bath_category, -bed_category)

# Saving the sampled file
write.csv(stratified_sample, "sampled_data.csv")

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



####
# XG boost
####

#Read the sampled data

stratified_sample <- read.csv("sampled_data.csv", sep = ",")

# Label encode categorical variables
# Extract column names that are not numeric
categorical <- names(stratified_sample)[sapply(stratified_sample, function(x) !is.numeric(x))]

# Convert these columns to factor 
stratified_sample[categorical] <- lapply(stratified_sample[categorical], as.factor)
stratified_sample[categorical] <- lapply(stratified_sample[categorical], as.integer)


# Creating an index to split the dataset into training dataset (70%)
Index <- createDataPartition(
  y = stratified_sample$price,
  p = .70, # The percentage of data in the training set
  list = FALSE 
)

# Splitting into training and test dataset using the Index created above
TrainDataXG <- stratified_sample[Index, ]
TestDataXG <- stratified_sample[-Index, ]

# Separate the features and target variables
x_train <- as.matrix(TrainDataXG %>%
                       dplyr::select(-price)) # Exclude target variable

y_train <- TrainDataXG$price

x_test <- as.matrix(TestDataXG %>%
                      dplyr::select(-price))
y_test <- TestDataXG$price

# Convert data into DMatrix, a special format for XGBoost
dtrain <- xgb.DMatrix(data = x_train, label = y_train)
dtest <- xgb.DMatrix(data = x_test, label = y_test)


###
# Training the XGBoost model using hyperparameters using grid search & CV
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
print(best_rmse)


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

# Printing the best parameter from Random Search
print(best_params_RS)
print(best_rmse_RS)

###
# Training on the Final Model
###

# The Grid Search had the lower RMSE so we will evaluate the final model using those parameters
final_params <- list(
  objective = "reg:squarederror",
  eta = 0.1,
  max_depth = 7,
  subsample = 0.8,
  colsample_bytree = 0.6
)

# Train the model with the 150 boosting rounds

set.seed(17)
xgb_model_cv <- xgb.train(
  params = final_params,
  data = dtrain,
  nrounds = 150
)

# Predict on test data
y_pred_cv <- predict(xgb_model_cv, newdata = dtest)


# Calculate evaluation metrics
mae_cv <- mean(abs(y_pred_cv - TestDataXG$price))
mse_cv <- mean((y_pred_cv - TestDataXG$price)^2)
rmse_cv <- sqrt(mse_cv)
r2_cv <- 1 - (sum((TestDataXG$price - y_pred_cv)^2) / sum((TestDataXG$price - mean(TestDataXG$price))^2))


# Print evaluation metrics
cat("MAE:", mae_cv, "\nMSE:", mse_cv, "\nRMSE:", rmse_cv, "\nR2:", r2_cv)

#MAE: 103769.6 
#MSE: 26226219981 
#RMSE: 161945.1 
#R2: 0.7028739
