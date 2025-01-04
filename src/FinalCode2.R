
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



# Reading the data file 
data_og <- read.csv("./Final Project/realtorData.csv", sep = ",")

# getting an overview of the data 
View(data_og)
str(data_og)
#There are 5 numeric variables, 4 variables with class character and 3 of integer class.

##########
#Preliminary Exploratory Data Analysis and Data Cleaning 
#########

####
# Dealing with NAs
####

#Getting a summary of features with missing data 
(colSums(is.na(data_og)) / nrow(data_og))*100

# visually representing distribution of NAs among variables 
gg_miss_var(data_og)

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
#Using all the variables to ensure that the same observation is not included twice (i.e.duplicated)
#If observation has a different price, different land size it would not be considered a duplicate.
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

# As we need to perform certain aggregation removing some variables from h_id that are unsuitable or irrelevant 
# With brokerage in composite key, the maximum number of sales frequency is 3 without it is 4, 
# thus broker is excluded from the composite key, since we are analysing overall property sales trend,
# thus a grouping this granulated is not neccesary.

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
  slice(1) %>% # Keep the first (most recent) record
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

# 75% of the houses are below 2484 square feet.
# However, the maximum value seems to be extreme
#Therefore, performing outlier detection using Inter Quartile Range.

# 1st Quartile - 1362
Q1_housesize <- 1362

# 3rd Quartile - 2484
Q3_housesize <- 2484


# 50% of the data lies within the range 
IQR_housesize<- Q3_housesize - Q1_housesize
lowerbound_house <- Q1_housesize - (1.5*IQR_housesize)  # This means no value needs to be eliminated from the lower end. 
                                    # No value will be classified as an outlier.
upperbound_house <- Q3_housesize + (1.5*IQR_housesize)

# Eliminating the outliers 
clean_data2 <- clean_data%>%
    filter(house_size<= upperbound_house)

# Checking the distribution after outlier removal
summary(clean_data2$house_size) # The difference between mean and median has reduced, 
                               # indicating that distribution is closer to normal now.

ggplot(clean_data2, aes(x = house_size)) + 
  geom_histogram(bins = 30, fill = "darkgreen", color = "black", alpha = 0.7) +
  labs(
    title = "Distribution of House Sizes",
    x = "Size (sq. feet)",
    y = "Count"
  ) +
  theme_bw() +
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

# 75% of the houses have upto 4 bedrooms.
#There seem to be an extreme value of 444 beds.
# Performing an outlier detection using Interquartile range 
# 1st Quartile - 3
Q1_bed <- 3

# 3rd Quartile - 4
Q3_bed <- 4

# 50% of the dataset has bedrooms within the range of
IQR_bed<- Q3_bed - Q1_bed
lowerbound_bed <- Q1_bed - (1.5*IQR_bed)
# No value will be classified as an outlier.
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

#75% of the properties have up to 3 bathrooms, however, there seem to be an extreme value of 175 bathrooms
# Performing an outlier detection using Interquartile range 

# 1st Quartile - 2
Q1_bath <- 2

# 3rd Quartile - 3
Q3_bath <- 3

# 50% of the dataset has bedrooms within the range of
IQR_bath<- Q3_bath - Q1_bath
lowerbound_bath <- Q1_bath - (1.5*IQR_bath) # The lowerbound is less than the minimum value.
                                            # That means no outliers exist on the lower end, therefore, 
                                            # No cut is necessary.

# No value will be classified as an outlier.
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
  theme_bw() +
  scale_x_log10()

# 75% of the properties are built on land <= 18,300 sq.feet.
# There seems to be a right tail, therefore performing an outlier detection using Interquartile range 

# 1st Quartile - 6098
Q1_land <- 6098

# 3rd Quartile - 18300
Q3_land <- 18300

# 50% of the dataset has land size within the range of
IQR_land<- Q3_land - Q1_land

lowerbound_land <- Q1_land - (1.5*IQR_land) # The lowerbound is less than the minimum value.
                                            # That means no outliers exist on the lower end, therefore, 
                                            # No cut is necessary
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

# still has a slight right tail.




#Grouping states by region, make a model where regions are one hot encoded instead of states 
#and see how that performs.

northeast <-c("Connecticut","Maine","Massachusetts","New Hampshire",
              "Rhode Island","Vermont", "New Jersey","New York",
              "Pennsylvania")
midwest <- c("Illinois","Indiana","Michigan","Ohio","Wisconsin",
             "Iowa","Kansas","Minnesota","Missouri",
             "Nebraska","North Dakota","South Dakota")
west <- c("Arizona","Colorado","Idaho"," Montana", "Nevada",
          "New Mexico","Utah","Wyoming","Alaska","California",
          "Hawaii","Oregon","Washington")
south <- c("Delaware", "Florida", "Georgia","Maryland",
           "North Carolina","South Carolina","Virginia",
           "West Virginia","District of Columbia",
           "Alabama","Kentucky","Mississippi",
          "Tennessee","Arkansas","Louisiana",
           "Oklahoma", "Texas")