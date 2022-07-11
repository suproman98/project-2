Analysis of the Online News Popularity Dataset
================
Suprotik Debnath and Michael Lightfoot





-   [Introduction](#introduction)
-   [Data](#data)
-   [Summarizations](#summarizations)
-   [Modeling](#modeling)
    -   [Four Fitted Models](#four-fitted-models)
-   [Comparison](#comparison)
-   [Automation](#automation)

## Introduction

We are working with the Online News Popularity Dataset from the UCI
Machine Learning Repository. You can find more information about this
dataset
[here](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity).

The goal of this analysis is to produce models in order to predict the
number of shares a website gathers. We will break up this analysis by
`data_channel`, which has 6 categories:

-   Lifestyle
-   Entertainment
-   Business
-   Social Media
-   Tech
-   World

We will automate the model selection and analysis report creation across
all data channels.

Looking at the data, after breaking it up by data channel, we are left
with 51 predictive attributes to help us predict the `shares` variable.
These variables have broad groups:

-   Variables referencing various counts of different types of content
    withing the article.
-   Variables referencing the keywords within the article.
-   Variables referencing statistics of shares in Mashable.
-   Variables referencing day of publishing.
-   Variables referencing closeness to LDA of the topics (Latent
    Dirichlet Allocation)
-   Variables referencing content polarity.

We will likely focus heavily on the variables regarding LDA, statistics
of shares, and the counts of different types of content.

We will utilize various methods to model the response, including random
forest, boosted tree, and linear regression models.

## Data

First, we will read in the data. Then, we will subset the data based on
the variable of interest, and then remove those variables from our
analysis. We are going to create a new factored variable called
`weekday` that merges all the `weekday_is` variables and labels articles
by what day they were published. We will also remove the `url` and
`timedelta` variables for the sake of our analysis as those are
non-predictive.

``` r
#Import packages
library(readr)
library(tidyverse)
library(ggplot2)
library(ggpubr)
library(scales)
library(caret)
library(knitr)
library(rmarkdown)
library(dplyr)

#Read in data 
data <- read_csv("OnlineNewsPopularity/OnlineNewsPopularity.csv")
data <- na.omit(as_tibble(data))

#Create new variables and remove non-needed variables
subdata <- data %>%
  mutate(
    weekday = case_when(
      weekday_is_monday == "1" ~ "Monday",
      weekday_is_tuesday == "1" ~ "Tuesday",
      weekday_is_wednesday == "1" ~ "Wednesday",
      weekday_is_thursday == "1" ~ "Thursday",
      weekday_is_friday == "1" ~ "Friday",
      weekday_is_saturday == "1" ~ "Saturday",
      weekday_is_sunday == "1" ~ "Sunday"
    ),
    channel = case_when(
      data_channel_is_lifestyle == "1" ~ "Lifestyle",
      data_channel_is_entertainment == "1" ~ "Entertainment",
      data_channel_is_bus == "1" ~ "Business",
      data_channel_is_socmed == "1" ~ "Social Media",
      data_channel_is_tech == "1" ~ "Tech",
      data_channel_is_world == "1" ~ "World"
    )
  ) %>%
  select(
    -c(
      url,
      timedelta,
      weekday_is_monday,
      weekday_is_tuesday,
      weekday_is_wednesday,
      weekday_is_thursday,
      weekday_is_friday,
      weekday_is_saturday,
      weekday_is_sunday,
      is_weekend,
      data_channel_is_lifestyle,
      data_channel_is_entertainment,
      data_channel_is_bus,
      data_channel_is_socmed,
      data_channel_is_tech,
      data_channel_is_world
    )
  )

#Now, we subset our data based on the channel of interest
data_life <- subdata %>% filter(channel == params$channel)

#Let's make sure our weekday variable is a factor so it can be 
#used in our analysis
data_life$weekday <- as.factor(data_life$weekday)

#Since we have already subset the data, let's remove the channel variable
data_life <- data_life %>% select(-channel)
head(data_life)
```

## Summarizations

We will create some summary statistics and plots to help us understand
our data. To get an idea of our response variable, let’s look at the
distribution of `shares`.

``` r
stats1 <- data_life %>% 
  select(shares) %>% 
  summarise(avg = round(mean(shares)), 
            med = median(shares), 
            sd = round(sd(shares)))
kable(stats1, "simple", 
      col.names = c("Average", "Median", "Std Dev"))
```

| Average | Median | Std Dev |
|--------:|-------:|--------:|
|    3063 |   1400 |   15046 |

This helps us understand the number of shares an article within this
channel needs to be considered above average, and so forth.

Let’s visualize this distribution of `shares`.

``` r
plot1 <- ggplot(data_life, aes(x=shares)) + 
  geom_histogram(alpha = 0.5, aes(y = ..density..), bins = 100, boundary = 0) + 
  geom_density(alpha = 0.5, kernel = "gaussian", position = "stack") +
  ggtitle("Total Distribution of Shares") + 
  xlab("Shares") + 
  ylab("Density") + 
  theme(plot.title = element_text(hjust = 0.5))
plot1
```

![](BusinessAnalysis_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

Looking at the width and height of the peak(s) of this plot can help us
understand the distribution of `shares` a bit better.

Our exploratory data analysis may benefit from a categorical variable
that splits article into categories based on their number of shares
compared to the mean. Let’s create that variable.

``` r
data_life$shares_cat <- ifelse(data_life$shares <= stats1$avg, 
                              "Below Average", 
                              "Above Average")
```

Now, let us look at these same summary statistics of `shares` grouped by
the `weekday` variable.

``` r
stats2 <- data_life %>% 
  group_by(weekday) %>% 
  summarise(avg = round(mean(shares)), 
            med = median(shares), 
            sd = round(sd(shares)))
kable(stats2, "simple",
      col.names = c("Weekday", "Average", "Median", "Std Dev"))
```

| Weekday   | Average | Median | Std Dev |
|:----------|--------:|-------:|--------:|
| Friday    |    2364 |   1400 |    5134 |
| Monday    |    3887 |   1400 |   28313 |
| Saturday  |    4427 |   2600 |   10139 |
| Sunday    |    3544 |   2200 |    4979 |
| Thursday  |    2885 |   1300 |   13138 |
| Tuesday   |    2932 |   1300 |   10827 |
| Wednesday |    2677 |   1300 |    8160 |

Here we are looking to see if certain days have significantly different
statistics than the overall statistics for `shares` that we observed
above. These statistical summaries by `weekday` may benefit from the
context of how many articles with above and below average numbers of
shares were released on each day. Let’s look at that as well.

``` r
stats3 <- table(data_life$weekday, data_life$shares_cat)
kable(stats3, "simple")
```

|           | Above Average | Below Average |
|-----------|--------------:|--------------:|
| Friday    |           148 |           684 |
| Monday    |           234 |           919 |
| Saturday  |            95 |           148 |
| Sunday    |           130 |           213 |
| Thursday  |           189 |          1045 |
| Tuesday   |           205 |           977 |
| Wednesday |           207 |          1064 |

Here we are looking to see if there are vastly different numbers of
articles in either category released on certain days than others. Let’s
look at a plot of `shares` by `weekday` to help us visualize this
relationship better.

``` r
plot2 <- ggplot(data_life, aes(x=weekday, y=shares, fill=shares_cat)) + 
  geom_bar(stat='identity') +  
  ggtitle("Number of Total Shares by Day") +
  xlab("Weekday") + 
  ylab("Shares")  + 
  theme(plot.title = element_text(hjust = 0.5)) + 
  labs(fill = "Shares Category")
plot2
```

![](BusinessAnalysis_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

Again, we are looking for weekdays that have much higher or lower
numbers of shares than other days to help confirm if `weekday` might be
a solid predictor for our models.

Now let’s look at the correlations between our `shares` variable and the
other numerical variables within our dataset.

``` r
cor <- cor((data_life[, unlist(lapply(data_life, is.numeric))]), data_life$shares)   
cor <- data.frame(Correlation=cor)
cor <- cor %>%
  arrange(desc(Correlation))

kable(cor, "simple")
```

|                              | Correlation |
|------------------------------|------------:|
| shares                       |   1.0000000 |
| self_reference_min_shares    |   0.1108296 |
| self_reference_avg_sharess   |   0.1049782 |
| kw_avg_avg                   |   0.0872109 |
| self_reference_max_shares    |   0.0757104 |
| LDA_03                       |   0.0644360 |
| kw_max_avg                   |   0.0545865 |
| num_videos                   |   0.0504925 |
| global_subjectivity          |   0.0478162 |
| num_hrefs                    |   0.0405516 |
| kw_avg_min                   |   0.0384254 |
| kw_min_avg                   |   0.0367765 |
| kw_max_min                   |   0.0362557 |
| global_rate_negative_words   |   0.0309948 |
| kw_avg_max                   |   0.0287217 |
| n_tokens_content             |   0.0279465 |
| num_imgs                     |   0.0275450 |
| num_keywords                 |   0.0256740 |
| abs_title_subjectivity       |   0.0179084 |
| num_self_hrefs               |   0.0177186 |
| max_positive_polarity        |   0.0174736 |
| global_rate_positive_words   |   0.0155627 |
| abs_title_sentiment_polarity |   0.0148878 |
| rate_negative_words          |   0.0143616 |
| title_subjectivity           |   0.0130827 |
| n_tokens_title               |   0.0106735 |
| kw_max_max                   |   0.0097667 |
| avg_positive_polarity        |   0.0084830 |
| n_non_stop_unique_tokens     |   0.0045198 |
| title_sentiment_polarity     |   0.0040846 |
| n_non_stop_words             |   0.0017967 |
| n_unique_tokens              |   0.0000332 |
| LDA_00                       |  -0.0002873 |
| LDA_01                       |  -0.0024604 |
| kw_min_max                   |  -0.0036566 |
| kw_min_min                   |  -0.0037689 |
| global_sentiment_polarity    |  -0.0126511 |
| rate_positive_words          |  -0.0126575 |
| min_positive_polarity        |  -0.0161531 |
| LDA_02                       |  -0.0182624 |
| LDA_04                       |  -0.0230477 |
| max_negative_polarity        |  -0.0266218 |
| average_token_length         |  -0.0267261 |
| min_negative_polarity        |  -0.0606017 |
| avg_negative_polarity        |  -0.0620893 |

Of course, the shares variable has a perfect correlation with itself.
What we are looking for here are the other variables with strong
correlations (positive or negative) as those could be solid predictors
for the `shares` variable.

Let’s visualize the relationship between some of these numerical
variables and `shares`. First, let’s look at `shares` versus the number
of words in the content of each article, colored by the text sentiment
polarity.

``` r
plot3 <- ggplot(data_life, aes(x = n_tokens_content, y = shares, color = global_sentiment_polarity)) + 
  geom_point() + 
  xlab("Words in the Content") + 
  ylab("Shares") + 
  ggtitle("Number of Shares vs Number of Words in the Content")  + 
  labs(color = "Text Sentiment Polarity") +
  theme(plot.title = element_text(hjust = 0.5))
plot3
```

![](BusinessAnalysis_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

Here, we are looking for trends in the scatter plot along with potential
clusters of different clusters to help us decipher if these numerical
variables will be strong predictors of `shares`.

Now let’s visualize the relationship between `shares` and the number of
videos in each article, colored by the text subjectivity.

``` r
plot4 <- ggplot(data_life, aes(x=num_videos, y=shares, color=global_subjectivity)) + 
  geom_point() + 
  ggtitle("Number of Shares vs Number of Videos") + 
  xlab("Videos") + 
  ylab("Shares") + 
  labs(color = "Text Subjectivity") +
  theme(plot.title = element_text(hjust = 0.5))
plot4
```

![](BusinessAnalysis_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

We are looking for similar indicators as the previous plot: trends in
the scatter plot and clusters of certain colors that show dependence of
`shares` on these variables.

Now let’s take a different visualization approach and look at the
distribution of the number of images, colored by if the articles have
above or below the average number of shares.

``` r
plot5 <- ggplot(data_life, aes(x = num_imgs, fill = shares_cat)) + 
  geom_histogram(alpha = 0.5, aes(y = ..density..), bins = 30, boundary = 0) + 
  geom_density(alpha = 0.5, position = "stack") +
  labs(title = "Distributions of the Number of Images per Article within Shares Categories",
       x = "Images",
       y = "Density",
       fill = "Shares Category")
plot5
```

![](BusinessAnalysis_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

To see if the number of images is a significant predictor of `shares`,
we are looking for a large difference in the density curves for the two
categories. Let’s perform this same visualization with the number of
links per article.

``` r
plot6 <- ggplot(data_life, aes(x = num_hrefs, fill = shares_cat)) + 
  geom_histogram(alpha = 0.5, aes(y = ..density..), bins = 30, boundary = 0) + 
  geom_density(alpha = 0.5, position = "stack") +
  labs(title = "Distributions of the Number of Links per Article within Shares Categories",
       x = "Links",
       y = "Density",
       fill = "Shares Category")
plot6
```

![](BusinessAnalysis_files/figure-gfm/unnamed-chunk-12-1.png)<!-- -->

Again, we are looking to see if the density curves of this variable are
vastly different between the different categories of `shares`.

Lastly, a more helpful analysis may be to group our data by these
categories of `shares` and create some summary statistics. Let’s
consider some variables we have yet to explore: the average length of
words in the context as well as the number of keywords.

``` r
stats4 <- data_life %>% 
  group_by(shares_cat) %>% 
  summarise(avg = round(mean(average_token_length), digits = 3), 
            med = median(average_token_length), 
            sd = round(sd(average_token_length), digits = 3))
kable(stats4, "simple", 
      col.names = c("Shares Category", "Average", "Median", "Std Dev"))
```

| Shares Category | Average |   Median | Std Dev |
|:----------------|--------:|---------:|--------:|
| Above Average   |   4.650 | 4.661917 |   0.415 |
| Below Average   |   4.697 | 4.696518 |   0.384 |

We are looking to see if the summary statistics in each category here
are vastly different in order to determine if the average length of
words in an article is a good predictor of `shares`.

``` r
stats5 <- data_life %>% 
  group_by(shares_cat) %>% 
  summarise(avg = round(mean(num_keywords), digits = 3), 
            med = median(num_keywords), 
            sd = round(sd(num_keywords), digits = 3))
kable(stats5, "simple", 
      col.names = c("Shares Category", "Average", "Median", "Std Dev"))
```

| Shares Category | Average | Median | Std Dev |
|:----------------|--------:|-------:|--------:|
| Above Average   |   6.882 |      7 |   1.983 |
| Below Average   |   6.396 |      6 |   1.962 |

Again, here we could claim the number of key words in an article could
be a solid predictor of `shares` if these summary statistics vary
drastically between these categories.

Also, let’s not forget to get rid of that categorical variable we made
before we start building our models!

``` r
data_life <- data_life %>%
  select(-shares_cat)
```

## Modeling

First, let’s create the testing and training data needed to generate our
models. We will be using a 70/30 training and testing split of our data.

``` r
set.seed(123)
train <- sample(1:nrow(data_life), size = nrow(data_life)*0.7)
test <- setdiff(1:nrow(data_life),train)

train_life <- data_life[train, ]
test_life <- data_life[test, ]
```

After splitting the data, we will be creating four unique models: two
linear regression models and two ensemble models.

### Four Fitted Models

A Linear Regression model measures the relationship between a number of
predictors `x` and a dependent variable `y`. In simple linear
regression, only one predictor variable is used. However, in more
complex modeling using multiple linear regression, we use either
polynomial regression to fit higher order terms or focusing on
main/interaction effects and max combinations.

For an accurate comparison, we will utilize 5-fold cross-validation for
each model. We will pick the model with the lowest RMSE when applied to
the test data.

Our first linear regression model is as follows:

E(*Shares*) =
![\\beta_0](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cbeta_0 "\beta_0") +
![\\beta_1 \\times](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cbeta_1%20%5Ctimes "\beta_1 \times")(*Average
Keywords*) +
![\\beta_2 \\times](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cbeta_2%20%5Ctimes "\beta_2 \times")(*Number
of Images*) +
![\\beta_3 \\times](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cbeta_3%20%5Ctimes "\beta_3 \times")(*Number
of Videos*) +
![\\beta_4 \\times](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cbeta_4%20%5Ctimes "\beta_4 \times")(*Number
of
Videos*)![\\times](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ctimes "\times")(*Number
of Images*)

``` r
#Creating train controls that will be applied to all models.
tr_control <- trainControl(method = "cv", number = 5)

#Creating first linear regression model.
mlrfit1 <- train(shares ~ kw_avg_avg + num_videos + num_imgs + num_videos*num_imgs, 
                 data = train_life, 
                 method = "lm", 
                 trControl = tr_control)

#Test model on test data
mlrfit1_pred <- predict(mlrfit1, newdata = test_life)
output1 <- postResample(mlrfit1_pred, test_life$shares)

#Save RMSE for comparison
RMSE1 <- output1[1]
```

Our second linear regression model will include all predictive variables
as only “main effect” terms:

E(*Shares*) =
![\\beta_0](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cbeta_0 "\beta_0") +
![\\beta_1 \\times](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cbeta_1%20%5Ctimes "\beta_1 \times")(*Number
of Words in Title*) +
![\\beta_2 \\times](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cbeta_2%20%5Ctimes "\beta_2 \times")(*Number
of Words in Content*)…

``` r
#Creating second linear regression model.
mlrfit2 <- train(shares ~ ., 
                 data = train_life, 
                 method = "lm", 
                 trControl = tr_control)

#Test model on test data
mlrfit2_pred <- predict(mlrfit2, newdata = test_life)
output2 <- postResample(mlrfit2_pred, test_life$shares)

#Save RMSE for comparison
RMSE2 <- output2[1]
```

Now, we will try a random forest model. This kind of model randomly
selects subsets of predictors from the training data and creates a set
of decision trees for each subset. Then we have a vote for the
prediction of our response variable from each decision tree. The average
of all of those results is the overall prediction of the random forest
model.

``` r
#Create small tune grid around sqrt(ncol) for efficiency
root_ncol = round(sqrt(ncol(train_life)))
start = root_ncol - 2
end = root_ncol + 2

#Create model
rforest <- train(shares ~ ., 
               method = 'rf', 
               trControl =  tr_control, 
               data = train_life, 
               preProcess = c("center", "scale"),
               tuneGrid = expand.grid(mtry=c(start:end)))

#Test model on test data.
rforest_pred <- predict(rforest, newdata = test_life)
output3 <- postResample(rforest_pred, test_life$shares)

#Save RMSE for comparison
RMSE3 <- output3[1]
```

Lastly, we will try a boosted tree model. This model has a very similar
process to the random forest model. The main difference is that the
decision trees are instead grown one after another, with each tree
growing on a modified version of the data. The predictions of our
response variable are then updated as the model progresses. Generally,
this model is slower but more accurate than the random forest model.

``` r
#Create Training model
gbm <- train(shares ~ ., 
               method = 'gbm', 
               trControl = tr_control,
               data = train_life, 
               preProcess = c("center", "scale"),
               tuneGrid = expand.grid(n.trees = c(25,50,100,150,200),
                                     interaction.depth = c(1:4),
                                     shrinkage = 0.1,
                                     n.minobsinnode = 10))
```

``` r
#Test model on test data.
gbm_pred <- predict(gbm, newdata = test_life)
output4 <- postResample(gbm_pred, test_life$shares)

#Save RMSE for comparison
RMSE4 <- output4[1]
```

## Comparison

Now, we need to select the best model by comparing the RMSE of each
model.

``` r
RMSE_list <- c(RMSE1, RMSE2, RMSE3, RMSE4)

RMSE_compare <- function(RMSE_list) {
  if (min(RMSE_list) == RMSE_list[1]) {
    return(paste("We will choose the first linear regression model! The winning RMSE was:", round(min(RMSE_list), digits = 2)))
  } else if (min(RMSE_list) == RMSE_list[2]) {
    return(paste("We will choose the second linear regression model! The winning RMSE was:", round(min(RMSE_list), digits = 2)))
  } else if (min(RMSE_list) == RMSE_list[3]) {
    return(paste("We will choose the random forest model! The winning RMSE was:", round(min(RMSE_list), digits = 2)))
  } else if (min(RMSE_list) == RMSE_list[4]) {
    return(paste("We will choose the boosted tree model! The winning RMSE was:", round(min(RMSE_list), digits = 2)))
  }
}

RMSE_compare(RMSE_list)
```

    ## [1] "We will choose the random forest model! The winning RMSE was: 15762.08"

## Automation

Below, we simply wanted to show the code used to automate the creation
of the documents.

``` r
#Get channels
channels <- unique(na.omit(subdata$channel))

#Create filenames
output_file <- paste0(channels, "Analysis.md")

#Create lists of parameters
params <- lapply(channels, FUN = function(x){list(channel = x)})

#Put into data frame
reports <- tibble(output_file, params)

#Automate creation of documents
apply(reports, MARGIN = 1,
      FUN = function(x){
        params = x[[2]]
        print(params$channel)
        render(input = "ST558_Proj2.Rmd", ,
               output_format = "github_document", 
               output_file = x[[1]], 
               output_options = list(
                    df_print = "default",
                    toc = TRUE,
                    number_sections = FALSE,
                    keep_html = FALSE),
               params = x[[2]])
      })
```
