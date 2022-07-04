Analysis of the Online News Popularity Dataset
================
Suprotik Debnath and Michael Lightfoot





-   [Introduction](#introduction)
-   [Data](#data)
-   [Summarizations](#summarizations)
-   [Modeling](#modeling)
-   [Comparison](#comparison)

### Introduction

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

### Data

First, we will read in the data. Then, we will subset the data based on
the variable of interest. We are going to create a new categorical
variable called `weekday` that merges all the `weekday_is` variables and
labels articles by what day they were published. We will also remove the
`url` and `timedelta` variables for the sake of our analysis as those
are non-predictive.

``` r
library(readr)
library(tidyverse)

#Read in data 
data <- read_csv("OnlineNewsPopularity/OnlineNewsPopularity.csv")
data <- as_tibble(data)

#Subset data
data_life <- data %>%
  filter(data$data_channel_is_lifestyle == 1) %>% 
  mutate(weekday = case_when(weekday_is_monday == "1" ~ "Monday", 
                             weekday_is_tuesday == "1" ~ "Tuesday",
                             weekday_is_wednesday == "1" ~ "Wednesday", 
                             weekday_is_thursday == "1" ~ "Thursday",
                             weekday_is_friday == "1" ~ "Friday",
                             weekday_is_saturday == "1" ~ "Saturday",
                             weekday_is_sunday == "1" ~ "Sunday")) %>%
  select(-c(url, timedelta, weekday_is_monday, weekday_is_tuesday, weekday_is_wednesday, weekday_is_thursday, weekday_is_friday, weekday_is_saturday, weekday_is_sunday, is_weekend)) 
```

### Summarizations

We will create some summary statistics and plots to help us understand
our data.

``` r
summary(data_life$shares)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##      28    1100    1700    3682    3250  208300

``` r
data_life %>% group_by(weekday) %>% summarise(avg = round(mean(shares)), med = median(shares), var = round(var(shares)))
```

    ## # A tibble: 7 × 4
    ##   weekday     avg   med       var
    ##   <chr>     <dbl> <dbl>     <dbl>
    ## 1 Friday     3026  1500  20608058
    ## 2 Monday     4346  1600 198047570
    ## 3 Saturday   4062  2100  28630516
    ## 4 Sunday     3790  2100  22771281
    ## 5 Thursday   3500  1600  33879694
    ## 6 Tuesday    4152  1500 183452832
    ## 7 Wednesday  3173  1600  31449814

``` r
library(ggplot2)
library(ggpubr)
library(scales)
plot1 <- ggplot(data_life, aes(x = n_tokens_content, y = shares, color = global_sentiment_polarity)) + geom_point()
plot1
```

![](README_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

``` r
plot2 <- ggplot(data_life, aes(x=weekday, y=shares, fill=weekday)) + geom_bar(stat='identity') +  scale_y_continuous(labels = scales::comma) + ggtitle("Number of Shares by Day") + xlab("Weekday") + ylab("Number of Shares")  + theme(legend.position = "none", plot.title = element_text(hjust = 0.5))
plot2
```

![](README_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

``` r
plot3 <- ggplot(data_life, aes(x=kw_avg_avg, y=shares, color=weekday)) + geom_point(stat='identity') + ggtitle("Avg Keyword by Shares") + xlab("Average Keyword") + ylab("Shares") + theme(plot.title = element_text(hjust = 0.5))
plot3
```

![](README_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

``` r
plot4 <- ggplot(data_life, aes(x=shares)) + geom_histogram() + ggtitle("Total Shares Spread") + xlab("Shares") + ylab("Count") + theme(plot.title = element_text(hjust = 0.5))
plot4
```

    ## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.

![](README_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

### Modeling

### Comparison
