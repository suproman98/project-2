Analysis of the Online News Popularity Dataset
==============================================
**Suprotik Debnath and Michael Lightfoot**


### This Repository

In this repository is our with the Online News Popularity Dataset from the UCI
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

We automated the model selection and analysis report creation across
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

We utilized various methods to model the response, including random
forest, boosted tree, and linear regression models. You can find the analysis
of each data channel below:

- [Analysis of Entertainment articles](EntertainmentAnalysis.html)
- [Analysis of Business articles](BusinessAnalysis.html)
- [Analysis of Tech articles](TechAnalysis.html)
- [Analysis of Lifestyle articles](LifestyleAnalysis.html)
- [Analysis of World articles](WorldAnalysis.html)
- [Analysis of Social Media articles](Social MediaAnalysis.html)

Below was the code we used to automate the creation of our documents. Keep in 
mind that the `subdata` variable is needed for this to function, which we 
create early on in the rmarkdown file within this repository. 

```{r}
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