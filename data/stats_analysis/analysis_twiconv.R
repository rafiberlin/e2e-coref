
```{r setup, include=FALSE, warning=FALSE, message=FALSE, message=FALSE, echo=FALSE, avel=FALSE }
knitr::opts_chunk$set(echo = TRUE)

set.seed(42)
library(MASS)
##be careful to load dplyr after MASS
library(dplyr)
library(tidyr)
library(purrr)
library(readr)
library(extraDistr)
library(ggplot2)
library(brms)
library(rstan)
## Save compiled models:
rstan_options(auto_write = TRUE)
## Parallelize the chains using all the cores:
options(mc.cores = parallel::detectCores())
library(bayesplot)
library(tictoc)
library(gridExtra)
# To solve some conflicts between packages
select <- dplyr::select
extract <- rstan::extract
base_dir <- "C:/Users/Rafi/Uni_Workspace/Bayesian2/bayescogsci/LOT2020WinterSchoolSlides"

knitr::opts_chunk$set(fig.width=4, fig.height=3) 
```

