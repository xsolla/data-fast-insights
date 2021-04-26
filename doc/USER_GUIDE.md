# User Guide
## 1. Data Preparation
### 1.1. Target variable
**Target variable must be represented in such a way that higher values mean better results.** 
Higher values must represent what result is desirable.  
If this is not the case, please invert the values.  

Example: your target variable is fraud rate: 0.01, 0.05, etc.  
In this case you would have to change it to "normal processing rate": 0.99, 0.95, etc. 
### 1.2. Data cleaning
Any kind of values is supported, as well as NaNs - it will be analyzed as a distinct category
### 1.3. Data Grouping

The most important part in preparing data for the analysis by Data Fast Insights is correctly grouping it.  
You must aggregate and group data by such a set of dimensions that it would be meaningful to compare objects to each other.  

#### Example
For example, suppose we have information about sales of some kind of software and following features: 
- sale ID
- sale date
- software ID 
- DRM / platform software is being distributed on
- supported operating systems
- type of software is (system / applied)
- company (that made this software) ID  

And we also have information about sales and conversion rate (successful payments to all payments attempts) for each software entry.  

We want to get insights about how these features affect sales so that companies that sell it might change something in their development process to increase revenue. It might turn out that support of certain DRM or OS is very important, for instance.

#### Example Grouping Options
##### Bad choice of dimensions.
1. First expected option would be to group data by company and software ID. 
Features are, for instance, would be aggregated by their modes.  
This likely won't give us valuable insights.
Some companies are simply way more known and popular than the others, 
their software sales better just because of the brand or general quality of their products - the fact that their software has certain feature might not be defining their success.
2. Same things as in (1) apply if you group only by company ID.

##### Good choice of dimensions / good alternative ways to research
1. Select another metric.  
Choose the ratio metric instead of absolute one. For example, you can look at payment conversion rate and group by software ID.
Even the products from the most successful companies might have low conversion (yet getting high revenue by attracting more customers in total).
2. Find and add the dimension that to some degree removes key differences between entries of "splitting" dimension (such as companies).  
It can be, for example, some kind of packages or packs software belongs to.  
3. You can always group by time periods, say, quarter or months, yet **you will only get seasonal insights with this**:
    1. group by time period
    2. construct [partial combinations](OTHER_FEATURES.md#partial-combinations-combinations-of-a-certain-features-with-others-of-size-2) 
of time period and other features:
        ```
        dmd.construct_partial_combs(selected_feature, consider_base="selected")
        ```

## 2. Feature selection
It's not required at all, and it's better to have a higher amount of synthesized features.

## 3. Other preprocessing notes
If you have categorical features with high cardinality, you might want to join categories into meta-categories
(i.e. make month dimension from day dimension).  
Library will always make a resulting dataframe which shows most influential segments, no matter how much of them there is, 
but plots are getting pretty difficult to look at when there are more than 10 segments (categories) in the feature.  

Another potential problem with high cardinality is that segments is very likely to be small, hence unstable.

## 4. Using the library and getting results
Section is to be added.
For now please take a look at the examples:
- a short example in the [README](../README.md)
- verbose example with detailed descriptions: [VERBOSE_EXAMPLE](VERBOSE_EXAMPLE.md)