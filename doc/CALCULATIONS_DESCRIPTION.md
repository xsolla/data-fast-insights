# What does the library calculates and how?
## Data representation
Internally library converts data to certain types which is described below.
  
### Features representation

In order for underlying model to work, every feature, including target variable (Y), must be represented as binary. 
Library converts categorical and numeric features differently.
- categorical feature are converted via one-hot encoding
- numeric features are split into bins so that **Information Value (IV)** is maximum
- binary features are passed as is (as categories)

##### Example:
We had a feature "charge_attempt_count" which had values "1" and "14". 
The library would instead operate on new features:
- charge_attempt_count[-inf, 2), which equals 1 if charge_attempt_count < 2 and equals 0 otherwise
- charge_attempt_count[2, inf), which equals 1 if charge_attempt_count >= 2 and equals 1 otherwise

### Target variable representation
Target variable can be converted to binary in a way that separates it into two intervals:
- separated by its mean value
- separated by one of its quantiles  
- separated in another custom way  

**Target variable must be constructed in a way so that higher values 
represent "better" results (e.g. higher revenue)**  
Binary target variable would represent whether value of target variable is lower than the threshold.
Thus binary target variable equals 1 for objects (rows) that are worse and 0 for those that are better.  

## Calculated metrics
### Feature influence
One of the main metrics that library relies on in its conclusions is how much a certain segment of data 
worsens the target variable.  

To calculate this
- model counts all rows where binary feature equals 1 
(so selects a segment from a feature or features combination) 
and binary target variable equals 1 as well (so all the "bad" objects")
- model divides previous value by all rows where binary feature equals 1 
(just the whole segment)

### Target variable
Another calculated metric is average value of the target variable in each segment for each feature.

### Other metrics
Other important metrics are described in 
 [calculate_dependence() docstring](data_fast_insights/calculations/_modelling.py)
 