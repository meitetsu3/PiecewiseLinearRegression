# Piecewise Linear Regression using neural network

![regression1](D:\Documents\GitHub\PiecewiseLinearRegression\regression1.png)

![regression2](D:\Documents\GitHub\PiecewiseLinearRegression\regression2.png)

### Motivation

 Relationships that can be explained by linear regression are limited in practice. Polynomial or other complex machine learning models are hard to explain, and could behave extreme outside of the data range. Piecewise linear regression, with flexible number of segments and break points may work when regression is too simple but patches of regression could express the phases of the relationship.

Some examples of piecewise linear regression applications are linked below:

* [A Tutorial on the Piecewise Regression Approach Applied to Bedload Transport Data](https://www.fs.fed.us/rm/pubs/rmrs_gtr189.pdf)

* [Water-cement ration v.s. compressive strength](https://onlinecourses.science.psu.edu/stat501/node/310/)
* [Piecewise Linear Regression: A Statistical Method for the Analysis of the Relationship between Traffic Signal Parameters and Air Pollutant Emissions](http://atrf.info/papers/2016/files/ATRF2016_Full_papers_resubmission_64.pdf)

### Previous works

`[1]`[A Tutorial on the Piecewise Regression Approach Applied to Bedload Transport Data](https://www.fs.fed.us/rm/pubs/rmrs_gtr189.pdf)

- Break point estimates need to be provided by user
- Use of SAM NLIN, Nonlinear least squares regression

`[2]`[segmented: An R Package to Fit Regression Models with Broken-Line Relationships](https://www.researchgate.net/publication/234092680_Segmented_An_R_Package_to_Fit_Regression_Models_With_Broken-Line_Relationships)

- Break point estimates need to be provided by user
- Iterative linear regression

`[3]`[A Learning Algorithm for Piecewise Linear Regression](https://pdfs.semanticscholar.org/7345/d357145bc19701397cb894d22e28f770513e.pdf)

* Clustering and regression. multi-variables. The line may be disconnected.
* separate gate for each hidden node.

###  Proposed method - Neural Network Model application

  Let's try to use simple neural network with 1 hidden layer with relu activation function. The benefit is that we can remove the manual input and let the data decide the number of segments and breakpoints, with a very simple feed forward network. In comparison with [3], it is quite similar idea, but the gate is simple relu, and there's no clustering etc. It's just summing up output of relus with no bias. 

 For simplicity, let's think about simple regression. In other words, we explain Y using a variable X. Let's also assume we want to consider up to 3 segments. These limits can be extended. We apply L1 regularization to regulate the number of segments.

### Sample data

[Sample data](https://www.fs.usda.gov/rds/archive/Product/RDS-2007-0004)

### 

