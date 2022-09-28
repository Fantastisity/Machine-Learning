## lightweight c++ implementation of common ML algorithms
### Sample Usage (partly retrieved from test.cpp)
~~~cpp
#define SUPERVISED // Supervised learning
#define TEST_OLS // Ordinary Least Squares

#ifdef TEST_OLS
#ifndef OLS_INCLUDED
#define OLS_INCLUDED
#include "src/models/supervised/linearModel/ols.h" // Include header
#endif
#endif

int main()
{
    using namespace MACHINE_LEARNING::UTIL_BASE::MODEL_UTIL;
    using namespace MACHINE_LEARNING::UTIL_BASE::MODEL_UTIL::PREPROCESSING;
    using
    MACHINE_LEARNING::Parser, 
    #ifdef TEST_OLS
    MACHINE_LEARNING::LinearRegression,
    #endif
    MACHINE_LEARNING::elem;
    
    Parser p("simple.csv"); // Parse the CSV file to a DataFrame object
    auto X = p.getX(2, 2); // Retrieve X values

    #ifdef SUPERVISED
    auto Y = p.getY(1); // Retrieve Y values

    auto [xtrain, xtest, ytrain, ytest] = train_test_split(X, Y, 0.25, 1, 1); // Split the data with random shuffling
    
    #ifdef TEST_OLS
        LinearRegression ols; // Create an OLS model
        ols.set_gd_type(GDType::BATCH); // Batch gradient descent
        Param param_grid {
            {"eta", {1e-6, 1e-7, 1e-8}}, 
            {"iteration", {200, 500}},
            {"gd_type", {static_cast<double>(GDType::None), static_cast<double>(GDType::BATCH)}},
            {"regularizor", {static_cast<double>(Regularizor::L1), static_cast<double>(Regularizor::L2)}},
            {"alpha", {0.1, 0.3}},
            {"lambda", {0.1, 0.3}}
        };

        auto [best_param, best_err] = grid_search(ols, param_grid, xtrain, ytrain); // Grid search for optimal parameters
        ols.set_params(best_param); // Set the optimal parameters
        ols.fit(xtrain, ytrain, 2); // Fit the model with output
        logger("validation set RMSE:", METRICS::RMSE(ols.predict(xtest), ytest.values()));
        logger("CV RMSE:", cross_validation(ols, xtrain, ytrain));
     #endif
};
~~~
##### Sample Output
![alt text](https://user-images.githubusercontent.com/61994603/192773215-ae799872-4557-4a3a-9a6f-3d2b508b473c.png=250x250)
### Build
```bash
cd {path to current directory}
g++ -w $(find . -type f -iregex '.*\.cpp') -o model -std=c++17 -pthreads -O2
```
