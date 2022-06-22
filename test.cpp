//#define TEST_OLS
//#define TEST_LR
#define TEST_KNNREGRESSOR

#define REGRESSION
//#define CLASSIFICATION

#ifndef PARSER_INCLUDED
#define PARSER_INCLUDED
#include "src/utils/parser.h"
#endif

#ifdef TEST_OLS
#ifndef OLS_INCLUDED
#define OLS_INCLUDED
#include "src/models/supervised/linearModel/ols.h"
#endif
#endif

#ifdef TEST_LR
#ifndef LR_INCLUDED
#define LR_INCLUDED
#include "src/models/supervised/linearModel/lr.h"
#endif
#endif

#ifdef TEST_KNNREGRESSOR
#ifndef KNNREGRESSOR_INCLUDED
#define KNNREGRESSOR_INCLUDED
#include "src/models/supervised/nnModel/knnRegressor.h"
#endif
#endif

int main() {
    using namespace MACHINE_LEARNING::UTIL_BASE::MODEL_UTIL;
    using
    MACHINE_LEARNING::Parser, 
    #ifdef TEST_OLS
    MACHINE_LEARNING::LinearRegression, 
    #elif defined TEST_LR
    MACHINE_LEARNING::LogisticRegression,
    #elif defined TEST_KNNREGRESSOR
    MACHINE_LEARNING::KNNRegressor,
    #endif
    MACHINE_LEARNING::DataFrame,
    MACHINE_LEARNING::elem,
    MACHINE_LEARNING::Param;

    Parser p(
        #ifdef REGRESSION
            "simple.csv"
        #elif defined CLASSIFICATION
            "iris.csv"
        #endif
    );
    auto X = p.getX(
        #ifdef REGRESSION
            2, 2
        #elif defined CLASSIFICATION
            0, 4
        #endif
    );
    auto Y = p.getY(
        #ifdef REGRESSION
            1
        #elif defined CLASSIFICATION
            4
        #endif
    );

    #ifdef CLASSIFICATION
        LabelEncoder<elem> encoder(1);
        encoder.fit_transform(Y, 0);
    #endif

    auto [xtrain, xtest, ytrain, ytest] = train_test_split(X, Y, 0.25, 1, 1);

    #ifdef TEST_OLS
        LinearRegression ols;
        ols.set_gd_type(GDType::BATCH);
        Param param_grid {
            {"eta", {1e-6, 1e-7, 1e-8}}, 
            {"iteration", {200, 500}},
            // {"gd_type", {static_cast<double>(GDType::None), static_cast<double>(GDType::BATCH)}},
            {"regularizor", {static_cast<double>(Regularizor::L1), static_cast<double>(Regularizor::L2)}},
            {"alpha", {0.1, 0.3}},
            {"lambda", {0.1, 0.3}}
        };

        auto [best_param, best_err] = grid_search(ols, param_grid, xtrain, ytrain);
        ols.set_params(best_param);
        ols.fit(xtrain, ytrain, 2);
        logger("validation set RMSE:", METRICS::RMSE(ols.predict(xtest), ytest.values()));
        logger("CV RMSE:", cross_validation(ols, xtrain, ytrain));

    #elif defined TEST_LR
        LogisticRegression clf;
        clf.set_eta(1e-3);
        clf.set_gd_type(GDType::SAG);
        clf.set_iteration(1000);
        clf.fit(xtrain, ytrain, 2);
        logger("Accuracy:", METRICS::ACCURACY(clf.predict(xtest), ytest.values()));
        logger("CV Accuracy:", cross_validation(clf, xtrain, ytrain, "ACC"));
    
    #elif defined TEST_KNNREGRESSOR
        KNNRegressor knn;
        knn.fit(xtrain, ytrain, 2);
        // logger("Accuracy:", METRICS::RMSE(knn.predict(xtest), ytest.values()));
    #endif
    return 0;
}