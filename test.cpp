//#define TEST_OLS
#define TEST_LR

#ifndef PARSER_INCLUDED
#define PARSER_INCLUDED
#include "src/utils/parser.h"
#endif

#ifdef TEST_OLS
#ifndef OLS_INCLUDED
#define OLS_INCLUDED
#include "src/models/supervisedModel/ols.h"
#endif
#endif

#ifdef TEST_LR
#ifndef LR_INCLUDED
#define LR_INCLUDED
#include "src/models/supervisedModel/lr.h"
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
    #endif
    MACHINE_LEARNING::DataFrame,
    MACHINE_LEARNING::elem,
    MACHINE_LEARNING::Param;

    Parser p(
        #ifdef TEST_OLS
            "simple.csv"
        #elif defined TEST_LR
            "iris.csv"
        #endif
    );
    auto X = p.getX(
        #ifdef TEST_OLS
            2, 2
        #elif defined TEST_LR
            0, 4
        #endif
    );
    auto Y = p.getY(
        #ifdef TEST_OLS
            1
        #elif defined TEST_LR
            4
        #endif
    );

    #ifdef TEST_OLS
        auto [xtrain, xtest, ytrain, ytest] = train_test_split(X, Y, 0.25, 1, 1);
        LinearRegression lr;
        Param param_grid {
            {"eta", {1e-6, 1e-7, 1e-8}}, 
            {"iteration", {200, 500}},
            {"gd_type", {static_cast<double>(GDType::BATCH), static_cast<double>(GDType::None)}},
            {"regularizor", {static_cast<double>(Regularizor::L1), static_cast<double>(Regularizor::L2)}},
            {"alpha", {0.1, 0.3}},
            {"lambda", {0.1, 0.3}}
        };

        auto [best_param, best_err] = grid_search(lr, param_grid, xtrain, ytrain);
        lr.set_params(best_param);
        lr.fit(xtrain, ytrain, 2);
        logger("validation set RMSE:", RMSE(lr.predict(xtest), ytest.values()));
        logger("CV RMSE:", cross_validation(lr, xtrain, ytrain));

    #elif defined TEST_LR
        LabelEncoder<elem> encoder(1);
        encoder.fit_transform(Y, 0);
        auto [xtrain, xtest, ytrain, ytest] = train_test_split(X, Y, 0.25, 1, 1);
        LogisticRegression clf;
        clf.set_eta(1e-3);
        clf.set_gd_type(GDType::SAG);
        clf.set_iteration(1000);
        clf.fit(xtrain, ytrain, 2);
        logger("Accuracy:", ACCURACY(clf.predict(xtest), ytest.values()));
        logger("CV Accuracy:", cross_validation(clf, xtrain, ytrain, "ACC"));
    #endif
    return 0;
}