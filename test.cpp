#ifndef PARSER_INCLUDED
#define PARSER_INCLUDED
#include "include/utils/parser.h"
#endif
#ifndef OLS_INCLUDED
#define OLS_INCLUDED
#include "include/models/supervisedModel/ols.h"
#endif
#ifndef LR_INCLUDED
#define LR_INCLUDED
#include "include/models/supervisedModel/lr.h"
#endif

int main() {
    using namespace MACHINE_LEARNING::UTIL_BASE::MODEL_UTIL;
    using
    MACHINE_LEARNING::Parser, 
    MACHINE_LEARNING::LinearRegression, 
    MACHINE_LEARNING::LogisticRegression,
    MACHINE_LEARNING::DataFrame,
    MACHINE_LEARNING::elem,
    MACHINE_LEARNING::Param;

    Parser p("simple.csv");
    auto X = p.getX(2, 2);
    auto Y = p.getY(1);
    // LabelEncoder<elem> encoder(1);
    // encoder.fit_transform(Y, 0);
    auto [xtrain, xtest, ytrain, ytest] = train_test_split(X, Y, 0.25, 1, 1);
    // LogisticRegression clf;
    // clf.set_eta(1e-3);
    // clf.set_gd_type(GDType::STOCHASTIC);
    // clf.set_iteration(1000);
    // clf.fit(xtrain, ytrain, 2);
    // logger(ACCURACY(clf.predict(xtest), ytest.values()));
    LinearRegression lr;

    // // lr.set_gd_type(GDType::BATCH);
    // // lr.set_eta(1e-6);
    // lr.set_gd_type({static_cast<double>(GDType::BATCH), static_cast<double>(GDType::MINI_BATCH)});
    // lr.set_regularizor({, 
    //                     static_cast<double>(Regularizor::L2), 
    //                     static_cast<double>(Regularizor::ENet)}, {0.1, 0.2});

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
    logger("BGD RMSE:", RMSE(lr.predict(xtest), ytest.values()));
    logger("BGD CV:  ", cross_validation(lr, xtrain, ytrain));

    return 0;
}