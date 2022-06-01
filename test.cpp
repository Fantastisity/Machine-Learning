#ifndef PARSER_INCLUDED
#define PARSER_INCLUDED
#include "include/utils/parser.h"
#endif
#ifndef LINEAR_MODEL_INCLUDED
#define LINEAR_MODEL_INCLUDED
#include "include/models/supervisedModel/ols.h"
#endif

int main() {
    using
    MACHINE_LEARNING::Parser, 
    MACHINE_LEARNING::LinearRegression, 
    MACHINE_LEARNING::modUtil,
    MACHINE_LEARNING::DataFrame;

    Parser p("simple.csv");
    auto X = p.getX(2, 2);
    auto Y = p.getY(1);

    auto [xtrain, xtest, ytrain, ytest] = modUtil.train_test_split(X, Y);
    LinearRegression lr;

    // lr.set_gd_type(GDType::BATCH);
    // lr.set_eta(1e-6);

    lr.set_eta({1e-6, 1e-7, 1e-8});
    lr.set_iteration({1000, 200});
    lr.set_gd_type({static_cast<double>(GDType::BATCH), static_cast<double>(GDType::MINI_BATCH)});
    lr.set_regularizor({static_cast<double>(Regularizor::L1), 
                        static_cast<double>(Regularizor::L2), 
                        static_cast<double>(Regularizor::ENet)}, {0.1, 0.2});

    auto [best_param, best_err] = modUtil.grid_search(lr, xtrain, ytrain);
    lr.set_params(best_param);
    lr.fit(xtrain, ytrain, 2);
    logger("BGD RMSE:", modUtil.RMSE(lr.predict(xtest), ytest.values()));
    logger("BGD CV:  ", modUtil.cross_validation(lr, xtrain, ytrain));

    return 0;
}