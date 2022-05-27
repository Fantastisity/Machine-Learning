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
    MACHINE_LEARNING::DataFrame, MACHINE_LEARNING::Matrix;

    Matrix<double> mat({{2, 7, 9}, {3, 8, 0}, {5, 24, 5}});
    mat.inverse();
    
    // Parser p("simple.csv");
    // auto X = p.getX(2, 2);
    // auto Y = p.getY(1);

    // auto [xtrain, xtest, ytrain, ytest] = modUtil.train_test_split(X, Y);
    // LinearRegression lr;
    // lr.set_eta({1e-6, 1e-7, 1e-8});
    // lr.set_iteration({1000, 200});
    // lr.set_gd_type({BATCH, MINI_BATCH});
    // lr.set_regularizor({L1, L2, ENet}, {0.1, 0.2});

    // auto [best_param, best_err] = modUtil.grid_search(lr, xtrain, ytrain);
    // lr.set_params(best_param);
    // lr.fit(xtrain, ytrain, 2);
    // logger("SGD RMSE:", modUtil.RMSE(lr.predict(xtest), ytest.values()));
    // logger("SGD CV:  ", modUtil.cross_validation(lr, xtrain, ytrain, 10));

    return 0;
}