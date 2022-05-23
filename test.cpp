#ifndef PARSER_INCLUDED
#define PARSER_INCLUDED
#include "include/utils/parser.h"
#endif
#ifndef LINEAR_MODEL_INCLUDED
#define LINEAR_MODEL_INCLUDED
#include "include/models/supervisedModel/lr.h"
#endif

int main() {
    using 
    MACHINE_LEARNING::Parser, 
    MACHINE_LEARNING::LinearRegression, 
    MACHINE_LEARNING::modUtil,
    MACHINE_LEARNING::DataFrame, MACHINE_LEARNING::Matrix;

    Parser p("simple.csv");
    auto X = p.getX(2, 2);
    auto Y = p.getY(1);

    // ModelUtil::LabelEncoder<elem> le(1);
    // le.fit_transform(Y, 0);
    // std::cout << Y;
    // le.inverse_transform(Y, 0);
    auto [xtrain, xtest, ytrain, ytest] = modUtil.train_test_split(X, Y);
    LinearRegression lr;
    lr.set_eta(1e-6);
    lr.set_iteration(1000);
    lr.set_gd_type(BATCH);
    // modUtil.grid_search(lr, xtrain, ytrain);
    lr.fit(xtrain, ytrain, 2);
    logger("SGD RMSE:", modUtil.RMSE(lr.predict(xtest), ytest.values()));
    logger("SGD CV:  ", modUtil.cross_validation(lr, xtrain.values(), ytrain.values()));
    // test1 += 2;
    // std::cout << test1;
    return 0;
}