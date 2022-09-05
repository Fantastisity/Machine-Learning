#define SUPERVISED
#define TEST_OLS
//#define TEST_LR
//#define TEST_PERCEP
//#define TEST_SVC
//#define TEST_KNNREGRESSOR
//#define TEST_KNNCLASSIFIER
//#define TEST_NN

#define REGRESSION
//#define BINARY_CLASSIFICATION
//#define MULTICLASS_CLASSIFICATION

// #define UNSUPERVISED
// #define CLUSTERING
// #define TEST_KMEANS

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

#ifdef TEST_PERCEP
#ifndef PERCEP_INCLUDED
#define PERCEP_INCLUDED
#include "src/models/supervised/linearModel/perceptron.h"
#endif
#endif

#ifdef TEST_SVC
#ifndef SVC_INCLUDED
#define SVC_INCLUDED
#include "src/models/supervised/SVM/svc.h"
#endif
#endif

#ifdef TEST_KNNREGRESSOR
#ifndef KNNREGRESSOR_INCLUDED
#define KNNREGRESSOR_INCLUDED
#include "src/models/supervised/nnModel/knnRegressor.h"
#endif
#endif

#ifdef TEST_KNNCLASSIFIER
#ifndef KNNCLASSIFIER_INCLUDED
#define KNNCLASSIFIER_INCLUDED
#include "src/models/supervised/nnModel/knnClassifier.h"
#endif
#endif

#ifdef TEST_KMEANS
#ifndef KMEANS_INCLUDED
#define KMEANS_INCLUDED
#include "src/models/unsupervised/clusteringModel/kMeans.h"
#endif
#endif

#ifdef TEST_NN
#ifndef NN_INCLUDED
#define NN_INCLUDED
#include "src/models/supervised/NeuralNetworks/nnClassifier.h"
#endif
#endif

int main() {
    using namespace MACHINE_LEARNING::UTIL_BASE::MODEL_UTIL;
    using namespace MACHINE_LEARNING::UTIL_BASE::MODEL_UTIL::PREPROCESSING;
    using
    MACHINE_LEARNING::Parser, 
    #ifdef TEST_OLS
    MACHINE_LEARNING::LinearRegression, 
    #elif defined TEST_LR
    MACHINE_LEARNING::LogisticRegression,
    #elif defined TEST_PERCEP
    MACHINE_LEARNING::Perceptron,
    #elif defined TEST_KNNREGRESSOR
    MACHINE_LEARNING::KNNRegressor,
    #elif defined TEST_KNNCLASSIFIER
    MACHINE_LEARNING::KNNClassifer,
    #elif defined TEST_SVC
    MACHINE_LEARNING::SVC,
    #elif defined TEST_KMEANS
    MACHINE_LEARNING::KMeans,
    #elif defined TEST_NN
    MACHINE_LEARNING::NeuralNetworkClassifier,
    #endif
    MACHINE_LEARNING::DataFrame,
    MACHINE_LEARNING::elem;

    Parser p(
        #ifdef REGRESSION
            "simple.csv"
        #elif defined BINARY_CLASSIFICATION
            "iris.csv"
        #elif defined MULTICLASS_CLASSIFICATION
            "irisfull.csv"
        #elif defined CLUSTERING
            "clustering.csv"
        #endif
    );
    auto X = p.getX(
        #ifdef REGRESSION
            2, 2
        #elif defined BINARY_CLASSIFICATION || defined MULTICLASS_CLASSIFICATION
            0, 4
        #elif defined CLUSTERING
            0, 9
        #endif
    );

    #ifdef SUPERVISED
    auto Y = p.getY(
        #ifdef REGRESSION
            1
        #elif defined BINARY_CLASSIFICATION || defined MULTICLASS_CLASSIFICATION
            4
        #endif
    );

    #if defined BINARY_CLASSIFICATION || defined MULTICLASS_CLASSIFICATION
        LabelEncoder<elem> encoder(1);
        encoder.fit_transform(Y, 0);
        #if defined TEST_PERCEP || defined TEST_SVC
            if (Y.unique().size() <= 2) for (size_t i = 0, n = Y.rowNum(); i < n; ++i) if (!Y(i, 0)) Y(i, 0) = -1;
        #endif
    #endif

    auto [xtrain, xtest, ytrain, ytest] = train_test_split(X, Y, 0.25, 1, 1);

    #else
        MinMaxScaler<elem> scaler;
        scaler.fit_transform(X);
    #endif

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
        logger("validation set Accuracy:", METRICS::ACCURACY(clf.predict(xtest), ytest.values()));
        logger("CV Accuracy:", cross_validation(clf, xtrain, ytrain, "ACC"));
    
    #elif defined TEST_PERCEP
        Perceptron clf;
        clf.set_eta(0.01);
        clf.fit(xtrain, ytrain, 2);
        logger("validation set Accuracy:", METRICS::ACCURACY(clf.predict(xtest), ytest.values()));
        logger("CV Accuracy:", cross_validation(clf, xtrain, ytrain, "ACC"));
    
    #elif defined TEST_SVC
        SVC clf;
        clf.fit(xtrain, ytrain, 2);
        logger("validation set Accuracy:", METRICS::ACCURACY(clf.predict(xtest), ytest.values()));
        logger("CV Accuracy:", cross_validation(clf, xtrain, ytrain, "ACC"));
    
    #elif defined TEST_KNNREGRESSOR
        KNNRegressor knn;
        knn.set_n_neighbors(10);
        knn.set_algo(NNAlgo::BALLTREE);
        knn.fit(xtrain, ytrain, 2);
        logger("validation set RMSE:", METRICS::RMSE(knn.predict(xtest), ytest.values()));
    
    #elif defined TEST_KNNCLASSIFIER
        KNNClassifer knn;
        knn.set_n_neighbors(10);
        knn.set_algo(NNAlgo::KDTREE);
        knn.fit(xtrain, ytrain, 2);
        auto ypred = knn.predict(xtest);
        Clf_report_dict report = METRICS::classification_report(ypred, ytest.values());
        for (auto& [k, metrics] : report) {
            std::cout << k << ":\n";
            for (auto& [metric, val] : metrics) 
                std::cout << metric << ": " << val << '\n';
        }
        logger("validation set Accuracy:", METRICS::ACCURACY(ypred, ytest.values()));

    #elif defined TEST_KMEANS
        KMeans kmeans;
        kmeans.set_n_clusters(6);
        kmeans.fit(X, KMAlgo::HARTIGAN);
        auto centroids = kmeans._centroids();
        std::cout << centroids;
    
    #elif defined TEST_NN
        NeuralNetworkClassifier NNClf;
        NNClf.set_layers({3, 3});
        NNClf.fit(xtrain, ytrain);
        logger("validation set Accuracy:", METRICS::ACCURACY(NNClf.predict(xtest), ytest.values()));
        logger("CV Accuracy:", cross_validation(NNClf, xtrain, ytrain, "ACC"));
    #endif
    return 0;
}