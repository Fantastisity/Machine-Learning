#ifndef LINEARMODEL_INCLUDED
#define LINEARMODEL_INCLUDED
#include "linearModel.h"
#endif

namespace MACHINE_LEARNING {
    class LogisticRegression : public LinearModel<LogisticRegression> {
            double loss();
            Matrix<double> gradient(const Matrix<double>& X, const Matrix<double>& Y);
        public:
            LogisticRegression();

            template<typename T, typename R>
            void fit(T&& x, R&& y, const uint8_t verbose = 0) {
                init(std::forward<T>(x), std::forward<R>(y));

                this->w = Matrix<double>(this->x.colNum(), 1);
                gradient_descent();

                if (verbose) {
                    if (verbose == 2) print_params();
                    print_weights();
                }
            }

            template<typename T>
            Matrix<double> predict(T&& xtest) {
                Matrix<double> tmp{0};
                if constexpr (UTIL_BASE::isDataframe<typename std::remove_reference<T>::type>::val) 
                    tmp = xtest.values().template asType<double>();
                else if constexpr (UTIL_BASE::isMatrix<typename std::remove_reference<T>::type>::val) 
                    tmp = xtest.template asType<double>();
                else tmp = std::forward<T>(xtest);
                
                size_t nrow = tmp.rowNum();
                tmp.addCol(std::vector<double>(nrow, 1.0).data());
                tmp = UTIL_BASE::MODEL_UTIL::METRICS::sigmoid(tmp * this->w);
                for (size_t i = 0; i < nrow; ++i) tmp(i, 0) = tmp(i, 0) < 0.5 ? 0 : 1;
                return tmp;
            }
    };
}