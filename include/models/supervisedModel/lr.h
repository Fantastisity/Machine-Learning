#ifndef MODEL_UTIL_INCLUDED
#define MODEL_UTIL_INCLUDED
#include "../../utils/modelUtil.h"
#endif

namespace MACHINE_LEARNING {
    class LogisticRegression : public SupervisedModel<LogisticRegression> {
            double loss();
            Matrix<double> gradient(Matrix<double>& X, Matrix<double>& Y);
        public:
            LogisticRegression();

            template<typename T, typename R>
            void fit(T&& x, R&& y, const uint8_t verbose = 0) {
                if constexpr (ModelUtil::isDataframe<typename std::remove_reference<T>::type>::val) 
                    this->x = x.values().template asType<double>();
                else this->x = x;

                this->x.addCol(std::vector<double>(x.rowNum(), 1.0).data());

                if constexpr (ModelUtil::isDataframe<typename std::remove_reference<R>::type>::val) 
                    this->y = y.values().template asType<double>();
                else this->y = y;

                if (verbose == 2) print_params(), puts("");
                
                this->w = Matrix<double>(std::vector<std::vector<double>>(this->x.colNum(), std::vector<double>(1, 0.0)));
                gradient_descent();
                if (verbose) print_weights();
            }

            template<typename T>
            Matrix<double> predict(T&& xtest) {
                Matrix<double> tmp{0};
                if constexpr (ModelUtil::isDataframe<typename std::remove_reference<T>::type>::val) 
                    tmp = xtest.values().template asType<double>();
                else tmp = xtest;
                
                size_t nrow = tmp.rowNum();
                tmp.addCol(std::vector<double>(nrow, 1.0).data());
                tmp = ModelUtil::sigmoid(tmp * this->w);
                for (size_t i = 0; i < nrow; ++i) tmp(i, 0) = tmp(i, 0) < 0.5 ? 0 : 1;
                return tmp;
            }
    };
}