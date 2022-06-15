#ifndef MODEL_UTIL_INCLUDED
#define MODEL_UTIL_INCLUDED
#include "../../utils/modelUtil.h"
#endif

namespace MACHINE_LEARNING {
    class LinearRegression : public SupervisedModel<LinearRegression> {
            double loss();
            Matrix<double> gradient(const Matrix<double>& X, const Matrix<double>& Y);
        public:
            LinearRegression();

            template<typename T, typename R>
            void fit(T&& x, R&& y, const uint8_t verbose = 0) {
                init(std::forward<T>(x), std::forward<R>(y));
                
                if (this->t == GDType::None) 
                    this->w = (this->x.trans() * this->x).inverse() * this->x.trans() * this->y;
                else {
                    this->w = Matrix<double>(this->x.colNum(), 1); 
                    gradient_descent();
                }

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
                
                tmp.addCol(std::vector<double>(tmp.rowNum(), 1.0).data());
                return tmp * this->w;
            }
    };
}