#ifndef MODEL_UTIL_INCLUDED
#define MODEL_UTIL_INCLUDED
#include "../../utils/modelUtil.h"
#endif

namespace MACHINE_LEARNING {
    class LinearRegression : public SupervisedModel<LinearRegression> {
            double loss();
            Matrix<double> gradient(Matrix<double>& X, Matrix<double>& Y);
        public:
            LinearRegression();

            template<typename T, typename R>
            void fit(T&& x, R&& y, const uint8_t verbose = 0) {
                if constexpr (UTIL_BASE::isDataframe<typename std::remove_reference<T>::type>::val) 
                    this->x = x.values().template asType<double>();
                else this->x = x;

                this->x.addCol(std::vector<double>(x.rowNum(), 1.0).data());

                if constexpr (UTIL_BASE::isDataframe<typename std::remove_reference<R>::type>::val) 
                    this->y = y.values().template asType<double>();
                else this->y = y;

                if (verbose == 2) print_params(), puts("");
                
                if (this->t == GDType::None) this->w = ((this->x).trans() * this->x).inverse() * (this->x).trans() * this->y; // Closed-form solution
                else {
                    if (t == GDType::SAG) {
                        if (!(seen = (bool*) calloc(x.rowNum(), 1))) {
                            std::cerr << "error calloc\n"; exit(1);
                        }
                        this->gradient_table = Matrix<double>(this->x.rowNum(), this->x.colNum());
                        this->gradient_sum = Matrix<double>(this->x.colNum(), 1);
                    }
                    this->w = Matrix<double>(this->x.colNum(), 1);
                    gradient_descent();
                }
                if (verbose) print_weights();
            }

            template<typename T>
            Matrix<double> predict(T&& xtest) {
                Matrix<double> tmp{0};
                if constexpr (UTIL_BASE::isDataframe<typename std::remove_reference<T>::type>::val) 
                    tmp = xtest.values().template asType<double>();
                else tmp = xtest;
                
                tmp.addCol(std::vector<double>(tmp.rowNum(), 1.0).data());
                return tmp * this->w;
            }
    };
}