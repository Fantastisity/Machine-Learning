#include "linearModel.h"

namespace MACHINE_LEARNING {
    class Perceptron : public LinearModel<Perceptron> {
        public:
            template<typename T, typename R>
            void fit(T&& x, R&& y, const uint8_t verbose = 0) {
                init(std::forward<T>(x), std::forward<R>(y));
                this->w = Matrix<double>(this->x.colNum(), 1); 
                for (size_t i = 0, term = 0; i < this->iter && !term; ++i, term = 0) {
                    for (size_t j = 0, n = this->x.rowNum(); j < n; ++j) {
                        auto x_tmp = this->x(rngSlicer(j, j + 1), rngSlicer(0, this->x.colNum()));
                        if ((this->y(j, 0) * (x_tmp * this->w))(0, 0) < 1) this->w += this->y(j, 0) * x_tmp.trans(), ++term;
                    }
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
                
                size_t nrow = tmp.rowNum();
                tmp.addCol(std::vector<double>(nrow, 1.0).data());

                tmp = tmp * this->w;

                for (size_t i = 0; i < nrow; ++i) tmp(i, 0) = tmp(i, 0) < 0 ? -1 : 1;
                return tmp;
            }
    };
}