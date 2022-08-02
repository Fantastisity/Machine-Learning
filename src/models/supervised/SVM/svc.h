#ifndef SVMBASE_INCLUDED
#define SVMBASE_INCLUDED
#include "svmBase.h"
#endif

namespace MACHINE_LEARNING {
    class SVC : public SVM<SVC> {
        public:
            template<typename T, typename R>
            void fit(T&& x, R&& y, const uint8_t verbose = 0) {
                init(std::forward<T>(x), std::forward<R>(y));
                this->train();
            }

            template<typename T>
            Matrix<double> predict(T&& xtest) {
                Matrix<double> tmp{0};
                if constexpr (UTIL_BASE::isDataframe<typename std::remove_reference<T>::type>::val) 
                    tmp = xtest.values().template asType<double>();
                else if constexpr (UTIL_BASE::isMatrix<typename std::remove_reference<T>::type>::val) 
                    tmp = xtest.template asType<double>();
                size_t n = tmp.rowNum();
                Matrix<double> res(n, 1);
                for (size_t i = 0; i < n; ++i) {
                    double ypred = this->f(&tmp(i, 0));
                    if (ypred > 0) res(i, 0) = 1;
                    else if (ypred < 0) res(i, 0) = -1;
                } 
                return res;
            }
    };
}