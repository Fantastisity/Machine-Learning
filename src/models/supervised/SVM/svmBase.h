#ifndef MODELUTIL_INCLUDED
#define MODELUTIL_INCLUDED
#include "../../../utils/modelUtil.h"
#endif

namespace MACHINE_LEARNING {
    template<typename M>
    class SVM {
        protected:
            Matrix<double> x{0}, y{0};
            void SMO(size_t i1, size_t i2) {
                if (i1 == i2) return;
            }
            template<typename T, typename R>
            void init(T&& x, R&& y) {
                if constexpr (UTIL_BASE::isDataframe<typename std::remove_reference<T>::type>::val) 
                    this->x = x.values().template asType<double>();
                else if constexpr (UTIL_BASE::isMatrix<typename std::remove_reference<T>::type>::val) 
                    this->x = x.template asType<double>();
                else this->x = std::forward<T>(x);

                if constexpr (UTIL_BASE::isDataframe<typename std::remove_reference<R>::type>::val) 
                    this->y = y.values().template asType<double>();
                else if constexpr (UTIL_BASE::isMatrix<typename std::remove_reference<R>::type>::val)
                    this->y = y.template asType<double>();
                else this->y = std::forward<T>(y);
            }
        public:
            template<typename T, typename R>
            void fit(T&& x, R&& y, const uint8_t verbose = 0) {
                (static_cast<M*>(this))->fit(std::forward<T>(x), std::forward<R>(y), verbose);
            }

            template<typename T>
            Matrix<double> predict(T&& xtest) {
                return (static_cast<M*>(this))->predict(std::forward<T>(xtest));
            }
    };
}