#ifndef NNBASE_INCLUDED
#define NNBASE_INCLUDED
#include "nnBase.h"
#endif

namespace MACHINE_LEARNING {
    class NeuralNetworkClassifier : public NNBase<NeuralNetworkClassifier> {
        public:
            template<typename T>
            void fit(T&& x, T&& y, const uint8_t verbose = 0) {
                init(std::forward<T>(x), std::forward<T>(y));
                train();
            }

            template<typename T>
            Matrix<double> predict(T&& xtest) {
                Matrix<double> tmp{0};
                if constexpr (UTIL_BASE::isDataframe<typename std::remove_reference<T>::type>::val) 
                    tmp = xtest.values().template asType<double>();
                else if constexpr (UTIL_BASE::isMatrix<typename std::remove_reference<T>::type>::val) 
                    tmp = xtest.template asType<double>();
                
                tmp.addCol(std::vector<double>(tmp.rowNum(), 1.0).data());

                feed_forward(tmp);
                return layer[num - 1].a.trans();
            }
    };
}