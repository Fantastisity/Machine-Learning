#ifndef MODEL_UTIL_INCLUDED
#define MODEL_UTIL_INCLUDED
#include "../../utils/modelUtil.h"
#endif

namespace MACHINE_LEARNING {
    class LinearRegression : public SupervisedModel {
            friend class ModelUtil;
            Matrix<double> w;

            auto loss();

            auto gradient(Matrix<double>& X, Matrix<double>& Y);

            void gradient_descent();
        public:
            explicit 
            LinearRegression();

            void print_params();

            void print_weights();

            void fit(const DataFrame<double>& x, const DataFrame<double>& y, const uint8_t verbose = 0);

            Matrix<double> predict(const DataFrame<double>& xtest);
    };
}