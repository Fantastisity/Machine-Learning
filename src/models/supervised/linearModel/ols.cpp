#ifndef OLS_INCLUDED
#define OLS_INCLUDED
#include "ols.h"
#endif

namespace MACHINE_LEARNING {
    LinearRegression::LinearRegression() {
        #ifdef WRITE_TO_FILE
            this->output.open("output.dat");
        #endif
    }

    double LinearRegression::loss() {
        Matrix<double> tmp = this->x * this->w - this->y, l = tmp.trans() * tmp * 0.5;
        switch (r) {
            case Regularizor::None:
                break;
            case Regularizor::L1:
                l += UTIL_BASE::MODEL_UTIL::METRICS::sum(UTIL_BASE::MODEL_UTIL::METRICS::abs(this->w)) * this->lamb;
                break;
            case Regularizor::L2:
                l += this->w.trans() * this->w * this->lamb * 0.5;
                break;
            case Regularizor::ENet:
                l += this->w.trans() * this->w * this->lamb * 0.5 * (1 - this->alpha) + 
                     UTIL_BASE::MODEL_UTIL::METRICS::sum(UTIL_BASE::MODEL_UTIL::METRICS::abs(this->w)) * this->lamb * this->alpha;
                break;
        }
        return l(0, 0);
    }

    Matrix<double> LinearRegression::gradient(const Matrix<double>& X, const Matrix<double>& Y) {
        Matrix<double> grad = X.trans() * (X * this->w - Y);
        switch (r) {
            case Regularizor::None:
                break;
            case Regularizor::L1:
                grad += UTIL_BASE::MODEL_UTIL::METRICS::sign(this->w) * this->lamb;
                break;
            case Regularizor::L2:
                grad += (this->w.trans() * this->w * this->lamb)(0, 0);
                break;
            case Regularizor::ENet:
                grad += UTIL_BASE::MODEL_UTIL::METRICS::sign(this->w) * this->lamb * this->alpha + 
                        (this->w.trans() * this->w * this->lamb)(0, 0) * (1 - this->alpha);
                break;
        }
        return grad;
    }
}