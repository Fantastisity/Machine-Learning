#ifndef LR_INCLUDED
#define LR_INCLUDED
#include "include/models/supervisedModel/lr.h"
#endif

namespace MACHINE_LEARNING {
    LogisticRegression::LogisticRegression() {
        #ifdef WRITE_TO_FILE
            this->output.open("test.dat");
        #endif
    }

    double LogisticRegression::loss() {
        Matrix<double> ypred = ModelUtil::sigmoid(this->x * this->w),
                       l = -(this->y.trans() * ModelUtil::loge(ypred) + 
                            (1 - this->y).trans() * ModelUtil::loge(1 - ypred));
        switch (r) {
            case Regularizor::None:
                break;
            case Regularizor::L1:
                l += ModelUtil::sum(ModelUtil::abs(this->w)) * this->lamb;
                break;
            case Regularizor::L2:
                l += this->w.trans() * this->w * this->lamb * 0.5;
                break;
            case Regularizor::ENet:
                l += this->w.trans() * this->w * this->lamb * 0.5 * (1 - this->alpha) + 
                     ModelUtil::sum(ModelUtil::abs(this->w)) * this->lamb * this->alpha;
                break;
        }
        return l(0, 0) / y.rowNum();
    }

    Matrix<double> LogisticRegression::gradient(Matrix<double>& X, Matrix<double>& Y) {
        Matrix<double> grad = X.trans() * (ModelUtil::sigmoid(X * this->w) - Y);
        switch (r) {
            case Regularizor::None:
                break;
            case Regularizor::L1:
                grad += ModelUtil::sign(this->w) * this->lamb;
                break;
            case Regularizor::L2:
                grad += (this->w.trans() * this->w * this->lamb)(0, 0);
                break;
            case Regularizor::ENet:
                grad += ModelUtil::sign(this->w) * this->lamb * this->alpha + (this->w.trans() * this->w * this->lamb)(0, 0) * (1 - this->alpha);
                break;
        }
        return grad;
    }
}