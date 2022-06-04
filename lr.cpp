#ifndef LINEAR_MODEL_INCLUDED
#define LINEAR_MODEL_INCLUDED
#include "include/models/supervisedModel/lr.h"
#endif

namespace MACHINE_LEARNING {
    LogisticRegression::LogisticRegression() {
        #ifdef WRITE_TO_FILE
            this->output.open("test.dat");
        #endif
    }

    double LogisticRegression::loss() {
        Matrix<double> l = -(this->y.trans() * MatrixUtil::naturalLog(MatrixUtil::sigmoid(X * this->w)) - 
                            (1 - this->y).trans() * MatrixUtil::naturalLog(1 - MatrixUtil::sigmoid(X * this->w)));
        switch (r) {
            case Regularizor::None:
                break;
            case Regularizor::L1:
                l += modUtil.sum(modUtil.abs(this->w)) * this->lamb;
                break;
            case Regularizor::L2:
                l += this->w.trans() * this->w * this->lamb * 0.5;
                break;
            case Regularizor::ENet:
                l += this->w.trans() * this->w * this->lamb * 0.5 * (1 - this->alpha) + 
                     modUtil.sum(modUtil.abs(this->w)) * this->lamb * this->alpha;
                break;
        }
        return l(0, 0);
    }

    Matrix<double> LogisticRegression::gradient(Matrix<double>& X, Matrix<double>& Y) {
        Matrix<double> grad = X.trans() * (MatrixUtil::sigmoid(X * this->w) - Y);
        switch (r) {
            case Regularizor::None:
                break;
            case Regularizor::L1:
                grad += modUtil.sign(this->w) * this->lamb;
                break;
            case Regularizor::L2:
                grad += (this->w.trans() * this->w * this->lamb)(0, 0);
                break;
            case Regularizor::ENet:
                grad += modUtil.sign(this->w) * this->lamb * this->alpha + (this->w.trans() * this->w * this->lamb)(0, 0) * (1 - this->alpha);
                break;
        }
        return grad;
    }
}