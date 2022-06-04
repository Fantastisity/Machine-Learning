#ifndef LINEAR_MODEL_INCLUDED
#define LINEAR_MODEL_INCLUDED
#include "include/models/supervisedModel/ols.h"
#endif

namespace MACHINE_LEARNING {
    LinearRegression::LinearRegression() {
        #ifdef WRITE_TO_FILE
            this->output.open("test.dat");
        #endif
    }

    double LinearRegression::loss() {
        Matrix<double> tmp = this->x * this->w - this->y, l = tmp.trans() * tmp * 0.5;
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

    Matrix<double> LinearRegression::gradient(Matrix<double>& X, Matrix<double>& Y) {
        Matrix<double> grad = X.trans() * (X * this->w - Y);
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

    void LinearRegression::print_params() {
        pretty_print("", '*', 58, '*');
        pretty_print("", ' ', 38, "Parameter Settings");
        pretty_print("", '*', 58, '*');
        switch (this->t) { 
            case GDType::BATCH: 
                pretty_print("gradient descent type:", ' ', 29, "BGD");
                break;
            case GDType::STOCHASTIC:
                pretty_print("gradient descent type:", ' ', 29, "SGD");
                break;
            case GDType::MINI_BATCH:
                pretty_print("gradient descent type:", ' ', 29, "MBGD"),
                pretty_print("batch size:", ' ', 40, this->batch_size);
                break;
        }
        if (this->r != Regularizor::None) {
            switch (this->r) {
                case Regularizor::L1:
                    pretty_print("regularizor:", ' ', 39, "Lasso");
                    break;
                case Regularizor::L2:
                    pretty_print("regularizor:", ' ', 39, "Ridge");
                    break;
                case Regularizor::ENet:
                    pretty_print("regularizor:", ' ', 39, "Elastic Net"),
                    pretty_print("alpha:", ' ', 45, this->alpha);
                    break;
            };
            pretty_print("lambda:", ' ', 44, this->lamb);
        }
        
        pretty_print("eta:", ' ', 47, this->eta);
        pretty_print("epsilon:", ' ', 43, this->eps);
        pretty_print("iterations:", ' ', 40, this->iter);
        pretty_print("", '*', 58, '*');
    }

    void LinearRegression::print_weights() {
        pretty_print("", '*', 58, '*');
        pretty_print("", ' ', 35, "Final Weights");
        pretty_print("", '*', 58, '*');
        for (size_t i = 0, r = this->w.rowNum(); i < r; ++i) pretty_print(i, ' ', 50, this->w(i, 0));
        pretty_print("", '*', 58, '*');
    }
}