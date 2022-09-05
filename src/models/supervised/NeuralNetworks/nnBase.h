#ifndef MODELUTIL_INCLUDED
#define MODELUTIL_INCLUDED
#include "../../../utils/modelUtil.h"
#endif

enum class ACFUNC {SIGMOID, SOFTMAX, TANH, RELU, ELU};

namespace MACHINE_LEARNING {
    template<typename M>
    class NNBase {
            struct Layer {
                Matrix<double> w, a, err;
            };

            Layer initLayer(size_t nrow, size_t ncol) {
                Layer tmp;
                std::random_device d;
                std::mt19937 mersenne_engine {d()};
                std::uniform_real_distribution<float> distribution(0.0f, 2.0f);
                std::vector<double> v(nrow * ncol);
                std::generate(v.begin(), v.end(), [&](){return distribution(mersenne_engine);});
                tmp.w = Matrix<double>(std::move(v), nrow, ncol);
                return tmp;
            }
            std::vector<uint16_t> k;
            std::vector<ACFUNC> func;
            size_t iter = 1000;
        protected:
            Matrix<double> x{0}, y{0};
            Layer* layer = nullptr;
            size_t num;
            template<typename T, typename R>
            void init(T&& x, R&& y) {
                if constexpr (UTIL_BASE::isDataframe<typename std::remove_reference<T>::type>::val) 
                    this->x = x.values().template asType<double>();
                else if constexpr (UTIL_BASE::isMatrix<typename std::remove_reference<T>::type>::val) 
                    this->x = x.template asType<double>();
                else this->x = std::forward<T>(x);

                this->x.addCol(std::vector<double>(x.rowNum(), 1.0).data());

                num = k.size();
                layer = new Layer[num];
                for (uint16_t i = 0; i < num; ++i) {
                    if (!i) layer[i] = initLayer(this->x.colNum(), k[i]); // input layer
                    else if (i == num - 1) layer[i] = initLayer(k[i], 1); // output layer
                    else layer[i] = initLayer(k[i], k[i + 1]);
                }

                if constexpr (UTIL_BASE::isDataframe<typename std::remove_reference<R>::type>::val) 
                    this->y = y.values().template asType<double>();
                else if constexpr (UTIL_BASE::isMatrix<typename std::remove_reference<R>::type>::val)
                    this->y = y.template asType<double>();
                else this->y = std::forward<T>(y);

                this->y = this->y.trans();
            }

            void feed_forward(const Matrix<double>& X) {
                for (uint16_t i = 1; i < num; ++i) {
                    switch (func[i]) {
                        case ACFUNC::ELU:
                            layer[i].a = UTIL_BASE::MODEL_UTIL::METRICS::elu((!i ? X : layer[i - 1].a) * layer[i].w, .1);
                            break;
                        case ACFUNC::TANH:
                            layer[i].a = UTIL_BASE::MODEL_UTIL::METRICS::tanh((!i ? X : layer[i - 1].a) * layer[i].w);
                            break;
                        case ACFUNC::RELU:
                            layer[i].a = UTIL_BASE::MODEL_UTIL::METRICS::relu((!i ? X : layer[i - 1].a) * layer[i].w);
                            break;
                        case ACFUNC::SOFTMAX:
                            layer[i].a = UTIL_BASE::MODEL_UTIL::METRICS::softmax((!i ? X : layer[i - 1].a) * layer[i].w);
                            break;
                        case ACFUNC::SIGMOID:
                            layer[i].a = UTIL_BASE::MODEL_UTIL::METRICS::sigmoid((!i ? X : layer[i - 1].a) * layer[i].w);
                    }
                }
            }

            void train() {
                auto back_prop = [&]() -> void {
                    for (uint16_t i = num - 1; ~i; --i) {
                        switch (func[i]) {
                            case ACFUNC::ELU:
                                if (i == num - 1) layer[i].err = (y - layer[i].a) * UTIL_BASE::MODEL_UTIL::METRICS::d_elu(layer[i].a, .1);
                                else if (!i) layer[i].err = (layer[0].w * layer[1].err) * UTIL_BASE::MODEL_UTIL::METRICS::d_elu(layer[0].a.trans(), .1);
                                else layer[i].err = (layer[i].w * layer[i + 1].err.trans()) * UTIL_BASE::MODEL_UTIL::METRICS::d_elu(layer[i].a.trans(), .1);
                                break;
                            case ACFUNC::TANH:
                                if (i == num - 1) layer[i].err = (y - layer[i].a) * UTIL_BASE::MODEL_UTIL::METRICS::d_tanh(layer[i].a);
                                else if (!i) layer[i].err = (layer[0].w * layer[1].err) * UTIL_BASE::MODEL_UTIL::METRICS::d_tanh(layer[0].a.trans());
                                else layer[i].err = (layer[i].w * layer[i + 1].err.trans()) * UTIL_BASE::MODEL_UTIL::METRICS::d_tanh(layer[i].a.trans());
                                break;
                            case ACFUNC::RELU:
                                if (i == num - 1) layer[i].err = (y - layer[i].a) * UTIL_BASE::MODEL_UTIL::METRICS::d_relu(layer[i].a);
                                else if (!i) layer[i].err = (layer[0].w * layer[1].err) * UTIL_BASE::MODEL_UTIL::METRICS::d_relu(layer[0].a.trans());
                                else layer[i].err = (layer[i].w * layer[i + 1].err.trans()) * UTIL_BASE::MODEL_UTIL::METRICS::d_relu(layer[i].a.trans());
                                break;
                            case ACFUNC::SOFTMAX:
                                if (i == num - 1) layer[i].err = (y - layer[i].a) * UTIL_BASE::MODEL_UTIL::METRICS::d_softmax(layer[i].a);
                                else if (!i) layer[i].err = (layer[0].w * layer[1].err) * UTIL_BASE::MODEL_UTIL::METRICS::d_softmax(layer[0].a.trans());
                                else layer[i].err = (layer[i].w * layer[i + 1].err.trans()) * UTIL_BASE::MODEL_UTIL::METRICS::d_softmax(layer[i].a.trans());
                                break;
                            case ACFUNC::SIGMOID:
                                if (i == num - 1) layer[i].err = (y - layer[i].a) * UTIL_BASE::MODEL_UTIL::METRICS::d_sigmoid(layer[i].a);
                                else if (!i) layer[i].err = (layer[0].w * layer[1].err) * UTIL_BASE::MODEL_UTIL::METRICS::d_sigmoid(layer[0].a.trans());
                                else layer[i].err = (layer[i].w * layer[i + 1].err.trans()) * UTIL_BASE::MODEL_UTIL::METRICS::d_sigmoid(layer[i].a.trans());
                        }
                    }
                    for (uint16_t i = 0; i < num; ++i) {
                        if (!i) layer[i].w += x.trans() * layer[i + 1].err.trans();
                        else if (i == num - 1) layer[i].w += layer[i].a.trans() * layer[i].err;
                        else layer[i].w += layer[i].a.trans() * layer[i + 1].err.trans();
                    }
                };

                for (size_t i = 0; i < iter; ++i) {
                    feed_forward(x);
                    back_prop();
                }
            }
        public:
            ~NNBase() {
                if (layer) {
                    delete[] layer;
                    layer = nullptr;
                }
            }
            void set_layers(std::vector<uint16_t> k, ACFUNC func = ACFUNC::TANH) {
                this->k = k, this->func = std::vector<ACFUNC>(k.size(), func);
            }

            void set_layers(std::vector<uint16_t> k, std::vector<ACFUNC> func) {
                this->k = k, this->func = func;
            }

            void set_iter(size_t iter) {
                this->iter = iter;
            }

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