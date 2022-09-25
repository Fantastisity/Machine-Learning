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
            std::vector<size_t> k;
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

                this->x.addCol(std::vector<double>(x.rowNum(), 1.0).data()); // Bias

                num = k.size();
                layer = new Layer[num + 1];
                layer[0] = initLayer(this->x.colNum(), k[0]); // Input layer
                for (uint16_t i = 0; i < num - 1; ++i) layer[i + 1] = initLayer(k[i], k[i + 1]); // Hidden layers
                layer[num] = initLayer(k[num - 1], 1); // Output layer

                if constexpr (UTIL_BASE::isDataframe<typename std::remove_reference<R>::type>::val) 
                    this->y = y.values().template asType<double>();
                else if constexpr (UTIL_BASE::isMatrix<typename std::remove_reference<R>::type>::val)
                    this->y = y.template asType<double>();
                else this->y = std::forward<T>(y);
            }

            void feed_forward(const Matrix<double>& X) {
                for (uint16_t i = 0; i <= num; ++i) {
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
                auto back_propagation = [&]() -> void {
                    for (int i = num; ~i; --i) {
                        switch (func[i]) {
                            case ACFUNC::ELU:
                                if (i == num) layer[i].err = (y - layer[i].a).multiply(UTIL_BASE::MODEL_UTIL::METRICS::d_elu(layer[i].a, .1));
                                else if (!i) layer[i].err = (layer[i + 1].w * layer[i + 1].err).multiply(UTIL_BASE::MODEL_UTIL::METRICS::d_elu(layer[i].a.trans(), .1));
                                else layer[i].err = (layer[i + 1].w * layer[i + 1].err.trans()).multiply(UTIL_BASE::MODEL_UTIL::METRICS::d_elu(layer[i].a.trans(), .1)); 
                                break;
                            case ACFUNC::TANH:
                                if (i == num) layer[i].err = (y - layer[i].a).multiply(UTIL_BASE::MODEL_UTIL::METRICS::d_tanh(layer[i].a));
                                else if (!i) layer[i].err = (layer[i + 1].w * layer[i + 1].err).multiply(UTIL_BASE::MODEL_UTIL::METRICS::d_tanh(layer[i].a.trans()));
                                else layer[i].err = (layer[i + 1].w * layer[i + 1].err.trans()).multiply(UTIL_BASE::MODEL_UTIL::METRICS::d_tanh(layer[i].a.trans()));                     
                                break;
                            case ACFUNC::RELU:
                                if (i == num) layer[i].err = (y - layer[i].a).multiply(UTIL_BASE::MODEL_UTIL::METRICS::d_relu(layer[i].a));
                                else if (!i) layer[i].err = (layer[i + 1].w * layer[i + 1].err).multiply(UTIL_BASE::MODEL_UTIL::METRICS::d_relu(layer[i].a.trans()));
                                else layer[i].err = (layer[i + 1].w * layer[i + 1].err.trans()).multiply(UTIL_BASE::MODEL_UTIL::METRICS::d_relu(layer[i].a.trans())); 
                                break;
                            case ACFUNC::SOFTMAX:
                                if (i == num) layer[i].err = (y - layer[i].a).multiply(UTIL_BASE::MODEL_UTIL::METRICS::d_softmax(layer[i].a));
                                else if (!i) layer[i].err = (layer[i + 1].w * layer[i + 1].err).multiply(UTIL_BASE::MODEL_UTIL::METRICS::d_softmax(layer[i].a.trans()));
                                else layer[i].err = (layer[i + 1].w * layer[i + 1].err.trans()).multiply(UTIL_BASE::MODEL_UTIL::METRICS::d_softmax(layer[i].a.trans())); 
                                break;
                            case ACFUNC::SIGMOID:
                                if (i == num) layer[i].err = (y - layer[i].a).multiply(UTIL_BASE::MODEL_UTIL::METRICS::d_sigmoid(layer[i].a));
                                else if (!i) layer[i].err = (layer[i + 1].w * layer[i + 1].err).multiply(UTIL_BASE::MODEL_UTIL::METRICS::d_sigmoid(layer[i].a.trans()));
                                else layer[i].err = (layer[i + 1].w * layer[i + 1].err.trans()).multiply(UTIL_BASE::MODEL_UTIL::METRICS::d_sigmoid(layer[i].a.trans())); 
                        }
                    }
                    for (size_t i = 0; i <= num; ++i) {
                        if (!i) layer[0].w += x.trans() * layer[0].err.trans();
                        else if (i == num) layer[i].w += layer[i - 1].a.trans() * layer[i].err;
                        else layer[i].w += layer[i - 1].a.trans() * layer[i].err.trans();
                    }
                };

                for (size_t i = 0; i < iter; ++i) {
                    feed_forward(x);
                    back_propagation();
                }
            }
        public:
            ~NNBase() {
                if (layer) {
                    delete[] layer;
                    layer = nullptr;
                }
            }
            void set_layers(std::vector<size_t> k, ACFUNC func = ACFUNC::TANH) {
                this->k = k, this->func = std::vector<ACFUNC>(k.size() + 1, func);
            }

            void set_layers(std::vector<size_t> k, std::vector<ACFUNC> func) {
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