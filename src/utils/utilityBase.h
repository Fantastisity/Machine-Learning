#ifndef LOGGER_ENABLED
#define LOGGER_ENABLED
#include "logger.h"
#endif
#include <vector>
#include <immintrin.h>
#include <utility>
#include <unordered_map>
#include <cmath>
#include <tuple>
#include <type_traits>
#include <bitset>
#include <string>
#include <cstring>
#include <thread>
#include <iomanip>
#include <iostream>

static auto _ = []() {
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);
  return nullptr;
}();
#include <algorithm>
#include <random>
#include <fstream>
#include <initializer_list>

namespace MACHINE_LEARNING {
    using ll = long long;
    namespace UTIL_BASE {
        template <typename...> using Void = void;
        template<typename U, typename R = void>
        struct isDataframe {
            const static bool val = 0;
        };

        template<typename U>
        struct isDataframe<U, Void<decltype(&U::addFeature)>> {
            const static bool val = 1;
        };

        template<typename U, typename R = void>
        struct isMatrix {
            const static bool val = 0;
        };

        template<typename U>
        struct isMatrix<U, Void<decltype(&U::trans)>> {
            const static bool val = 1;
        };

        template<typename T>
        struct isNumerical {
            const static bool val = 1;
        };

        template<>
        struct isNumerical<std::string> {
            const static bool val = 0;
        };

        template<>
        struct isNumerical<char> {
            const static bool val = 0;
        };
    }
}