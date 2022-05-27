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

#define pretty_print(prefix, fill, width, suffix) (std::cout << prefix << std::setfill(fill) << std::setw(width) << suffix << '\n')

namespace MACHINE_LEARNING {
    struct UtilBase {
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
    };
}