#include <iostream>
#include <chrono>
#include <cstdio>

struct Logger {
    template<typename... args>
    void operator()(args... arg) {
        #ifdef DEBUG
        printf("== DEBUG == \t");
        #endif
        __attribute__((unused)) auto _ = {((void)(std::cout << arg << "  "), 0)...};
        puts("");
    }

    void set_timer_repeats(size_t r) {
        timer_repeats = r;
    }

    void start_timer() {
        if (timer_repeats) start = std::chrono::steady_clock::now();
    }

    void end_timer() {
        if (timer_repeats) end = std::chrono::steady_clock::now();
    }

    void elapsed_time(char unit = 'u') {
        if (!timer_repeats) return;
        --timer_repeats;
        switch (unit) {
            case 's':
                printf("elapsed time -- %u s\n", std::chrono::duration_cast<std::chrono::seconds>(end - start).count());
                break;
            case 'u':
                printf("elapsed time -- %u us\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
                break;
            case 'n':
                printf("elapsed time -- %u ns\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        }
    }

    private:
        std::chrono::steady_clock::time_point start, end;
        size_t timer_repeats = 1; // Number of repeats that the timer should run
};

extern Logger logger;