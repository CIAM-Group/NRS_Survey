#pragma once

#include <cstdint>
#include <thread>


using int64 = int64_t;
using int32 = int32_t;

template<typename CallableType, typename IntableType, int32 _workers, 
         bool _enable_fn_copy>
struct Parallelizer {
    inline __attribute__((always_inline)) static void sync_perform_with_fn_copy(const CallableType fn, const IntableType range, const int32 workers = _workers) {
        std::thread threads[workers];
        const IntableType task_num_per_worker = range / workers;
        const IntableType remainder = range % workers;
        IntableType next_task = 0;
        #pragma unroll 8
        for (IntableType id = 0; id < workers; ++id) {
            threads[id] = std::thread([id, &fn, next_task, &task_num_per_worker, &remainder, &range]() {
                const IntableType _end = next_task + task_num_per_worker + (id < remainder);
                #pragma unroll 8
                for (IntableType i = next_task; i < _end; ++i) {
                    fn(i);
                }
            });
            next_task += task_num_per_worker + (id < remainder);
        }
        #pragma unroll 8
        for (IntableType id = 0; id < workers; ++id) {
            threads[id].join();
        }
    }


    inline __attribute__((always_inline)) static void sync_perform_without_fn_copy(const CallableType & fn, const IntableType range, const int32 workers = _workers) {
        std::thread threads[workers];
        const IntableType task_num_per_worker = range / workers;
        const IntableType remainder = range % workers;
        IntableType next_task = 0;
        #pragma unroll 8
        for (IntableType id = 0; id < workers; ++id) {
            threads[id] = std::thread([id, &fn, next_task, &task_num_per_worker, &remainder]() {
                const IntableType _end = next_task + task_num_per_worker + (id < remainder);
                #pragma unroll 8
                for (IntableType i = next_task; i < _end; ++i) {
                    fn(i);
                }
            });
            next_task += task_num_per_worker + (id < remainder);
        }
        #pragma unroll 8
        for (IntableType id = 0; id < workers; ++id) {
            threads[id].join();
        }
    }

    // if stdc++ > 23, we can define a static ()
    #if __cplusplus >= 202302L
    static inline __attribute__((always_inline)) void operator()(const CallableType & fn, const IntableType range, const int32 workers = _workers) {
        if constexpr (_enable_fn_copy) {
            sync_perform_with_fn_copy(fn, range, workers);
        } else {
            sync_perform_without_fn_copy(fn, range, workers);
        }
    }
    #else
    inline __attribute__((always_inline)) void operator()(const CallableType & fn, const IntableType range, const int32 workers = _workers) {
        if constexpr (_enable_fn_copy) {
            sync_perform_with_fn_copy(fn, range, workers);
        } else {
            sync_perform_without_fn_copy(fn, range, workers);
        }
    }
    #endif
};



template<typename CallableType, typename IntableType, 
         int32 _workers = 20, bool _enable_fn_copy = false, bool _turnoff = false> // turnoff is for debug
inline __attribute__((always_inline)) void parallelize(const CallableType & fn, const IntableType range, const int32 workers = _workers) {
    if constexpr (!_turnoff) {
        if (workers <= 1) {
            for (IntableType i = 0; i < range; ++i) {
                fn(i);
            }
        } else {
            Parallelizer<CallableType, IntableType, _workers, _enable_fn_copy>{}(fn, range, workers);
        }
    } else {
        for (IntableType i = 0; i < range; ++i) {
            fn(i);
        }
    }
}
