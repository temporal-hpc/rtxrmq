#pragma once

#include <omp.h>
#include "../hrmq/includes/RMQRMM64.h"

template <typename T>
T cpu_min(T *A, int l, int r) {
    T min = A[l];
    for (int i = l; i <= r; ++i) {
        if (A[i] < min){
            min = A[i];
        }
    }
    return min;
}

template <typename T>
T* cpu_rmq(int n, int nq, T *A, int2 *Q, int nt) {
    Timer timer;
    T* out = new T[nq];

    omp_set_num_threads(nt);
    printf("%sComputing RMQs (%-11s, nt=%2i).......%s", AC_BOLDCYAN, algStr[ALG_CPU_BASE], nt, AC_RESET); fflush(stdout);
    timer.restart();
    for (int i = 0; i < REPS; ++i) {
        #pragma omp parallel for shared(out, A, Q)
        for (int i = 0; i < nq; ++i) {
            out[i] = cpu_min<T>(A, Q[i].x, Q[i].y);
        }
    }
    timer.stop();
    float timems = timer.get_elapsed_ms();
    float time_it = timems/REPS;
    printf(AC_BOLDCYAN "done (%i reps): %f secs: [%.2f RMQs/sec, %f nsec/RMQ]\n" AC_RESET, REPS, timems/1000.0, (double)nq/(time_it/1000.0), (double)time_it*1e6/nq);
    write_results(timems, nq);

    return out;
}

int *rmq_rmm_par(int n, int nq, int *A, int2 *Q, int nt) {
    using namespace rmqrmm;
    omp_set_num_threads(nt);

    Timer timer;
    RMQRMM64 *rmq = NULL;
    int* out = new int[nq];

    // create rmq struct
    printf("Creating MinMaxTree......................."); fflush(stdout);
    timer.restart();
    rmq = new RMQRMM64(A, (unsigned long)n);
    uint size = rmq->getSize();
    timer.stop();
    printf("done: %f ms (%f MB)\n",timer.get_elapsed_ms(), (double)size/1e9);

    //printf("%sAnswering Querys [%2i threads]......", AC_BOLDCYAN, nt); fflush(stdout);
    printf("%sComputing RMQs (%-11s, nt=%2i).......%s", AC_BOLDCYAN, algStr[ALG_CPU_HRMQ], nt, AC_RESET); fflush(stdout);
    timer.restart();
    for (int i = 0; i < REPS; ++i) {
        #pragma omp parallel for shared(rmq, out, A, Q)
	for (int i = 0; i < nq; ++i) {
	    int idx = rmq->queryRMQ(Q[i].x, Q[i].y);
	    out[i] = A[idx];
       }
    }
    timer.stop();
    double timems = timer.get_elapsed_ms();
    float time_it = timems/REPS;
    printf(AC_BOLDCYAN "done (%i reps): %f secs: [%.2f RMQs/sec, %f nsec/RMQ]\n" AC_RESET, REPS, timems/1000.0, (double)nq/(time_it/1000.0), (double)time_it*1e6/nq);
    write_results(timems, nq);
    return out;
}
