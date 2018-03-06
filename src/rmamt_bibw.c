/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 *   Copyright (C) 2007      University of Chicago
 *   Copyright (c) 2016-2018 Los Alamos National Security, LLC. All rights
 *                           reserved.
 * ****** SANDIA ADD YOUR COPYRIGHTS BEFORE RELEASE ******
 *   See COPYRIGHT notice in top-level directory.
 */

#include "rmamt_common.h"

#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <strings.h>
#include <stdint.h>
#include <inttypes.h>

uint64_t find_max();

/* target-side functions for single window */
static void *runfunc_put (ArgStruct* a);
static void *runfunc_get (ArgStruct* a);

/* origin-side window per thread functions */
static void *bibw_put_lock_all_winperthread (ArgStruct* a);
static void *bibw_get_lock_all_winperthread (ArgStruct* a);
static void *bibw_put_fence_winperthread (ArgStruct* a);
static void *bibw_get_fence_winperthread (ArgStruct* a);
static void *bibw_put_lock_per_rank_winperthread (ArgStruct* a);
static void *bibw_get_lock_per_rank_winperthread (ArgStruct* a);
static void *bibw_put_flush_winperthread (ArgStruct* a);
static void *bibw_get_flush_winperthread (ArgStruct* a);
static void *bibw_put_pscw_winperthread (ArgStruct* a);
static void *bibw_get_pscw_winperthread (ArgStruct* a);

static rmamt_fn_t rmamt_winperthread_fns[RMAMT_OPERATIONS_MAX][RMAMT_SYNC_MAX] = {
    [RMAMT_PUT] = {
        [RMAMT_LOCK_ALL] = bibw_put_lock_all_winperthread,
        [RMAMT_FENCE] = bibw_put_fence_winperthread,
        [RMAMT_LOCK] = bibw_put_lock_per_rank_winperthread,
        [RMAMT_FLUSH] = bibw_put_flush_winperthread,
        [RMAMT_PSCW] = bibw_put_pscw_winperthread,
    },
    [RMAMT_GET] = {
        [RMAMT_LOCK_ALL] = bibw_get_lock_all_winperthread,
        [RMAMT_FENCE] = bibw_get_fence_winperthread,
        [RMAMT_LOCK] = bibw_get_lock_per_rank_winperthread,
        [RMAMT_FLUSH] = bibw_get_flush_winperthread,
        [RMAMT_PSCW] = bibw_get_pscw_winperthread,
    },
};

/* origin-side functions */
static void *bibw_orig_lock_all (ArgStruct *a);
static void *bibw_orig_lock (ArgStruct *a);
static void *bibw_orig_flush (ArgStruct *a);
static void *bibw_orig_fence (ArgStruct *a);
static void *bibw_orig_pscw (ArgStruct *a);

static rmamt_fn_t rmamt_origin_fns[RMAMT_SYNC_MAX] = {
    [RMAMT_LOCK_ALL] = bibw_orig_lock_all,
    [RMAMT_FENCE] = bibw_orig_fence,
    [RMAMT_LOCK] = bibw_orig_lock,
    [RMAMT_FLUSH] = bibw_orig_flush,
    [RMAMT_PSCW] = bibw_orig_pscw,
};

static ArgStruct args[MAX_THREADS];
static uint64_t thread_etimes[MAX_THREADS];
static char* tbufs[MAX_THREADS];
static char* obuf;
static MPI_Win win[MAX_THREADS];
static int64_t times[MAX_THREADS][256];

int main(int argc,char *argv[])
{
    MPI_Group group = MPI_GROUP_NULL, comm_group;
    int nprocs, provided, rank, rc;
    pthread_t id[MAX_THREADS];
    MPI_Request req;
    int64_t win_size;
    int win_count;
    size_t max_size, min_size;
    uint64_t stt, ttt = 0;
    FILE *output_file = NULL;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE) {
        printf("Thread multiple needed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    if (nprocs != 2) {
        printf("Run with 2 processes\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_group (MPI_COMM_WORLD, &comm_group);

    rmamt_parse_options ("rmamt_bibw", argc, argv);

    if (rmamt_output_file && 0 == rank) {
	output_file = fopen (rmamt_output_file, "w");
	if (NULL == output_file) {
	    fprintf (stderr, "Could not open %s for writing\n", rmamt_output_file);
	    MPI_Abort (MPI_COMM_WORLD, 1);
	}
	free (rmamt_output_file);
    }

    if (rmamt_bind_threads) {
	rc = rmamt_bind_init ();
	if (-1 == rc) {
	    printf ("***** WARNING: Thread binding requested but not available *****\n");
	}
    }

    min_size = rmamt_min_size / rmamt_threads;
    if (min_size < 1) {
	min_size = 1;
    }

    max_size = rmamt_max_size / rmamt_threads;
    win_count = rmamt_win_per_thread ? rmamt_threads : 1;
    win_size = rmamt_max_size / win_count;

    obuf = rmamt_malloc (rmamt_max_size);
    if (!obuf) {
        printf("Cannot allocate buffer\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int k = 0 ; k < rmamt_threads ; ++k) {
        memset(obuf + max_size * k, (char)k%9+'0', max_size);
    }

    /* create windows */
    for (int i = 0 ; i < win_count ; ++i) {
        MPI_CHECK(MPI_Win_allocate (win_size, 1, MPI_INFO_NULL, MPI_COMM_WORLD,
                                    tbufs + i, win + i));
        if (win_size) {
            memset (tbufs[i], '-', win_size);
        }
    }

    if (RMAMT_PSCW == rmamt_sync) {
        MPI_Group_incl (comm_group, 1, &(int){!rank}, &group);
    }


    if (!rank) {
        printf ("##########################################\n");
        printf ("# RMA-MT Bi-directional Bandwidth\n");
        printf ("#\n");
        printf ("# Operation: %s\n", rmamt_operation_strings[rmamt_operation]);
        printf ("# Sync: %s\n", rmamt_sync_strings[rmamt_sync]);
        printf ("# Thread count: %u\n", (unsigned) rmamt_threads);
        printf ("# Iterations: %u\n", (unsigned) rmamt_iterations);
        printf ("# Ibarrier: %s, sleep interval: %uns\n", rmamt_use_ibarrier ? "yes" : "no",
                rmamt_sleep_interval);
	printf ("# Bind worker threads: %s\n", rmamt_bind_threads ? "yes" : "no");
	printf ("# Number of windows: %u\n", rmamt_win_per_thread ? rmamt_threads : 1);
        printf ("##########################################\n");
	printf ("  BpT(%i)\t  BxT(%i)\tBandwidth(MiB/s)\tMessage_Rate(M/s)\n", rmamt_threads, rmamt_threads);
	if (output_file) {
	    fprintf (output_file, "##########################################\n");
	    fprintf (output_file, "# RMA-MT Bandwidth\n");
	    fprintf (output_file, "#\n");
	    fprintf (output_file, "# Operation: %s\n", rmamt_operation_strings[rmamt_operation]);
	    fprintf (output_file, "# Sync: %s\n", rmamt_sync_strings[rmamt_sync]);
	    fprintf (output_file, "# Thread count: %u\n", (unsigned) rmamt_threads);
	    fprintf (output_file, "# Iterations: %u\n", (unsigned) rmamt_iterations);
	    fprintf (output_file, "# Ibarrier: %s, sleep interval: %uns\n", rmamt_use_ibarrier ? "yes" : "no",
		     rmamt_sleep_interval);
	    fprintf (output_file, "# Bind worker threads: %s\n", rmamt_bind_threads ? "yes" : "no");
	    fprintf (output_file, "# Number of windows: %u\n", rmamt_win_per_thread ? rmamt_threads : 1);
	    fprintf (output_file, "##########################################\n");
	    fprintf (output_file, "BpT(%i),BxT(%i),Bandwidth(MiB/s),Message_Rate(M/s)\n", rmamt_threads, rmamt_threads);
	}
    }

    thread_barrier_init (rmamt_win_per_thread ? rmamt_threads : rmamt_threads + 1);

    stt = time_getns ();
    for (int i = 0 ; i < rmamt_threads ; ++i) {
	args[i].tid = i;
	args[i].max_size = max_size;
	args[i].min_size = min_size;
	args[i].win = rmamt_win_per_thread ? win[i] : win[0];
	args[i].group = group;
	args[i].target = !rank;

	//printf("args[%u].tid = %u\n", i, arggs[i].tid);
	if (!rmamt_win_per_thread) {
	    pthread_create(id+i, NULL, (void *(*)(void *)) (RMAMT_GET == rmamt_operation ? runfunc_get : runfunc_put), args+i);
	} else {
	    pthread_create(id+i, NULL, (void *(*)(void *)) rmamt_winperthread_fns[rmamt_operation][rmamt_sync], args+i);
	}
    }

    /* wait for threads to be ready */
    thread_barrier (0);

    if (ttt < find_max()-stt) ttt = find_max()-stt;

    if (!rmamt_win_per_thread) {
	rmamt_origin_fns[rmamt_sync] (&(ArgStruct){.min_size = min_size, .max_size = max_size, .group = group, .win = win[0], .target = !rank});
    }

    for (int i = 0 ; i < rmamt_threads ; ++i) {
	pthread_join(id[i], NULL);
    }

    if (0 == rank) {
        for (uint32_t j = min_size, step = 0 ; j <= max_size ; j <<= 1, ++step) {
	    /* there are messages going in both directions */
            size_t sz = 2 * j * (rmamt_threads / win_count);
	    float max_time = 0.0;
            float speed = 0.0;

            for (int i = 0 ; i < win_count ; ++i) {
                speed += ((float) (sz * rmamt_iterations) * 953.67431640625) / (float) times[i][step];
		max_time = max_time > (float) times[i][step] ? max_time : (float) times[i][step];
            }

	    if (output_file) {
		fprintf (output_file, "%lu,%lu,%f,%f\n", (unsigned long) j, (unsigned long) j * rmamt_threads, speed,
			 (rmamt_threads * 2 * rmamt_iterations * 1000000000.0) / max_time);
	    }

	    printf ("%9lu\t%9lu\t%13.3f\t\t%13.3f\n", (unsigned long) j, (unsigned long) j * rmamt_threads, speed,
		    (rmamt_threads * 2 * rmamt_iterations * 1000000000.0) / max_time);
        }
    }

    if (MPI_GROUP_NULL != group) {
        MPI_Group_free (&group);
    }

    MPI_Group_free (&comm_group);

    MPI_Barrier (MPI_COMM_WORLD);

    if (output_file) {
	fclose (output_file);
    }

    for (int i = 0 ; i < win_count ; ++i) {
        MPI_CHECK(MPI_Win_free(win + i));
    }

    if (rmamt_bind_threads) {
	rmamt_bind_finalize ();
    }

    rmamt_free (obuf, rmamt_max_size);

    MPI_Finalize();
    return 0;
}

uint64_t find_max(){
  uint64_t max = 0;
  int tmp;
  int sz = sizeof(thread_etimes)/sizeof(thread_etimes[0]);
  for (tmp = 0; tmp < sz; tmp++)
    if(max < thread_etimes[tmp]) max=thread_etimes[tmp];
  return (double) max;
}

#define DEFINE_ORIGIN_THREAD_FN(sync, type, fn, init_fn, start_sync, end_sync, fini_fn, expose, release) \
    static void *bibw_ ## type ## _ ## sync ## _winperthread (ArgStruct* a) { \
        const int tid = (int) a->tid;                                   \
        uint64_t start, stop, ttime;					\
        size_t max_size = a->max_size;                                  \
        size_t min_size = a->min_size;                                  \
                                                                        \
	if (rmamt_bind_threads) {					\
	    rmamt_bind (tid);						\
	}								\
									\
        thread_etimes[tid] = time_getns ();                             \
                                                                        \
        init_fn;                                                        \
        /* signal the main thread that we are ready */                  \
        thread_barrier (0);                                             \
                                                                        \
        for (uint32_t j = min_size, cycle = 0 ; j <= max_size ; j <<= 1) { \
	    expose;							\
            start_sync;                                                 \
                                                                        \
            for (int l = 0 ; l < RMAMT_WARMUP_ITERATIONS ; l++) {       \
                fn (obuf + tid * j, j, MPI_BYTE, a->target, 0, j,	\
                    MPI_BYTE, a->win);					\
            }                                                           \
                                                                        \
            end_sync;                                                   \
	    release;							\
									\
	    MPI_Barrier (MPI_COMM_WORLD);				\
                                                                        \
            thread_barrier (cycle * 2 + 1);                             \
                                                                        \
            start = time_getns ();                                      \
	    expose;							\
            start_sync;                                                 \
                                                                        \
            for (int l = 0 ; l < rmamt_iterations ; l++) {              \
                fn (obuf + tid * j, j, MPI_BYTE, a->target, 0, j,	\
                    MPI_BYTE, a->win);					\
            }                                                           \
                                                                        \
            end_sync;                                                   \
	    release;							\
	    MPI_Barrier (MPI_COMM_WORLD);				\
            ttime = time_getns () - start;				\
                                                                        \
            times[tid][cycle] = ttime;					\
            thread_barrier (cycle * 2 + 2);                             \
            ++cycle;                                                    \
        }                                                               \
                                                                        \
        fini_fn;                                                        \
                                                                        \
        return 0;                                                       \
    }

DEFINE_ORIGIN_THREAD_FN(lock_all, put, MPI_Put, (void) 0, MPI_Win_lock_all (0, a->win), MPI_Win_unlock_all (a->win),
			(void) 0, (void) 0, (void) 0)
DEFINE_ORIGIN_THREAD_FN(lock_all, get, MPI_Get, (void) 0, MPI_Win_lock_all (0, a->win), MPI_Win_unlock_all (a->win),
			(void) 0, (void) 0, (void) 0)
DEFINE_ORIGIN_THREAD_FN(lock_per_rank, put, MPI_Put, (void) 0, MPI_Win_lock (MPI_LOCK_SHARED, a->target, 0, a->win),
			MPI_Win_unlock (a->target, a->win), (void) 0, (void) 0, (void) 0)
DEFINE_ORIGIN_THREAD_FN(lock_per_rank, get, MPI_Get, (void) 0, MPI_Win_lock (MPI_LOCK_SHARED, a->target, 0, a->win),
			MPI_Win_unlock (a->target, a->win), (void) 0, (void) 0, (void) 0)
DEFINE_ORIGIN_THREAD_FN(flush, put, MPI_Put, MPI_Win_lock (MPI_LOCK_SHARED, a->target, 0, a->win), (void) 0,
			MPI_Win_flush (a->target, a->win), MPI_Win_unlock (a->target, a->win), (void) 0, (void) 0)
DEFINE_ORIGIN_THREAD_FN(flush, get, MPI_Get, MPI_Win_lock (MPI_LOCK_SHARED, a->target, 0, a->win), (void) 0,
			MPI_Win_flush (a->target, a->win), MPI_Win_unlock (a->target, a->win), (void) 0, (void) 0)
DEFINE_ORIGIN_THREAD_FN(pscw, put, MPI_Put, (void) 0, MPI_Win_start (a->group, 0, a->win), MPI_Win_complete (a->win), (void) 0,
			MPI_Win_post (a->group, 0, a->win), MPI_Win_wait (a->win))
DEFINE_ORIGIN_THREAD_FN(pscw, get, MPI_Get, (void) 0, MPI_Win_start (a->group, 0, a->win), MPI_Win_complete (a->win), (void) 0,
			MPI_Win_post (a->group, 0, a->win), MPI_Win_wait (a->win))
DEFINE_ORIGIN_THREAD_FN(fence, put, MPI_Put, MPI_Win_fence (MPI_MODE_NOPRECEDE, a->win), (void) 0, MPI_Win_fence (0, a->win),
			(void) 0, (void) 0, (void) 0)
DEFINE_ORIGIN_THREAD_FN(fence, get, MPI_Get, MPI_Win_fence (MPI_MODE_NOPRECEDE, a->win), (void) 0, MPI_Win_fence (0, a->win),
			(void) 0, (void) 0, (void) 0)

/* origin-side loops */
#define DEFINE_ORIGIN_FN(sync, init_fn, start_sync, end_sync, fini_fn, expose, release)	\
    static void *bibw_orig_ ## sync (ArgStruct *a) {			\
        size_t max_size = a->max_size;                                  \
        size_t min_size = a->min_size;                                  \
                                                                        \
        init_fn;                                                        \
                                                                        \
        for (uint64_t j = min_size, cycle = 0, barrier_cycle = 1 ; j <= max_size ; j <<= 1, cycle++) { \
            uint64_t stime, etime, ttime;				\
                                                                        \
            /* warm up */                                               \
	    MPI_CHECK( expose );					\
            MPI_CHECK( start_sync );                                    \
                                                                        \
            thread_barrier (barrier_cycle++);                           \
            thread_barrier (barrier_cycle++);                           \
                                                                        \
            MPI_CHECK( end_sync);                                       \
	    MPI_CHECK( release );					\
                                                                        \
	    MPI_Barrier (MPI_COMM_WORLD);				\
									\
            /* timing */                                                \
            stime = time_getns();                                       \
	    MPI_CHECK( expose );					\
            MPI_CHECK( start_sync );                                    \
                                                                        \
            thread_barrier (barrier_cycle++);                           \
            thread_barrier (barrier_cycle++);                           \
                                                                        \
            MPI_CHECK( end_sync );                                      \
	    MPI_CHECK( release );					\
	    MPI_Barrier (MPI_COMM_WORLD);				\
            etime = time_getns ();                                      \
            ttime = etime - stime;					\
									\
            times[0][cycle] = ttime;                                    \
        }                                                               \
                                                                        \
        fini_fn;                                                        \
                                                                        \
        return NULL;                                                    \
    }

DEFINE_ORIGIN_FN(lock_all, (void) 0, MPI_Win_lock_all(0, a->win), MPI_Win_unlock_all (a->win), (void) 0,
		 MPI_SUCCESS, MPI_SUCCESS)
DEFINE_ORIGIN_FN(lock, (void) 0, MPI_Win_lock(MPI_LOCK_SHARED, a->target, 0, a->win), MPI_Win_unlock (a->target, a->win),
		 (void) 0, MPI_SUCCESS, MPI_SUCCESS)
DEFINE_ORIGIN_FN(flush, MPI_Win_lock (MPI_LOCK_SHARED, a->target, 0, a->win), MPI_SUCCESS, MPI_Win_flush (a->target, a->win),
		 MPI_Win_unlock (a->target, a->win), MPI_SUCCESS, MPI_SUCCESS)
DEFINE_ORIGIN_FN(fence, MPI_Win_fence (MPI_MODE_NOPRECEDE, a->win), MPI_SUCCESS, MPI_Win_fence (0, a->win),
		 (void) 0, MPI_SUCCESS, MPI_SUCCESS)
DEFINE_ORIGIN_FN(pscw, (void) 0, MPI_Win_start (a->group, 0, a->win), MPI_Win_complete (a->win), (void) 0,
		 MPI_Win_post (a->group, 0, a->win), MPI_Win_wait (a->win))

#define DEFINE_ORIGIN_THREAD_RUNFN(fn, type)                            \
    static void *runfunc_ ## type (ArgStruct* a) {                      \
        int tid = (int) a->tid;                                         \
        size_t max_size = a->max_size;                                  \
        size_t min_size = a->min_size;                                  \
									\
	if (rmamt_bind_threads) {					\
	    rmamt_bind (tid);						\
	}								\
                                                                        \
        thread_etimes[tid] = time_getns ();                             \
                                                                        \
        /* signal the main thread that we are ready */                  \
        thread_barrier (0);                                             \
                                                                        \
        for (uint32_t j = min_size, cycle = 1 ; j <= max_size ; j <<= 1) {     \
            thread_barrier (cycle++);                                   \
                                                                        \
            for (int l = 0 ; l < RMAMT_WARMUP_ITERATIONS ; l++) {       \
                fn (obuf + tid * j, j, MPI_BYTE, a->target, tid *  j,	\
                    j, MPI_BYTE, win[0]);				\
            }                                                           \
                                                                        \
            thread_barrier (cycle++);                                   \
            thread_barrier (cycle++);                                   \
                                                                        \
            for (int l = 0 ; l < rmamt_iterations ; l++) {              \
                fn (obuf + tid * j, j, MPI_BYTE, a->target, tid *  j,	\
                    j, MPI_BYTE, win[0]);				\
            }                                                           \
                                                                        \
            thread_barrier (cycle++);                                   \
        }                                                               \
                                                                        \
        return 0;                                                       \
    }

DEFINE_ORIGIN_THREAD_RUNFN(MPI_Get, get)
DEFINE_ORIGIN_THREAD_RUNFN(MPI_Put, put)
