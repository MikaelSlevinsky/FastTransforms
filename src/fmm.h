#ifndef FMM_H
#define FMM_H

#include <assert.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>

enum { L2C = 0, C2L = 1, BOTH = 2 };

#ifdef CLOCK_UPTIME_RAW
#define tic clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
#define dtics(a, b) (double)(b - a) / 1.0E9
#define toc(a) (double)(tic - a) / 1.0E9
#else
#define tic clock()
#define dtics(a, b) (double)(b - a) / (double)CLOCKS_PER_SEC
#define toc(a) (double)(tic - a) / (double)CLOCKS_PER_SEC
#endif

#endif