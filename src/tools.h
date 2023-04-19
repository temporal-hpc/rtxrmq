#pragma once
#include <unistd.h>
#include <string>
#include <time.h>
#include <getopt.h>

#define ARG_BS 1
#define ARG_NB 2
#define ARG_REPS 3
#define ARG_DEV 4
#define ARG_NT 5
#define ARG_SEED 6
#define ARG_CHECK 7
#define ARG_TIME 8
#define ARG_POWER 9

struct VBHMem {
    size_t out_buffer;
    size_t temp_buffer;
};

struct CmdArgs {
    int n, q, lr, alg, bs, nb, reps, dev, nt, seed, check, save_time, save_power;
    std::string time_file, power_file;
};

#define NUM_REQUIRED_ARGS 10
void print_help(){
    fprintf(stderr, AC_BOLDGREEN "run as ./rtxrmq <n> <q> <lr> <alg>\n\n" AC_RESET
                    "n   = num elements\n"
                    "q   = num RMQ querys\n"
                    "lr  = length of range; min 1, max n\n"
                    "  >0 -> value\n"
                    "  -1 -> uniform distribution (large values)\n"
                    "  -2 -> lognormal distribution (medium values)\n"
                    "  -3 -> lognormal distribution (small values)\n"
                    "alg = algorithm\n"
                    "   0 -> %s\n"
                    "   1 -> %s\n"
                    "   2 -> %s\n"
                    "   3 -> %s\n"
                    "   4 -> %s\n"
                    "   5 -> %s\n"
                    "   6 -> %s\n"
                    "   7 -> %s\n"
                    "   8 -> %s\n"
                    "   9 -> %s\n\n"
                    "Options:\n"
                    "   --bs <block size>         block size for RTX_blocks (default: 2^15)\n"
                    "   --nb <#blocks>            number of blocks for RTX_blocks (overrides --bs)\n"
                    "   --reps <repetitions>      RMQ repeats for the avg time (default: 10)\n"
                    "   --dev <device ID>         device ID (default: 0)\n"
                    "   --nt  <thread num>        number of CPU threads\n"
                    "   --seed <seed>             seed for PRNG\n"
                    "   --check                   check correctness\n"
                    "   --save-time=<file>        \n"
                    "   --save-power=<file>       \n",
                    algStr[0],
                    algStr[1],
                    algStr[2],
                    algStr[3],
                    algStr[4],
                    algStr[5],
                    algStr[6],
                    algStr[7],
                    algStr[8],
                    algStr[9]
                );
}


CmdArgs get_args(int argc, char *argv[]) {
    if (argc < 5) {
        print_help();
        exit(EXIT_FAILURE);
    }

    CmdArgs args;
    args.n = atoi(argv[1]);
    args.q = atoi(argv[2]);
    args.lr = atoi(argv[3]);
    args.alg = atoi(argv[4]);
    if (!args.n || !args.q || !args.lr) {
        print_help();
        exit(EXIT_FAILURE);
    }
    if (args.lr > args.n) {
        fprintf(stderr, "Error: lr=%i > n=%i  (lr must be between '1' and 'n')\n", args.lr, args.n);
        exit(EXIT_FAILURE);
    }

    args.bs = 1<<15;
    args.nb = args.n / args.bs;
    args.reps = 10;
    args.seed = time(0);
    args.dev = 0;
    args.check = 0;
    args.save_time = 0;
    args.save_power = 0;
    args.nt = 1;
    args.time_file = "";
    args.power_file = "";
    
    static struct option long_option[] = {
        // {name , has_arg, flag, val}
        {"bs", required_argument, 0, ARG_BS},
        {"nb", required_argument, 0, ARG_NB},
        {"reps", required_argument, 0, ARG_REPS},
        {"dev", required_argument, 0, ARG_DEV},
        {"nt", required_argument, 0, ARG_NT},
        {"seed", required_argument, 0, ARG_SEED},
        {"check", no_argument, 0, ARG_CHECK},
        {"save-time", optional_argument, 0, ARG_TIME},
        {"save-power", optional_argument, 0, ARG_POWER},
    };
    int opt, opt_idx;
    while ((opt = getopt_long(argc, argv, "12345", long_option, &opt_idx)) != -1) {
        if (isdigit(opt))
                continue;
        switch (opt) {
            case ARG_BS:
                args.bs = min(args.n, atoi(optarg));
                args.nb = args.n / args.bs;
                break;
            case ARG_NB:
                args.nb = min(args.n, atoi(optarg));
                args.bs = args.n / args.nb;
                break;
            case ARG_REPS:
                args.reps = atoi(optarg);
                break;
            case ARG_DEV:
                args.dev = atoi(optarg);
                break;
            case ARG_NT: 
                args.nt = atoi(optarg);
                break;
            case ARG_SEED:
                args.seed = atoi(optarg);
                break;
            case ARG_CHECK:
                args.check = 1;
                break;
            case ARG_TIME:
                args.save_time = 1;
                if (optarg != NULL)
                    args.time_file = optarg;
                break;
            case ARG_POWER:
                args.save_power = 1;
                if (optarg != NULL)
                    args.power_file = optarg;
                break;
            default:
                break;
        }
    }

    if (args.alg != ALG_GPU_RTX_CAST &&
            args.alg != ALG_GPU_RTX_TRANS &&
            args.alg != ALG_GPU_RTX_BLOCKS &&
            args.alg != ALG_GPU_RTX_LUP &&
            args.alg != ALG_GPU_RTX_IAS &&
	    args.alg != ALG_GPU_RTX_IAS_TRANS) {
        args.bs = 0;
        args.nb = 0;
    }

    printf( "Params:\n"
            "   reps = %i\n"
            "   seed = %i\n"
            "   dev  = %i\n"
            AC_GREEN "   n    = %i (~%f GB, float)\n" AC_RESET
            "   bs   = %i\n"
            AC_GREEN "   q    = %i (~%f GB, int2)\n" AC_RESET
            "   lr   = %i\n"
            "   nt   = %i CPU threads\n"
            "   alg  = %i (%s)\n\n",
            args.reps, args.seed, args.dev, args.n, sizeof(float)*args.n/1e9, args.bs, args.q,
            sizeof(int2)*args.q/1e9, args.lr, args.nt, args.alg, algStr[args.alg]);

    return args;
}

bool is_equal(float a, float b) {
    float epsilon = 1e-4f;
    return abs(a - b) < epsilon;
}

bool check_result(float *hA, int2 *hQ, int q, float *expected, float *result, int *indices){
    bool pass = true;
    for (int i = 0; i < q; ++i) {
        //if (expected[i] != result[i]) { // RT-cores don't introduce floating point errors
        if (!is_equal(expected[i], result[i])) {
            printf("Error on %i-th query: got %f, expected %f at idx %i\n", i, result[i], expected[i], indices[i]);
            printf("  [%i,%i]\n", hQ[i].x, hQ[i].y);
            pass = false;
            //for (int j = hQ[i].x; j <= hQ[i].y; ++j) {
            //    printf("%f ", hA[j]);
            //}
            //printf("\n");
            //return false;
        }
    }
    //for (int j = 0; j <= 1<<24; ++j) {
    //    printf("%f\n", hA[j]);
    //}
    return pass;
}

bool check_result(float *hA, int2 *hQ, int q, int *expected, int *result){
    for (int i = 0; i < q; ++i) {
        if (expected[i] != result[i]) {
            printf("Error on %i-th query: got %i, expected %i\n", i, result[i], expected[i]);
            //printf("[%i,%i]\n", hQ[i].x, hQ[i].y);
            //for (int j = hQ[i].x; j <= hQ[i].y; ++j) {
            //    printf("%f ", hA[j]);
            //}
            //printf("\n");
            //return false;
        }
    }
    return true;
}

void print_gpu_specs(int dev){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("Device Number: %d\n", dev);
    printf("  Device name:                  %s\n", prop.name);
    printf("  Memory:                       %f GB\n", prop.totalGlobalMem/(1024.0*1024.0*1024.0));
    printf("  Multiprocessor Count:         %d\n", prop.multiProcessorCount);
    printf("  Concurrent Kernels:           %s\n", prop.concurrentKernels == 1? "yes" : "no");
    printf("  Memory Clock Rate:            %d MHz\n", prop.memoryClockRate);
    printf("  Memory Bus Width:             %d bits\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth:        %f GB/s\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
}

void write_results(int dev, int alg, int n, int bs, int q, int lr, int reps, CmdArgs args) {
    if (!args.save_time) return;
    std::string filename;
    if (args.time_file.empty())
        filename = std::string("../results/data.csv");
    else
        filename = args.time_file;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    char *device = prop.name;
    if (alg == ALG_CPU_BASE || alg == ALG_CPU_HRMQ) {
        strcpy(device, "CPU ");
        char hostname[50];
        gethostname(hostname, 50);
        strcat(device, hostname);
    }

    FILE *fp;
    fp = fopen(filename.c_str(), "a");
    fprintf(fp, "%s,%s,%i,%i,%i,%i,%i",
            device,
            algStr[alg],
            reps,
            n,
            bs,
            q,
            lr);
    fclose(fp);
}

void write_results(float time_ms, int q, float construction_time, int reps, CmdArgs args) {
    if (!args.save_time) return;
    std::string filename;
    if (args.time_file.empty())
        filename = std::string("../results/data.csv");
    else
        filename = args.time_file;

    float time_it = time_ms/reps;
    FILE *fp;
    fp = fopen(filename.c_str(), "a");
    fprintf(fp, ",%f,%f,%f,%f,0,0\n",
            time_ms/1000.0,
            (double)q/(time_it/1000.0),
            (double)time_it*1e6/q,
            construction_time);
    fclose(fp);
}
void write_results(float time_ms, int q, float construction_time, int reps, CmdArgs args, VBHMem mem) {
    if (!args.save_time) return;
    std::string filename;
    if (args.time_file.empty())
        filename = std::string("../results/data.csv");
    else
        filename = args.time_file;

    float time_it = time_ms/reps;
    FILE *fp;
    fp = fopen(filename.c_str(), "a");
    fprintf(fp, ",%f,%f,%f,%f,%f,%f\n",
            time_ms/1000.0,
            (double)q/(time_it/1000.0),
            (double)time_it*1e6/q,
            construction_time,
            mem.out_buffer/1e6,
            mem.temp_buffer/1e6);
    fclose(fp);
}

