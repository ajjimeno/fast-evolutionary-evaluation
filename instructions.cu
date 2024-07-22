#ifndef PROGRAM_INSTRUCTIONS_C
#define PROGRAM_INSTRUCTIONS_C

#include "types.cuh"

__device__ int get_0(Run *run, Program **)
{
    return 0;
}
__device__ int get_1(Run *run, Program **)
{
    return 1;
}
__device__ int get_2(Run *run, Program **)
{
    return 2;
}
__device__ int get_3(Run *run, Program **)
{
    return 3;
}
__device__ int get_4(Run *run, Program **)
{
    return 4;
}
__device__ int get_5(Run *run, Program **)
{
    return 5;
}
__device__ int get_6(Run *run, Program **)
{
    return 6;
}
__device__ int get_7(Run *run, Program **)
{
    return 7;
}
__device__ int get_8(Run *run, Program **)
{
    return 8;
}
__device__ int get_9(Run *run, Program **)
{
    return 9;
}

__device__ int input_beginning(Run *run, Program **)
{
    return run->input_x == 0;
}

__device__ int input_read(Run *run, Program **)
{
    Problem p = run->problem;
    int **i = p.input;
    return i[run->input_x][run->input_y];
}

typedef int (*pfunc)(Run *run, Program **p);

__device__ pfunc dev_func_ptr = get_0;

__device__ int (*functions[10])(Run *, Program **) =
    {
        get_0, get_1, get_2, get_3, get_4, get_5, get_6, get_7, get_8, get_9};

Program *copy_programs_to_gpu(int n_programs)
{
    // Example program creation
    Program program = {
        &get_0, // progran
        0,      // arguments
        NULL    // pointer to args
    };

    Program *programs = (Program *)malloc(n_programs * sizeof(struct Problem));

    for (int i = 0; i < n_programs; i++)
    {
        cudaMemcpyFromSymbol(&programs[i].pointer, dev_func_ptr, sizeof(pfunc));
        programs[i].args = program.args;
        programs[i].n_args = program.n_args;
    }

    Program *d_programs;

    cudaMalloc(&d_programs, n_programs * sizeof(struct Problem));

    cudaMemcpy(d_programs, programs, n_programs * sizeof(struct Problem), cudaMemcpyHostToDevice);

    free(programs);

    return d_programs;
}

#endif