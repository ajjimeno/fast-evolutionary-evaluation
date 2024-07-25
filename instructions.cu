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

__device__ int testing_input_beginning(Run *run, Program **)
{
    return run->input_x == 0;
}

__device__ int testing_input_read(Run *run, Program **)
{
    Problem p = run->problem;
    int **i = p.input;
    return i[run->input_x][run->input_y];
}

__device__ int testing_output_write(Run *run, Program **p)
{
    // int value = (p[0]->pointer)(run, p[0]->args);
    run->output[run->output_y][run->output_x] = 0; // value;

    return 0;
}

typedef int (*pfunc)(Run *run, Program **p);

__device__ pfunc dev_func_ptr = get_0;

__device__ pfunc dev_func_ptr_write = testing_output_write;

//__device__ int (*functions[10])(Run *, Program **) =
//    {
//        get_0, get_1, get_2, get_3, get_4, get_5, get_6, get_7, get_8, get_9};

Program *copy_to_device(Program *p)
{
    Program *h_p = (Program *)malloc(sizeof(Program));

    h_p->n_args = p->n_args;

    if (p->n_args>0)
    {
        Program ** args = (Program **)malloc(p->n_args*sizeof(Program *));
        // Save the subprograms in device memory
        for (int i = 0; i < p->n_args; i++)
        {
            args[i] = copy_to_device(p->args[i]);
        }

        cudaMemcpy(&h_p->args, args, p->n_args*sizeof(Program *), cudaMemcpyHostToDevice);

        free(args);
    }

    Program * p_p;

    cudaMemcpyFromSymbol(&h_p->pointer, p->pointer, sizeof(pfunc));

    cudaMemcpy(&p_p, h_p, sizeof(Program), cudaMemcpyHostToDevice);

    return p_p;
}

Program *copy_programs_to_gpu(int n_programs)
{

    // Example program creation
    Program program_get = {
        dev_func_ptr, // progran
        0,            // arguments
        NULL          // pointer to args
    };

    Program **subprogram = new struct Program *[1];
    subprogram[0] = &program_get;

    // Example program creation
    Program program = {
        dev_func_ptr_write, // progran
        1,                  // arguments
        subprogram          // pointer to args
    };

    // Copy programs to device memory
    Program * d_p = copy_to_device(&program);

    // Create array of programs in host memory
    Program *programs = (Program *)malloc(n_programs * sizeof(struct Problem));

    for (int i = 0; i < n_programs; i++)
    {
        //cudaMemcpyFromSymbol(&programs[i].pointer, dev_func_ptr_write, sizeof(pfunc));
        //programs[i].pointer = d_p->pointer;
        //programs[i].args = d_p->args;
        //programs[i].n_args = d_p->n_args;
        cudaMemcpy(&programs[i], d_p, sizeof(Program), cudaMemcpyDeviceToHost);
    }

    // Copy array of programs from host memory to device memory
    Program *d_programs;

    cudaMalloc(&d_programs, n_programs * sizeof(struct Problem));

    cudaMemcpy(d_programs, programs, n_programs * sizeof(struct Problem), cudaMemcpyHostToDevice);

    free(programs);
    

    return d_programs;
}

#endif