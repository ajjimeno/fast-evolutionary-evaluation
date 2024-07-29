#ifndef PROGRAM_TYPES_H
#define PROGRAM_TYPES_H

struct Problem
{
    int **input;
    int input_x, input_y;

    int **out_gt;

    int **output;
    int output_x, output_y;
};

struct Problems
{
    int n_problems;
    Problem *problems;
};
struct Program
{
    int pointer; // int (*pointer)(Run *, Program **);
    int n_args;
    struct Program **args;
};

struct Run; 
typedef int (*pfunc)(Run *run, Program **p);

struct Run
{
    int input_x, input_y;
    int output_x, output_y;
    pfunc * pfuncs;
    Problem problem;
    int **output;
    int inner_loop;
    int status;
    int memory;
};

#endif