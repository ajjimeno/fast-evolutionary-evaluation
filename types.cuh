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

struct Run
{
    int input_x, input_y;
    int output_x, output_y;

    Problem problem;
    int **output;
};

struct Program
{
    int (*pointer)(Run *, Program **);
    int n_args;
    struct Program **args;
};

#endif