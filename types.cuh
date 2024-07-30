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

struct Node
{
    int pointer;
    int n_args;
    int args[3];
};

struct Programs
{
    int n_programs;
    int *programs;

    int n_nodes;
    Node *nodes;
};

struct Run;
typedef int (*pfunc)(Run *run, int *p);

struct Run
{
    int input_x, input_y;
    int output_x, output_y;
    pfunc *pfuncs;
    Problem problem;
    int **output;
    int inner_loop;
    int status;
    int memory;
    Programs *programs;
    int program_offset;
};

#endif