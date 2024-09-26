#ifndef PROGRAM_TYPES_H
#define PROGRAM_TYPES_H

#include <opencv2/opencv.hpp>


#define STRING std::string_view

struct Array
{
    int y;
    int x;
    int **array;
};

struct Instance
{
    Array **training;
    int n_training;
    Array input;
    Array output;
    Array gt;
    Array initial;
};

struct Instances
{
    int n_instances;
    Instance *instances;
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

struct BoundingBox
{
    int top;
    int left;
    int width;
    int height;
};

struct Color
{
    int value;
    int frequency;
};

struct Shape
{
    int y;
    int x;

    int area;

    BoundingBox box;

    Color *Color;
};

struct InputInstance
{
    int min;
    int max;
};


struct Run
{
    int input_x, input_y;
    int output_x, output_y;
    // pfunc *pfuncs;
    Instance problem;
    int **output;
    int inner_loop;
    int status;
    int memory;
    // Programs *programs;
    // long program_offset;
    int training_id;
    int training_input_x, training_input_y;
    int training_output_x, training_output_y;
    Node *nodes;
    InputInstance inputInstance;
};

#ifndef SETUP_BUILDING_CPU
#include <cuda.h>

void allocate_memory(void **output, size_t size)
{
    cudaMallocManaged(output, size);
}

void free_memory(void *output)
{
    cudaFree(output);
}

#else

void allocate_memory(void **output, size_t size)
{
    *output = malloc(size);
}

void free_memory(void *output)
{
    free(output);
}

#endif

#endif