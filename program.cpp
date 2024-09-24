#include <iostream>
#include "instructions.cu"
#include "problems.cu"
#include "types.cuh"

#ifndef PROGRAM_RUNNING
#define PROGRAM_RUNNING

#define MAX_OUTPUT_SIZE 40

float run(Programs *programs, int program_id, Instances *problems)
{
    float total_accuracy = 0.0;

    // Limit program size to 1000 nodes
    Node program[1000];

    int pointer_end;
    if ((program_id + 1) == programs->n_programs)
        pointer_end = programs->n_nodes;
    else
        pointer_end = programs->programs[program_id + 1];

    for (int i = 0; i < pointer_end - programs->programs[program_id]; i++)
    {
        program[i] = programs->nodes[programs->programs[program_id] + i];
    }

    int myArray[MAX_OUTPUT_SIZE][MAX_OUTPUT_SIZE];
    int *output[MAX_OUTPUT_SIZE];
    for (int i = 0; i < MAX_OUTPUT_SIZE; i++)
    {
        output[i] = myArray[i];
    }

    for (int p = 0; p < problems->n_instances; p++)
    {
        for (int i = 0; i < problems->instances[p].initial.y; i++)
        {
            for (int j = 0; j < problems->instances[p].initial.x; j++)
            {
                if (j < problems->instances[p].input.x && i < problems->instances[p].input.y)
                { output[i][j] = problems->instances[p].input.array[i][j]; }
                else
                { output[i][j] = 0; }
            }
        }

        Run r = {
            0,                      // input_x
            0,                      // input_y
            0,                      // output_x
            0,                      // output_y
            problems->instances[p], // problem
            output,                 // output
            0,                      // inner_loop
            0,                      // status
            0,                      // memory
            0,                      // training_id
            0,                      // training_input_x
            0,                      // training_input_y
            0,                      // training_output_x
            0,                      // training_output_y
            program};

        r.inputInstance.max = input_max(&r);
        r.inputInstance.min = input_min(&r);

        for (int i = 0; i < 200; i++)
        {
            function_switch(0, &r);

            //if (r.status != 0)
            //    break;
        }

        total_accuracy += accuracy_calculation(problems->instances[p], output);
    }
    return total_accuracy / (float)problems->n_instances;
}

// Programs, Problems, split programs
void create_and_run(Programs *programs, int n_programs, Instances *problems, float *accuracy, int start, int end)
{
    for (int i = start; i < end && i < n_programs; i++)
    {
        accuracy[i] = run(programs, i, problems);
    }
}

int execute_and_evaluate(int n_programs, STRING **programs, float *accuracy, Instances *problems)
{
    Programs *d_programs = copy_programs_to_gpu(n_programs, programs);

    std::cout << "Starting kernel " << n_programs << std::endl;

    std::vector<std::thread> threads;

    int n_threads = std::min(n_programs, 20);
    int chunk_size = n_programs / n_threads;

    for (int i = 0; i < n_threads; ++i)
    {
        int start_index = i * chunk_size;
        int end_index = (i == n_threads - 1) ? n_programs : (i + 1) * chunk_size;

        threads.emplace_back(create_and_run, d_programs, n_programs, problems, accuracy, start_index, end_index);
    }

    for (auto &t : threads)
    {
        t.join();
    }


    std::cout << "Kernel finished" << std::endl;

    float total = 0.0;

    for (int i = 0; i < n_programs; i++)
    {
        total += accuracy[i];
    }

    std::cout << "Total: " << total << " " << n_programs << std::endl;

    free_programs_from_gpu(d_programs);

    return 0;
}

#endif
