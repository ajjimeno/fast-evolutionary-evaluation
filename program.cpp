#include <iostream>
#include "instructions.cu"
#include "types.cuh"

#ifndef PROGRAM_RUNNING
#define PROGRAM_RUNNING

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

    int n_threads = std::min(n_programs, 40);
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
