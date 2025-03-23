#include <iostream>
#include <chrono>

#include "instructions.cu"
#include "types.cuh"

#ifndef PROGRAM_RUNNING
#define PROGRAM_RUNNING

#define TEST_SPEED

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

    //std::cout << "Starting kernel " << n_programs << std::endl;

    int c = cores();

#ifdef TEST_SPEED
    auto start = std::chrono::high_resolution_clock::now();
    c=2;
#endif

    std::vector<std::thread> threads;

    //std::cout << "Num_cores " << cores() << std::endl;
    int n_threads = std::min(n_programs, c);
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

#ifdef TEST_SPEED
    // Record end time
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate duration
    std::chrono::duration<double> elapsed = end - start;

    // Output the elapsed time
    std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";
#endif
    //std::cout << "Kernel finished" << std::endl;

    float total = 0.0;

    for (int i = 0; i < n_programs; i++)
    {
        total += accuracy[i];
    }

    //std::cout << "Total: " << total << " " << n_programs << std::endl;

    free_programs_from_gpu(d_programs);

    return 0;
}

#endif
