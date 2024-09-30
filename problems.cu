
#ifndef PROGRAM_PROBLEMS_C
#define PROGRAM_PROBLEMS_C

#ifndef SETUP_BUILDING_CPU
__device__ float accuracy_calculation(Instance problem, int **output)
#else
float accuracy_calculation(Instance problem, int **output)
#endif
{
    float tp = 0.0;
    float total = 0.0;

    int count[10];
    for (int i = 0; i < problem.gt.y; i++)
    {
        for (int j = 0; j < problem.gt.x; j++)
        {
            if (problem.gt.array[i][j] >= 0 && problem.gt.array[i][j] < 10)
                count[problem.gt.array[i][j]]++;
        }
    }


    float inv[10];
    for (int i = 0; i < 10; i++)
    {
        // std::cout << i << ":" << count[i] << std::endl;
        // inv[i] = 1;
        if (count[i] == 0)
            inv[i] = 0.0;
        else
            inv[i] = 1.0 / count[i];
    }

    // Count number of equal entries
    for (int i = 0; i < problem.gt.y; i++)
    {
        for (int j = 0; j < problem.gt.x; j++)
        {
            if (problem.gt.array[i][j] == output[i][j])
            {
                tp+=inv[problem.gt.array[i][j]];
            }

            total+=inv[problem.gt.array[i][j]];
        }
    }

    // Total number of entries
    // int total = problem.output.y * problem.output.x;
    if (total == 0)
    {
        return 0.0;
    }

    return (float)tp / (float)total;
}

#endif