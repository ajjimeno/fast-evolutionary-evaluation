
#ifndef PROGRAM_PROBLEMS_C
#define PROGRAM_PROBLEMS_C

#ifndef SETUP_BUILDING_CPU
__device__ float accuracy_calculation(Instance problem, int **output)
#else
float accuracy_calculation(Instance problem, int **output)
#endif
{
    float tp = 0.0;

    // Count number of equal entries
    for (int i = 0; i < problem.gt.y; i++)
    {
        for (int j = 0; j < problem.gt.x; j++)
        {
            if (problem.gt.array[i][j] == output[i][j])
            {
                tp++;
            }
        }
    }

    // Total number of entries
    int total = problem.output.y * problem.output.x;

    return (float)tp / (float)total;
}

#endif