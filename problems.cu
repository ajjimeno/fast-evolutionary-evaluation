
#ifndef PROGRAM_PROBLEMS_C
#define PROGRAM_PROBLEMS_C

Problems *load_problems()
{
    int in[3][3] = {{0, 0, 5},
                    {0, 5, 0},
                    {5, 0, 0}};

    int out[3][3] = {{3, 3, 3},
                     {4, 4, 4},
                     {2, 2, 2}};

    // Example set creation
    Problem problem = {
        (int **)in,  // input
        3,           // input_x
        3,           // input_y
        (int **)out, // out_gt
        (int **)out, // output
        3,           // output_x
        3            // output_y
    };

    int **d_in = (int **)malloc(problem.input_y * sizeof(int *));

    for (int i = 0; i < problem.input_y; i++)
    {
        cudaMalloc(&d_in[i], problem.input_x * sizeof(int));
        cudaMemcpy(d_in[i], in[i], problem.input_x * sizeof(int), cudaMemcpyHostToDevice);
    }

    cudaMalloc(&problem.input, problem.input_y * sizeof(int *));
    cudaMemcpy(problem.input, d_in, problem.input_y * sizeof(int *), cudaMemcpyHostToDevice);

    free(d_in);

    int **d_output = (int **)malloc(problem.output_y * sizeof(int *));

    for (int i = 0; i < problem.output_y; i++)
    {
        cudaMalloc(&d_output[i], problem.output_x * sizeof(int));
        cudaMemcpy(d_output[i], out[i], problem.output_x * sizeof(int), cudaMemcpyHostToDevice);
    }

    cudaMalloc(&problem.output, problem.output_y * sizeof(int *));
    cudaMemcpy(problem.output, d_in, problem.output_y * sizeof(int *), cudaMemcpyHostToDevice);

    free(d_output);

    int **d_output_gt = (int **)malloc(problem.output_y * sizeof(int *));

    for (int i = 0; i < problem.output_y; i++)
    {
        cudaMalloc(&d_output_gt[i], problem.output_x * sizeof(int));
        cudaMemcpy(d_output_gt[i], out[i], problem.output_x * sizeof(int), cudaMemcpyHostToDevice);
    }

    cudaMalloc(&problem.out_gt, problem.output_y * sizeof(int *));
    cudaMemcpy(problem.out_gt, d_in, problem.output_y * sizeof(int *), cudaMemcpyHostToDevice);

    free(d_output_gt);

    Problem *d_problem;
    cudaMalloc(&d_problem, sizeof(Problem));
    cudaMemcpy(d_problem, &problem, sizeof(Problem), cudaMemcpyHostToDevice);

    Problems h_problems;

    h_problems.n_problems = 1;

    cudaMalloc(&h_problems.problems, sizeof(Problem));
    cudaMemcpy(h_problems.problems, d_problem, sizeof(Problem), cudaMemcpyHostToDevice);

    Problems *d_problems;
    cudaMalloc(&d_problems, sizeof(Problems));
    cudaMemcpy(d_problems, &h_problems, sizeof(Problems), cudaMemcpyHostToDevice);

    return d_problems;
}

__device__ float accuracy_calculation(Problem problem, int **output)
{
    float tp = 0.0;

    // Count number of equal entries
    for (int i = 0; i < problem.output_y; i++)
    {
        for (int j = 0; j < problem.output_x; j++)
        {
            if (problem.out_gt[i][j] == output[i][j])
            {
                tp++;
            }
        }
    }

    // Total number of entries
    int total = problem.output_y * problem.output_x;

    return (float)tp / (float)total;
}

#endif