#include <iostream>
#include "instructions.cu"
#include "problems.cu"
#include "types.cuh"

#define N_BLOCKS 50
#define N_THREADS 100

__device__ float run(Program program, Problems *problems)
{
	float total_accuracy = 0.0;

	for (int p = 0; p < problems->n_problems; p++)
	{
		int **output = (int **)malloc(problems->problems[p].output_y * sizeof(int *));

		for (int i = 0; i < problems->problems[p].output_y; i++)
		{
			output[i] = (int *)malloc(problems->problems[p].output_x * sizeof(int));
			for (int j = 0; j < problems->problems[p].output_x; j++)
			{
				output[i][j] = problems->problems[p].output[i][j];
			}
		}

		Run *r = (Run *)malloc(sizeof(Run));

		r[0] = {
			0,					   // input_x
			0,					   // input_y
			0,					   // output_x
			0,					   // output_y
			problems->problems[p], // problem
			output				   // output
		};

		for (int i = 0; i < 100; i++)
		{
			program.pointer(r, program.args);
		}

		total_accuracy += accuracy_calculation(problems->problems[p], output);

		free(r);

		for (int i = 0; i < problems->problems[p].output_y; i++)
		{
			free(output[i]);
		}
		free(output);
	}
	return total_accuracy / (float)problems->n_problems;
}

// Programs, Problems, split programs
__global__ void create_and_run(Program *programs, int n_programs, Problems *problems, float *accuracy)
{
	int programs_per_block = n_programs / (N_BLOCKS * N_THREADS);

	int start = (blockIdx.x * blockDim.x + threadIdx.x) * programs_per_block;
	int end = start + programs_per_block;

	for (int i = start; i < end; i++)
	{

		accuracy[i] = run(programs[i], problems);
	}
}

int main(void)
{
	int device_count = 0;
	// Get the number of CUDA-capable devices
	cudaError_t err = cudaGetDeviceCount(&device_count);

	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
		return 1;
	}
	else
	{
		std::cout << "Device count: " << device_count << std::endl;
	}

	Problems *problems = load_problems();

	int n_programs = 30000;

	float *d_accuracy;
	float *accuracy;

	accuracy = (float *)malloc(n_programs * sizeof(float));

	cudaMalloc(&d_accuracy, n_programs * sizeof(float));

	Program *d_programs = copy_programs_to_gpu(n_programs);

	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("Error launching kernel: %s\n", cudaGetErrorString(err));
	}

	for (int i = 0; i < 1000; i++)
	{
		std::cout << "Starting kernel " << i << std::endl;

		create_and_run<<<N_BLOCKS, N_THREADS>>>(d_programs, n_programs, problems, d_accuracy);

		err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			printf("Error launching kernel: %s\n", cudaGetErrorString(err));
			// Handle the error (e.g., exit the program)
		}
		cudaDeviceSynchronize();
		std::cout << "Kernel finished" << std::endl;
	}
	cudaMemcpy(accuracy, d_accuracy, n_programs * sizeof(float), cudaMemcpyDeviceToHost);

	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("Error launching kernel: %s\n", cudaGetErrorString(err));
		// Handle the error (e.g., exit the program)
	}

	std::cout << std::fixed;
	std::cout << "A[0]: " << accuracy[0] << std::endl;

	float total = 0.0;

	for (int i = 0; i < n_programs; i++)
	{
		total += accuracy[i];

	}

	std::cout << "Total: " << total << " " << n_programs << std::endl;

	cudaFree(d_accuracy);
	cudaFree(d_programs);
	free(accuracy);

	return 0;
}
