#include <iostream>
#include "instructions.cu"
#include "types.cuh"

#ifndef PROGRAM_RUNNING
#define PROGRAM_RUNNING

#define N_BLOCKS 500
#define N_THREADS 1024



// Programs, Problems, split programs
__global__ void create_and_run(Programs *programs, int n_programs, Instances *problems, float *accuracy, int blocks, int threads)
{
	int programs_per_block = (n_programs / (blocks * threads)) + 1;

	int start = (blockIdx.x * blockDim.x + threadIdx.x) * programs_per_block;
	int end = start + programs_per_block;

	for (int i = start; i < end && i < n_programs; i++)
	{
		accuracy[i] = run(programs, i, problems);
	}
}

int execute_and_evaluate(int n_programs, STRING **programs, float *accuracy, Instances *problems)
{
	cudaError_t err;
	float *d_accuracy;

	cudaMallocManaged(&d_accuracy, n_programs * sizeof(float));

	Programs *d_programs = copy_programs_to_gpu(n_programs, programs);

	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("Error creating programs: %s\n", cudaGetErrorString(err));
		return 1;
	}

	std::cout << "Starting kernel " << n_programs << std::endl;

	int threads = std::min(N_THREADS, n_programs);

	int blocks = std::min((int)(n_programs / threads), N_BLOCKS);

	create_and_run<<<blocks, threads>>>(d_programs, n_programs, problems, d_accuracy, blocks, threads);

	cudaDeviceSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("Error launching kernel: %s\n", cudaGetErrorString(err));
		// Handle the error (e.g., exit the program)
		return 1;
	}
	std::cout << "Kernel finished" << std::endl;

	cudaMemcpy(accuracy, d_accuracy, n_programs * sizeof(float), cudaMemcpyDeviceToHost);

	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("Error launching kernel: %s\n", cudaGetErrorString(err));
		// Handle the error (e.g., exit the program)
		return 1;
	}

	float total = 0.0;

	for (int i = 0; i < n_programs; i++)
	{
		total += accuracy[i];
	}

	std::cout << "Total: " << total << " " << n_programs << std::endl;

	cudaFree(d_accuracy);

	free_programs_from_gpu(d_programs);
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("Error freeing programs: %s\n", cudaGetErrorString(err));
		return 1;
	}

	return 0;
}

/*
int main()
{
	int n_programs = 30000;
	std::vector<std::string> programs;

	for (int i = 0; i < n_programs; i++)
	{
		programs.push_back("prog2(testing_output_write(get0()),testing_output_move_right())");
		// programs.push_back("get0()");
	}

	float *accuracy;
	accuracy = (float *)malloc(n_programs * sizeof(float));

	execute_and_evaluate(n_programs, programs.data(), accuracy);
}*/

#endif