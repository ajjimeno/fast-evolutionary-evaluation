#include <iostream>
#include "instructions.cu"
#include "problems.cu"
#include "types.cuh"

#define N_BLOCKS 50
#define N_THREADS 100

__device__ float run(Programs *programs, int program_id, Problems *problems, pfunc *pfuncs)
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
			0,							   // input_x
			0,							   // input_y
			0,							   // output_x
			0,							   // output_y
			pfuncs,						   // funcs
			problems->problems[p],		   // problem
			output,						   // output
			0,							   // inner_loop
			0,							   // status
			0,							   // memory
			programs,					   // programs
			programs->programs[program_id] // program_offset
		};

		for (int i = 0; i < 100; i++)
		{
			Node node = programs->nodes[programs->programs[program_id]];
			pfuncs[node.pointer](r, node.args);
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
__global__ void create_and_run(Programs *programs, int n_programs, Problems *problems, pfunc *pfuncs, float *accuracy)
{
	int programs_per_block = n_programs / (N_BLOCKS * N_THREADS);

	int start = (blockIdx.x * blockDim.x + threadIdx.x) * programs_per_block;
	int end = start + programs_per_block;

	for (int i = start; i < end; i++)
	{

		accuracy[i] = run(programs, i, problems, pfuncs);
	}
}

int execute_and_evaluate(int n_programs, std::string *programs, float *accuracy)
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

	float *d_accuracy;

	cudaMallocManaged(&d_accuracy, n_programs * sizeof(float));

	pfunc *d_pfuncs;
	cudaMallocManaged(&d_pfuncs, 2 * sizeof(pfunc));

	fill_function_pointers<<<1, 1>>>(d_pfuncs);
	cudaDeviceSynchronize();

	Programs *d_programs = copy_programs_to_gpu(n_programs, programs);

	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("Error creating programs: %s\n", cudaGetErrorString(err));
		return 1;
	}
	std::cout << "Starting kernel " << std::endl;

	create_and_run<<<N_BLOCKS, N_THREADS>>>(d_programs, n_programs, problems, d_pfuncs, d_accuracy);

	cudaDeviceSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("Error launching kernel: %s\n", cudaGetErrorString(err));
		// Handle the error (e.g., exit the program)
		return 1;
	}
	std::cout << "Kernel finished" << std::endl;

	free_programs_from_gpu(d_programs);
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("Error freeing programs: %s\n", cudaGetErrorString(err));
		return 1;
	}

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

	cudaFree(d_pfuncs);
	cudaFree(d_accuracy);

	return 0;
}

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
}