#ifndef PROGRAM_INSTRUCTIONS_C
#define PROGRAM_INSTRUCTIONS_C

#include "types.cuh"
#include <thread>
#include <unordered_map>
#include <vector>

__device__ int get0(Run *run, int *)
{
    return 0;
}
__device__ int get1(Run *run, int *)
{
    return 1;
}
__device__ int get2(Run *run, int *)
{
    return 2;
}
__device__ int get3(Run *run, int *)
{
    return 3;
}
__device__ int get4(Run *run, int *)
{
    return 4;
}
__device__ int get5(Run *run, int *)
{
    return 5;
}
__device__ int get6(Run *run, int *)
{
    return 6;
}
__device__ int get7(Run *run, int *)
{
    return 7;
}
__device__ int get8(Run *run, int *)
{
    return 8;
}
__device__ int get9(Run *run, int *)
{
    return 9;
}

__device__ int get_testing_length_input_x(Run *run, int *)
{
    return run->problem.input_x;
}

__device__ int get_testing_length_input_y(Run *run, int *)
{
    return run->problem.input_y;
}

__device__ int get_testing_length_output_x(Run *run, int *)
{
    return run->problem.output_x;
}

__device__ int get_testing_length_output_y(Run *run, int *)
{
    return run->problem.output_y;
}

__device__ int get_testing_input_position_y(Run *run, int *)
{
    return run->input_y;
}

__device__ int get_testing_input_position_x(Run *run, int *)
{
    return run->input_x;
}

__device__ int get_testing_output_position_y(Run *run, int *)
{
    return run->output_y;
}

__device__ int get_testing_output_position_x(Run *run, int *)
{
    return run->output_x;
}

__device__ int testing_input_max(Run *run, int *)
{
    int *arr = run->problem.input[run->input_y];
    int max = arr[0];
    for (int i = 1; i < run->problem.output_x; i++)
    {
        if (arr[i] > max)
        {
            max = arr[i];
        }
    }
    return max;
}

__device__ int testing_input_min(Run *run, int *)
{
    int *arr = run->problem.input[run->input_y];
    int min = arr[0];
    for (int i = 1; i < run->problem.output_x; i++)
    {
        if (arr[i] < min)
        {
            min = arr[i];
        }
    }
    return min;
}

__device__ int testing_input_read(Run *run, int *)
{
    return run->problem.input[run->input_y][run->input_x];
}

__device__ int testing_output_read_previous(Run *run, int *)
{
    if (run->output_x > 0)
        return run->output[run->output_y][run->output_x - 1];

    return -1;
}

__device__ int testing_output_read(Run *run, int *)
{
    return run->output[run->output_y][run->output_x];
}

__device__ int testing_reset_input_position(Run *run, int *)
{
    run->input_x = 0;
    return 0;
}

__device__ int testing_reset_input_down_position(Run *run, int *)
{
    run->input_y = 0;
    return 0;
}

__device__ int testing_output_write_previous(Run *run, int *p)
{

    if (run->output_x > 0)
    {
        int node_id = run->program_offset + p[0];
        Node node = run->programs->nodes[node_id];

        int value = run->pfuncs[(node.pointer)](run, node.args);
        run->output[run->output_y][run->output_x - 1] = value;
    }

    return 0;
}

__device__ int testing_output_write(Run *run, int *p)
{
    int node_id = run->program_offset + p[0];
    Node node = run->programs->nodes[node_id];
    int value = run->pfuncs[(node.pointer)](run, node.args);
    run->output[run->output_y][run->output_x] = value;

    return 0;
}

__device__ int testing_reset_output_position(Run *run, int *)
{
    run->output_x = 0;
    return 0;
}

__device__ int testing_reset_output_down_position(Run *run, int *)
{
    run->output_y = 0;
    return 0;
}

__device__ int testing_output_move_left(Run *run, int *)
{
    if (run->output_x > 0)
        run->output_x--;

    return 0;
}

__device__ int testing_output_move_right(Run *run, int *)
{
    if (run->output_x < (run->problem.output_x - 1))
        run->output_x++;

    return 0;
}

__device__ int testing_output_move_down(Run *run, int *)
{
    if (run->output_y < (run->problem.output_y - 1))
        run->output_y++;
    // else
    //     status = -1;
    return 0;
}

__device__ int testing_output_move_up(Run *run, int *)
{
    if (run->output_y > 0)
        run->output_y--;

    return 0;
}

__device__ int testing_is_output_end(Run *run, int *)
{
    return run->output_x == (run->problem.output_x - 1);
}

__device__ int testing_is_output_down(Run *run, int *)
{
    return run->output_y == (run->problem.output_y - 1);
}

__device__ int testing_input_move_left(Run *run, int *)
{
    if (run->input_x > 0)
        run->input_x--;

    return 0;
}

__device__ int testing_input_move_right(Run *run, int *)
{
    if (run->input_x < (run->problem.input_x - 1))
        run->input_x++;

    return 0;
}

__device__ int testing_input_move_down(Run *run, int *)
{
    if (run->input_y < (run->problem.input_y - 1))
        run->input_y++;

    return 0;
}

__device__ int testing_input_move_up(Run *run, int *)
{
    if (run->input_y > 0)
        run->input_y--;

    return 0;
}

__device__ int comparison(Run *run, int *p)
{
    int node_0_id = run->program_offset + p[0];
    Node node_0 = run->programs->nodes[node_0_id];

    if (run->pfuncs[(node_0.pointer)](run, node_0.args) == 1)
    {
        int node_1_id = run->program_offset + p[1];
        Node node_1 = run->programs->nodes[node_1_id];
        run->pfuncs[(node_1.pointer)](run, node_1.args);
    }
    else
    {
        int node_2_id = run->program_offset + p[2];
        Node node_2 = run->programs->nodes[node_2_id];
        run->pfuncs[(node_2.pointer)](run, node_2.args);
    }

    return 0;
}

__device__ int bigger_than_output_next(Run *run, int *p)
{
    /*
    if (output_position_x < (get_length_output_x(NULL) - 1))
    {
        return data->training.at(input_id).at(1).array[output_position_y][output_position_x] >
               data->training.at(input_id).at(1).array[output_position_y][output_position_x + 1];
    }*/

    return 0;
}

__device__ int bigger_than_testing_output_next(Run *run, int *)
{
    if (run->output_x < (run->problem.output_x - 1))
    {
        return run->output[run->output_y][run->output_x] >
               run->output[run->output_y][run->output_x + 1];
    }

    return 0;
}

__device__ int swap_testing_output_next(Run *run, int *)
{
    if (!testing_is_output_end(run, NULL))
    {
        int tmp = run->output[run->output_y][run->output_x];
        run->output[run->output_y][run->output_x] = run->output[run->output_y][run->output_x + 1];
        run->output[run->output_y][run->output_x + 1] = tmp;
    }

    return 0;
}

__device__ int bigger_than(Run *run, int *p)
{
    int node_0_id = run->program_offset + p[0];
    Node node_0 = run->programs->nodes[node_0_id];
    int output1 = run->pfuncs[(node_0.pointer)](run, node_0.args);

    int node_1_id = run->program_offset + p[1];
    Node node_1 = run->programs->nodes[node_1_id];
    int output2 = run->pfuncs[(node_1.pointer)](run, node_1.args);
    return output1 > output2;
}

__device__ int equal(Run *run, int *p)
{
    int node_0_id = run->program_offset + p[0];
    Node node_0 = run->programs->nodes[node_0_id];
    int output1 = run->pfuncs[(node_0.pointer)](run, node_0.args);

    int node_1_id = run->program_offset + p[1];
    Node node_1 = run->programs->nodes[node_1_id];
    int output2 = run->pfuncs[(node_1.pointer)](run, node_1.args);
    return output1 == output2;
}

__device__ int no(Run *run, int *p)
{
    int node_0_id = run->program_offset + p[0];
    Node node_0 = run->programs->nodes[node_0_id];
    int output1 = run->pfuncs[(node_0.pointer)](run, node_0.args);
    return !output1;
}

__device__ int prog2(Run *run, int *p)
{
    int node_0_id = run->program_offset + p[0];
    Node node_0 = run->programs->nodes[node_0_id];
    run->pfuncs[(node_0.pointer)](run, node_0.args);

    int node_1_id = run->program_offset + p[1];
    Node node_1 = run->programs->nodes[node_1_id];
    run->pfuncs[(node_1.pointer)](run, node_1.args);
    return 0;
}

__device__ int prog3(Run *run, int *p)
{

    int node_0_id = run->program_offset + p[0];
    Node node_0 = run->programs->nodes[node_0_id];
    run->pfuncs[(node_0.pointer)](run, node_0.args);

    int node_1_id = run->program_offset + p[1];
    Node node_1 = run->programs->nodes[node_1_id];
    run->pfuncs[(node_1.pointer)](run, node_1.args);

    int node_2_id = run->program_offset + p[2];
    Node node_2 = run->programs->nodes[node_2_id];
    run->pfuncs[(node_2.pointer)](run, node_2.args);

    return 0;
}

int loop(Run *run, int *p)
{
    int node_0_id = run->program_offset + p[0];
    Node node_0 = run->programs->nodes[node_0_id];
    int v = run->pfuncs[(node_0.pointer)](run, node_0.args);

    run->inner_loop++;

    if (run->inner_loop < 5 && v > 0 && v <= 30)
    {
        for (int i = 0; i < v; i++)
        {
            int node_1_id = run->program_offset + p[1];
            Node node_1 = run->programs->nodes[node_1_id];
            run->pfuncs[(node_1.pointer)](run, node_1.args);
        }
    }
    else
    {
        run->status = -2;
    }

    run->inner_loop--;

    return 0;
}

__device__ int dowhile(Run *run, int *p)
{
    int c = 0;

    run->inner_loop++;

    if (run->inner_loop < 3)
    {
        int node_0_id = run->program_offset + p[0];
        Node node_0 = run->programs->nodes[node_0_id];
        while (!run->pfuncs[(node_0.pointer)](run, node_0.args) && c < 10)
        {
            c++;
            int node_1_id = run->program_offset + p[1];
            Node node_1 = run->programs->nodes[node_1_id];
            run->pfuncs[(node_1.pointer)](run, node_1.args);
        }
    }
    else
    {
        run->status = -2;
    }

    run->inner_loop--;

    return 0;
}

__device__ int read_memory(Run *run, int *)
{
    return run->memory;
}

__device__ int write_memory(Run *run, int *p)
{
    int node_0_id = run->program_offset + p[0];
    Node node_0 = run->programs->nodes[node_0_id];
    int value = run->pfuncs[(node_0.pointer)](run, node_0.args);

    run->memory = value;

    return 0;
}

#define MAP_INSTRUCTIONS std::unordered_map<std::string, int>

__global__ void fill_function_pointers(pfunc *pfuncs)
{
    pfuncs[0] = get0;
    pfuncs[1] = get1;
    pfuncs[2] = get2;
    pfuncs[3] = get3;
    pfuncs[4] = get4;
    pfuncs[5] = get5;
    pfuncs[6] = get6;
    pfuncs[7] = get7;
    pfuncs[8] = get8;
    pfuncs[9] = get9;
    pfuncs[10] = get_testing_length_input_x;
    pfuncs[11] = get_testing_length_input_y;
    pfuncs[12] = get_testing_length_output_x;
    pfuncs[13] = get_testing_length_output_y;
    pfuncs[14] = get_testing_input_position_y;
    pfuncs[15] = get_testing_input_position_x;
    pfuncs[16] = get_testing_output_position_y;
    pfuncs[17] = get_testing_output_position_x;
    pfuncs[18] = testing_input_max;
    pfuncs[19] = testing_input_min;
    pfuncs[20] = testing_input_read;
    pfuncs[21] = testing_output_read_previous;
    pfuncs[22] = testing_output_read;
    pfuncs[23] = testing_reset_input_position;
    pfuncs[24] = testing_reset_input_down_position;
    pfuncs[25] = testing_output_write_previous;
    pfuncs[26] = testing_output_write;
    pfuncs[27] = testing_reset_output_position;
    pfuncs[28] = testing_reset_output_down_position;
    pfuncs[29] = testing_output_move_left;
    pfuncs[30] = testing_output_move_right;
    pfuncs[31] = testing_output_move_down;
    pfuncs[32] = testing_output_move_up;
    pfuncs[33] = testing_is_output_end;
    pfuncs[34] = testing_is_output_down;
    pfuncs[35] = testing_input_move_left;
    pfuncs[36] = testing_input_move_right;
    pfuncs[37] = testing_input_move_down;
    pfuncs[38] = testing_input_move_up;
    pfuncs[39] = comparison;
    pfuncs[40] = bigger_than_output_next;
    pfuncs[41] = bigger_than_testing_output_next;
    pfuncs[42] = swap_testing_output_next;
    pfuncs[43] = bigger_than;
    pfuncs[44] = equal;
    pfuncs[45] = no;
    pfuncs[46] = prog2;
    pfuncs[47] = prog3;
    pfuncs[48] = dowhile;
    pfuncs[49] = read_memory;
    pfuncs[50] = write_memory;
}

MAP_INSTRUCTIONS get_map()
{

    MAP_INSTRUCTIONS map;

    map["get0"] = 0;
    map["get1"] = 1;
    map["get2"] = 2;
    map["get3"] = 3;
    map["get4"] = 4;
    map["get5"] = 5;
    map["get6"] = 6;
    map["get7"] = 7;
    map["get8"] = 8;
    map["get9"] = 9;
    map["get_testing_length_input_x"] = 10;
    map["get_testing_length_input_y"] = 11;
    map["get_testing_length_output_x"] = 12;
    map["get_testing_length_output_y"] = 13;
    map["get_testing_input_position_y"] = 14;
    map["get_testing_input_position_x"] = 15;
    map["get_testing_output_position_y"] = 16;
    map["get_testing_output_position_x"] = 17;
    map["testing_input_max"] = 18;
    map["testing_input_min"] = 19;
    map["testing_input_read"] = 20;
    map["testing_output_read_previous"] = 21;
    map["testing_output_read"] = 22;
    map["testing_reset_input_position"] = 23;
    map["testing_reset_input_down_position"] = 24;
    map["testing_output_write_previous"] = 25;
    map["testing_output_write"] = 26;
    map["testing_reset_output_position"] = 27;
    map["testing_reset_output_down_position"] = 28;
    map["testing_output_move_left"] = 29;
    map["testing_output_move_right"] = 30;
    map["testing_output_move_down"] = 31;
    map["testing_output_move_up"] = 32;
    map["testing_is_output_end"] = 33;
    map["testing_is_output_down"] = 34;
    map["testing_input_move_left"] = 35;
    map["testing_input_move_right"] = 36;
    map["testing_input_move_down"] = 37;
    map["testing_input_move_up"] = 38;
    map["comparison"] = 39;
    map["bigger_than_output_next"] = 40;
    map["bigger_than_testing_output_next"] = 41;
    map["swap_testing_output_next"] = 42;
    map["bigger_than"] = 43;
    map["equal"] = 44;
    map["no"] = 45;
    map["prog2"] = 46;
    map["prog3"] = 47;
    map["dowhile"] = 48;
    map["read_memory"] = 49;
    map["write_memory"] = 50;

    return map;
}

int getProgram(std::string string, MAP_INSTRUCTIONS map, std::vector<Node> *nodes, int &position)
{
    int program = -1;

    std::vector<int> subprograms;

    int initial_position = position;

    while (initial_position < string.length() && string[initial_position] == ' ')
        initial_position++;

    // Traverse string into tree
    while (position < string.length())
    {
        if (string[position] == '(')
        {
            // Create new program entry
            program = nodes->size();
            int pointer = map[string.substr(initial_position, position - initial_position)];

            nodes->push_back({pointer, 0, {0, 0, 0}});

            position++;

            if (string[position] == ')')
            {
                position++;
                return program;
            }
            else
            {
                int sub = getProgram(string, map, nodes, position);
                subprograms.push_back(sub);
            }
        }
        else if (string[position] == ')')
        {
            // Close and return
            position++;
            nodes->at(program).n_args = subprograms.size();

            for (int i = 0; i < subprograms.size(); i++)
            {
                nodes->at(program).args[i] = subprograms[i];
            }

            return program;
        }
        else if (string[position] == ',')
        {
            // After this, there is a new program
            position++;
            subprograms.push_back(getProgram(string, map, nodes, position));
        }
        else
        {
            position++;
        }
    }

    return program;
}

void getProgram(std::string string, MAP_INSTRUCTIONS map, std::vector<Node> *nodes)
{
    int position = 0;
    getProgram(string, map, nodes, position);
}

void copy_program(int start_index, int end_index, std::vector<int> *programs, std::vector<Node> *nodes, std::string *code, MAP_INSTRUCTIONS map)
{
    for (int i = start_index; i < end_index; ++i)
    {
        std::vector<Node> subnodes;
        getProgram(code[i], map, &subnodes);

        programs->push_back(nodes->size());

        for (int i = 0; i < subnodes.size(); i++)
        {
            nodes->push_back({subnodes[i].pointer, subnodes[i].n_args, {subnodes[i].args[0], subnodes[i].args[1], subnodes[i].args[2]}});
        }
    }
}

Programs *copy_programs_to_gpu(int n_programs, std::string *code)
{
    MAP_INSTRUCTIONS map = get_map();

    // Create array of programs in host memory
    Programs *d_sprograms;
    cudaMallocManaged((void **)&d_sprograms, sizeof(struct Problems));

    int n_threads = std::min(n_programs, 1);
    int chunk_size = n_programs / n_threads;

    std::vector<std::thread> threads;

    std::vector<int> programs;
    std::vector<Node> nodes;
    for (int i = 0; i < n_threads; ++i)
    {
        int start_index = i * chunk_size;
        int end_index = (i == n_threads - 1) ? n_programs : (i + 1) * chunk_size;

        threads.emplace_back(copy_program, start_index, end_index, &programs, &nodes, code, map);
    }

    for (auto &t : threads)
    {
        t.join();
    }

    d_sprograms->n_nodes = nodes.size();
    d_sprograms->n_programs = programs.size();

    cudaMalloc(&d_sprograms->nodes, nodes.size() * sizeof(Node));
    cudaMemcpy(d_sprograms->nodes, nodes.data(), nodes.size() * sizeof(Node), cudaMemcpyHostToDevice);

    cudaMalloc(&d_sprograms->programs, programs.size() * sizeof(int));
    cudaMemcpy(d_sprograms->programs, programs.data(), programs.size() * sizeof(int), cudaMemcpyHostToDevice);

    return d_sprograms;
}

int free_programs_from_gpu(Programs *programs)
{
    cudaFree(programs->nodes);
    cudaFree(programs->programs);
    cudaFree(programs);
    return 0;
}

#endif