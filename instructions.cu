#ifndef PROGRAM_INSTRUCTIONS_C
#define PROGRAM_INSTRUCTIONS_C

#include "types.cuh"
#include <map>
#include <thread>
#include <unordered_map>
#include <vector>

__forceinline__ __device__ int function_switch(int pointer, Run *run);

__forceinline__ __device__ int get0(Run *run, int *)
{
    return 0;
}
__forceinline__ __device__ int get1(Run *run, int *)
{
    return 1;
}
__forceinline__ __device__ int get2(Run *run, int *)
{
    return 2;
}
__forceinline__ __device__ int get3(Run *run, int *)
{
    return 3;
}
__forceinline__ __device__ int get4(Run *run, int *)
{
    return 4;
}
__forceinline__ __device__ int get5(Run *run, int *)
{
    return 5;
}
__forceinline__ __device__ int get6(Run *run, int *)
{
    return 6;
}
__forceinline__ __device__ int get7(Run *run, int *)
{
    return 7;
}
__forceinline__ __device__ int get8(Run *run, int *)
{
    return 8;
}
__forceinline__ __device__ int get9(Run *run, int *)
{
    return 9;
}

// Training
__forceinline__ __device__ int input_end(Run *run, int *)
{
    if (run->problem.n_training > 0)
        return run->training_input_x == (run->problem.training[run->training_id][0].x - 1);

    return 0;
}

__forceinline__ __device__ int input_beginning(Run *run, int *)
{
    return run->training_input_x == 0;
}

__forceinline__ __device__ int input_down_end(Run *run, int *)
{
    if (run->problem.n_training > 0)
        return run->training_input_y == (run->problem.training[run->training_id][0].y - 1);
    return 0;
}

__forceinline__ __device__ int output_end(Run *run, int *)
{
    if (run->problem.n_training > 0)
        return run->training_output_x == (run->problem.training[run->training_id][1].x - 1);

    return 0;
}

__forceinline__ __device__ int output_beginning(Run *run, int *)
{
    return run->training_output_x == 0;
}

__forceinline__ __device__ int output_down_end(Run *run, int *)
{
    if (run->problem.n_training > 0)
        return run->training_output_y == (run->problem.training[run->training_id][1].y - 1);

    return 0;
}

__forceinline__ __device__ int output_move_left(Run *run, int *)
{
    if (run->training_output_x > 0)
        run->training_output_x--;

    return 0;
}

__forceinline__ __device__ int output_move_right(Run *run, int *)
{
    if (run->problem.n_training > 0)
    {
        if (run->training_output_x < (run->problem.training[run->training_id][1].x - 1))
            run->training_output_x++;
    }
    return 0;
}

__forceinline__ __device__ int output_move_down(Run *run, int *)
{
    if (run->problem.n_training > 0)
    {
        if (run->training_output_y < (run->problem.training[run->training_id][1].y - 1))
            run->training_output_y++;
    }
    return 0;
}

__forceinline__ __device__ int output_move_up(Run *run, int *)
{
    if (run->training_output_y > 0)
        run->training_output_y--;

    return 0;
}

__forceinline__ __device__ int get_input_position_x(Run *run, int *)
{
    return run->training_input_x;
}

__forceinline__ __device__ int get_input_position_y(Run *run, int *)
{
    return run->training_input_y;
}

__forceinline__ __device__ int get_output_position_x(Run *run, int *)
{
    return run->training_output_x;
}

__forceinline__ __device__ int get_output_position_y(Run *run, int *)
{
    return run->training_output_y;
}

__forceinline__ __device__ int get_length_input_x(Run *run, int *)
{
    if (run->problem.n_training > 0)
    {
        Array m = run->problem.training[run->training_id][0];
        return m.x;
    }
    return 0;
}

__forceinline__ __device__ int get_length_input_y(Run *run, int *)
{
    if (run->problem.n_training > 0)
    {
        Array m = run->problem.training[run->training_id][0];
        return m.y;
    }
    return 0;
}

__forceinline__ __device__ int get_length_output_x(Run *run, int *)
{
    if (run->problem.n_training > 0)
    {
        Array m = run->problem.training[run->training_id][1];
        return m.x;
    }
    return 0;
}

__forceinline__ __device__ int get_length_output_y(Run *run, int *)
{
    if (run->problem.n_training > 0)
    {
        Array m = run->problem.training[run->training_id][1];
        return m.y;
    }
    return 0;
}

// Training Input
__forceinline__ __device__ int input_next(Run *run, int *)
{
    if (run->training_id < (run->problem.n_training - 1))
    {
        run->training_id++;
        run->training_input_x = 0;
        run->training_input_y = 0;
        run->training_output_x = 0;
        run->training_output_y = 0;
    }

    return 0;
}

__forceinline__ __device__ int input_previous(Run *run, int *)
{
    if (run->training_id > 0)
    {
        run->training_id--;
        run->training_input_x = 0;
        run->training_input_y = 0;
        run->training_output_x = 0;
        run->training_output_y = 0;
    }

    return 0;
}

__forceinline__ __device__ int input_move_left(Run *run, int *)
{
    if (run->training_input_x > 0)
        run->training_input_x--;

    return 0;
}

__forceinline__ __device__ int input_move_right(Run *run, int *)
{
    if (run->problem.n_training > 0)
    {
        if (run->training_input_x < (run->problem.training[run->training_id][0].x - 1))
            run->training_input_x++;
    }
    return 0;
}

__forceinline__ __device__ int input_move_down(Run *run, int *)
{
    if (run->problem.n_training > 0)
    {
        if (run->training_input_y < (run->problem.training[run->training_id][0].y - 1))
            run->training_input_y++;
    }
    return 0;
}

__forceinline__ __device__ int input_move_up(Run *run, int *)
{
    if (run->training_input_y > 0)
        run->training_input_y--;

    return 0;
}

__forceinline__ __device__ int reset_input_position(Run *run, int *)
{
    run->training_input_x = 0;
    return 0;
}

__forceinline__ __device__ int reset_input_down_position(Run *run, int *)
{
    run->training_input_y = 0;
    return 0;
}

__forceinline__ __device__ int input_max(Run *run, int *)
{
    if (run->problem.n_training > 0)
    {
        int *arr = run->problem.training[run->training_id][0].array[run->training_input_y];

        int max = arr[0];
        for (int i = 1; i < run->problem.training[run->training_id][0].x; i++)
        {
            if (arr[i] > max)
            {
                max = arr[i];
            }
        }
        return max;
    }

    return 0;
}

__forceinline__ __device__ int input_min(Run *run, int *)
{
    if (run->problem.n_training > 0)
    {
        int *arr = run->problem.training[run->training_id][0].array[run->training_input_y];

        int min = arr[0];
        for (int i = 1; i < run->problem.training[run->training_id][0].x; i++)
        {
            if (arr[i] < min)
            {
                min = arr[i];
            }
        }
        return min;
    }

    return 0;
}

__forceinline__ __device__ int input_read(Run *run, int *)
{
    if (run->problem.n_training > 0)
        return run->problem.training[run->training_id][0].array[run->training_input_y][run->training_input_x];

    return 0;
}

__forceinline__ __device__ int output_read(Run *run, int *)
{
    if (run->problem.n_training > 0)
        return run->problem.training[run->training_id][1].array[run->training_output_y][run->training_output_x];

    return 0;
}

__forceinline__ __device__ int reset_output_position(Run *run, int *)
{
    run->training_output_x = 0;
    return 0;
}

__forceinline__ __device__ int reset_output_down_position(Run *run, int *)
{
    run->training_output_y = 0;
    return 0;
}

// Testing
__forceinline__ __device__ int get_testing_length_input_x(Run *run, int *)
{
    return run->problem.input.x;
}

__forceinline__ __device__ int get_testing_length_input_y(Run *run, int *)
{
    return run->problem.input.y;
}

__forceinline__ __device__ int get_testing_length_output_x(Run *run, int *)
{
    return run->problem.output.x;
}

__forceinline__ __device__ int get_testing_length_output_y(Run *run, int *)
{
    return run->problem.output.y;
}

__forceinline__ __device__ int get_testing_input_position_y(Run *run, int *)
{
    return run->input_y;
}

__forceinline__ __device__ int get_testing_input_position_x(Run *run, int *)
{
    return run->input_x;
}

__forceinline__ __device__ int get_testing_output_position_y(Run *run, int *)
{
    return run->output_y;
}

__forceinline__ __device__ int get_testing_output_position_x(Run *run, int *)
{
    return run->output_x;
}

__forceinline__ __device__ int testing_input_max(Run *run, int *)
{
    int *arr = run->problem.input.array[run->input_y];
    int max = arr[0];
    for (int i = 1; i < run->problem.input.x; i++)
    {
        if (arr[i] > max)
        {
            max = arr[i];
        }
    }
    return max;
}

__forceinline__ __device__ int testing_input_min(Run *run, int *)
{
    int *arr = run->problem.input.array[run->input_y];
    int min = arr[0];
    for (int i = 1; i < run->problem.input.x; i++)
    {
        if (arr[i] < min)
        {
            min = arr[i];
        }
    }
    return min;
}

__forceinline__ __device__ int testing_input_read(Run *run, int *)
{
    return run->problem.input.array[run->input_y][run->input_x];
}

__forceinline__ __device__ int testing_output_read_previous(Run *run, int *)
{
    if (run->output_x > 0)
        return run->output[run->output_y][run->output_x - 1];

    return -1;
}

__forceinline__ __device__ int testing_output_read(Run *run, int *)
{
    return run->output[run->output_y][run->output_x];

    return 0;
}

__forceinline__ __device__ int testing_reset_input_position(Run *run, int *)
{
    run->input_x = 0;
    return 0;
}

__forceinline__ __device__ int testing_reset_input_down_position(Run *run, int *)
{
    run->input_y = 0;
    return 0;
}

__forceinline__ __device__ int testing_output_write_previous(Run *run, int *p)
{
    if (run->output_x > 0)
    {
        int value = run->pfuncs[(run->nodes[p[0]].pointer)](run, run->nodes[p[0]].args);

        run->output[run->output_y][run->output_x - 1] = value;
    }

    return 0;
}

__forceinline__ __device__ int testing_output_write(Run *run, int *p)
{
    int value = run->pfuncs[(run->nodes[p[0]].pointer)](run, run->nodes[p[0]].args);

    run->output[run->output_y][run->output_x] = value;

    return 0;
}

__forceinline__ __device__ int testing_reset_output_position(Run *run, int *)
{
    run->output_x = 0;
    return 0;
}

__forceinline__ __device__ int testing_reset_output_down_position(Run *run, int *)
{
    run->output_y = 0;
    return 0;
}

__forceinline__ __device__ int testing_output_move_left(Run *run, int *)
{
    if (run->output_x > 0)
        run->output_x--;
    else
        run->status = -1;

    return 0;
}

__forceinline__ __device__ int testing_output_move_right(Run *run, int *)
{
    if (run->output_x < (run->problem.output.x - 1))
        run->output_x++;
    else
        run->status = -1;

    return 0;
}

__forceinline__ __device__ int testing_output_move_down(Run *run, int *)
{
    if (run->output_y < (run->problem.output.y - 1))
        run->output_y++;
    else
        run->status = -1;
    return 0;
}

__forceinline__ __device__ int testing_output_move_up(Run *run, int *)
{
    if (run->output_y > 0)
        run->output_y--;
    else
        run->status = -1;

    return 0;
}

__forceinline__ __device__ int testing_is_output_end(Run *run, int *)
{
    return run->output_x == (run->problem.output.x - 1);
}

__forceinline__ __device__ int testing_is_output_down(Run *run, int *)
{
    return run->output_y == (run->problem.output.y - 1);
}

__forceinline__ __device__ int testing_input_move_left(Run *run, int *)
{
    if (run->input_x > 0)
        run->input_x--;
    else
        run->status = -1;

    return 0;
}

__forceinline__ __device__ int testing_input_move_right(Run *run, int *)
{
    if (run->input_x < (run->problem.input.x - 1))
        run->input_x++;
    else
        run->status = -1;

    return 0;
}

__forceinline__ __device__ int testing_input_move_down(Run *run, int *)
{
    if (run->input_y < (run->problem.input.y - 1))
        run->input_y++;
    else
        run->status = -1;

    return 0;
}

__forceinline__ __device__ int testing_input_move_up(Run *run, int *)
{
    if (run->input_y > 0)
        run->input_y--;
    else
        run->status = -1;

    return 0;
}

__forceinline__ __device__ int comparison(Run *run, int *p)
{
    if (run->pfuncs[(run->nodes[p[0]].pointer)](run, run->nodes[p[0]].args) == 1)
    {
        run->pfuncs[(run->nodes[p[1]].pointer)](run, run->nodes[p[1]].args);
    }
    else
    {
        run->pfuncs[(run->nodes[p[2]].pointer)](run, run->nodes[p[2]].args);
    }

    return 0;
}

__forceinline__ __device__ int bigger_than_output_next(Run *run, int *p)
{
    if (run->problem.n_training > 0)
        if (run->training_output_x < (run->problem.training[run->training_id][1].x - 1))
        {
            return run->problem.training[run->training_id][1].array[run->training_output_y][run->training_output_x] >
                   run->problem.training[run->training_id][1].array[run->training_output_y][run->training_output_x + 1];
        }

    return 0;
}

__forceinline__ __device__ int bigger_than_testing_output_next(Run *run, int *)
{
    if (run->output_x < (run->problem.output.x - 1))
    {
        return run->output[run->output_y][run->output_x] >
               run->output[run->output_y][run->output_x + 1];
    }

    return 0;
}

__forceinline__ __device__ int swap_testing_output_next(Run *run, int *)
{
    if (run->output_x < (run->problem.output.x - 1))
    {
        int tmp = run->output[run->output_y][run->output_x];
        run->output[run->output_y][run->output_x] = run->output[run->output_y][run->output_x + 1];
        run->output[run->output_y][run->output_x + 1] = tmp;
    }

    return 0;
}

__forceinline__ __device__ int bigger_than(Run *run, int *p)
{
    int output1 = run->pfuncs[(run->nodes[p[0]].pointer)](run, run->nodes[p[0]].args);

    int output2 = run->pfuncs[(run->nodes[p[1]].pointer)](run, run->nodes[p[1]].args);
    return output1 > output2;
}

__forceinline__ __device__ int equal(Run *run, int *p)
{
    int output1 = run->pfuncs[(run->nodes[p[0]].pointer)](run, run->nodes[p[0]].args);

    int output2 = run->pfuncs[(run->nodes[p[1]].pointer)](run, run->nodes[p[1]].args);

    return output1 == output2;
}

__forceinline__ __device__ int no(Run *run, int *p)
{
    int output1 = run->pfuncs[(run->nodes[p[0]].pointer)](run, run->nodes[p[0]].args);

    return !output1;
}

__forceinline__ __device__ int prog2(Run *run, int *p)
{
    run->pfuncs[(run->nodes[p[0]].pointer)](run, run->nodes[p[0]].args);

    run->pfuncs[(run->nodes[p[1]].pointer)](run, run->nodes[p[1]].args);
    return 0;
}

__forceinline__ __device__ int prog3(Run *run, int *p)
{
    run->pfuncs[(run->nodes[p[0]].pointer)](run, run->nodes[p[0]].args);

    run->pfuncs[(run->nodes[p[1]].pointer)](run, run->nodes[p[1]].args);

    run->pfuncs[(run->nodes[p[2]].pointer)](run, run->nodes[p[2]].args);

    return 0;
}

__forceinline__ __device__ int loop(Run *run, int *p)
{
    run->inner_loop++;

    int v = function_switch(p[0], run); // run->pfuncs[(run->nodes[p[0]].pointer)](run, run->nodes[p[0]].args);

    if (run->inner_loop < 5 && v > 0 && v <= 30)
    {
        // int pointer = (run->nodes[p[1]].pointer);
        // int * args = run->nodes[p[1]].args;

        for (int i = 0; i < v; i++)
        {
            function_switch(p[1], run);
            // run->pfuncs[pointer](run, args);

            if (run->status != 0)
                break;
        }
    }
    else
    {
        run->status = -2;
    }

    run->inner_loop--;

    return v;
}

__forceinline__ __device__ int dowhile(Run *run, int *p)
{
    int c = 0;

    run->inner_loop++;

    if (run->inner_loop < 3)
    {
        while (!run->pfuncs[(run->nodes[p[0]].pointer)](run, run->nodes[p[0]].args) && c < 10)
        {
            c++;
            run->pfuncs[(run->nodes[p[1]].pointer)](run, run->nodes[p[1]].args);
        }
    }
    else
    {
        run->status = -2;
    }

    run->inner_loop--;

    return 0;
}

__forceinline__ __device__ int read_memory(Run *run, int *)
{
    return run->memory;
}

__forceinline__ __device__ int write_memory(Run *run, int *p)
{
    int value = run->pfuncs[(run->nodes[p[0]].pointer)](run, run->nodes[p[0]].args);

    run->memory = value;

    return 0;
}

struct SNode
{
    int node_pointer;
    int case_operation;
};

__forceinline__ __device__ int function_switch(int pointer, Run *run)
{
    SNode stack[1000];
    int s_pointer = 1;

    // Root of the program tree
    stack[0] = {pointer, run->nodes[pointer].pointer};

    int reg = 0;
    int reg1 = 0;

    // Till no more nodes are available
    while (s_pointer > 0 && run->status == 0)
    {
        SNode node = stack[--s_pointer];

        switch (node.case_operation)
        {
        case 0:
            reg = get0(NULL, NULL);
            break;
        case 1:
            reg = get1(NULL, NULL);
            break;
        case 2:
            reg = get2(NULL, NULL);
            break;
        case 3:
            reg = get3(NULL, NULL);
            break;
        case 4:
            reg = get4(NULL, NULL);
            break;
        case 5:
            reg = get5(NULL, NULL);
            break;
        case 6:
            reg = get6(NULL, NULL);
            break;
        case 7:
            reg = get7(NULL, NULL);
            break;
        case 8:
            reg = get8(NULL, NULL);
            break;
        case 9:
            reg = get9(NULL, NULL);
            break;
        case 10:
            reg = input_end(run, NULL);
            break;
        case 11:
            reg = input_beginning(run, NULL);
            break;
        case 12:
            reg = input_down_end(run, NULL);
            break;
        case 13:
            reg = output_end(run, NULL);
            break;
        case 14:
            reg = output_beginning(run, NULL);
            break;
        case 15:
            reg = output_down_end(run, NULL);
            break;
        case 16:
            reg = output_move_left(run, NULL);
            break;
        case 17:
            reg = output_move_right(run, NULL);
            break;
        case 18:
            reg = output_move_down(run, NULL);
            break;
        case 19:
            reg = output_move_up(run, NULL);
            break;
        case 20:
            reg = get_input_position_x(run, NULL);
            break;
        case 21:
            reg = get_input_position_y(run, NULL);
            break;
        case 22:
            reg = get_output_position_x(run, NULL);
            break;
        case 23:
            reg = get_output_position_y(run, NULL);
            break;
        case 24:
            reg = get_length_input_x(run, NULL);
            break;
        case 25:
            reg = get_length_input_y(run, NULL);
            break;
        case 26:
            reg = get_length_output_x(run, NULL);
            break;
        case 27:
            reg = get_length_output_y(run, NULL);
            break;
        case 28:
            reg = input_next(run, NULL);
            break;
        case 29:
            reg = input_previous(run, NULL);
            break;
        case 30:
            reg = input_move_left(run, NULL);
            break;
        case 31:
            reg = input_move_right(run, NULL);
            break;
        case 32:
            reg = input_move_down(run, NULL);
            break;
        case 33:
            reg = input_move_up(run, NULL);
            break;
        case 34:
            reg = reset_input_position(run, NULL);
            break;
        case 35:
            reg = reset_input_down_position(run, NULL);
            break;
        case 36:
            reg = input_max(run, NULL);
            break;
        case 37:
            reg = input_min(run, NULL);
            break;
        case 38:
            reg = input_read(run, NULL);
            break;
        case 39:
            reg = output_read(run, NULL);
            break;
        case 40:
            reg = reset_output_position(run, NULL);
            break;
        case 41:
            reg = reset_output_down_position(run, NULL);
            break;
        case 42:
            reg = get_testing_length_input_x(run, NULL);
            break;
        case 43:
            reg = get_testing_length_input_y(run, NULL);
            break;
        case 44:
            reg = get_testing_length_output_x(run, NULL);
            break;
        case 45:
            reg = get_testing_length_output_y(run, NULL);
            break;
        case 46:
            reg = get_testing_input_position_y(run, NULL);
            break;
        case 47:
            reg = get_testing_input_position_x(run, NULL);
            break;
        case 48:
            reg = get_testing_output_position_y(run, NULL);
            break;
        case 49:
            reg = get_testing_output_position_x(run, NULL);
            break;
        case 50:
            reg = testing_input_max(run, NULL);
            break;
        case 51:
            reg = testing_input_min(run, NULL);
            break;
        case 52:
            reg = testing_input_read(run, NULL);
            break;
        case 53:
            reg = testing_output_read_previous(run, NULL);
            break;
        case 54:
            reg = testing_output_read(run, NULL);
            break;
        case 55:
            reg = testing_reset_input_position(run, NULL);
            break;
        case 56:
            reg = testing_reset_input_down_position(run, NULL);
            break;
        // Write operations missing
        case 57: // testing_output_write_previous
            if (run->output_x > 0)
            {

                Node pnode = run->nodes[node.node_pointer];
                stack[s_pointer++] = {0, 1001};

                // Add read value node
                stack[s_pointer++] = {pnode.args[0], run->nodes[pnode.args[0]].pointer};
            }

            break;
        case 58: // testing_output_write
        {
            Node pnode = run->nodes[node.node_pointer];
            stack[s_pointer++] = {0, 1002};

            // Add read value node
            stack[s_pointer++] = {pnode.args[0], run->nodes[pnode.args[0]].pointer};
        }
        break;
        case 59:
            reg = testing_reset_output_position(run, NULL);
            break;
        case 60:
            reg = testing_reset_output_down_position(run, NULL);
            break;
        case 61:
            reg = testing_output_move_left(run, NULL);
            break;
        case 62:
            reg = testing_output_move_right(run, NULL);
            break;
        case 63:
            reg = testing_output_move_down(run, NULL);
            break;
        case 64:
            reg = testing_output_move_up(run, NULL);
            break;
        case 65:
            reg = testing_is_output_end(run, NULL);
            break;
        case 66:
            reg = testing_is_output_down(run, NULL);
            break;
        case 67:
            reg = testing_input_move_left(run, NULL);
            break;
        case 68:
            reg = testing_input_move_right(run, NULL);
            break;
        case 69:
            reg = testing_input_move_down(run, NULL);
            break;
        case 70:
            reg = testing_input_move_up(run, NULL);
            break;
        case 71: // comparison
        {
            Node pnode = run->nodes[node.node_pointer];
            stack[s_pointer++] = {node.node_pointer, 1004};

            // Add read value node
            stack[s_pointer++] = {pnode.args[0], run->nodes[pnode.args[0]].pointer};
        }
        break;
        case 72:
            reg = bigger_than_output_next(run, NULL);
            break;

        case 73:
            reg = bigger_than_testing_output_next(run, NULL);
            break;
        case 74:
            reg = swap_testing_output_next(run, NULL);
            break;
        case 75: // bigger_than
        {
            Node pnode = run->nodes[node.node_pointer];
            stack[s_pointer++] = {node.node_pointer, 1008};

            // Add read value node
            stack[s_pointer++] = {pnode.args[0], run->nodes[pnode.args[0]].pointer};
        }
        break;
        case 76: // equal
        {
            Node pnode = run->nodes[node.node_pointer];
            stack[s_pointer++] = {node.node_pointer, 1006};

            // Add read value node
            stack[s_pointer++] = {pnode.args[0], run->nodes[pnode.args[0]].pointer};
        }
        break;
        case 77: // no
        {
            Node pnode = run->nodes[node.node_pointer];
            stack[s_pointer++] = {node.node_pointer, 1005};

            // Add read value node
            stack[s_pointer++] = {pnode.args[0], run->nodes[pnode.args[0]].pointer};
        }
        break;
        case 78: // prog2
            Node pnode = run->nodes[node.node_pointer];
            stack[s_pointer++] = {pnode.args[1], run->nodes[pnode.args[1]].pointer};
            stack[s_pointer++] = {pnode.args[0], run->nodes[pnode.args[0]].pointer};
            break;
        case 79: // prog3
        {
            Node pnode = run->nodes[node.node_pointer];
            stack[s_pointer++] = {pnode.args[2], run->nodes[pnode.args[2]].pointer};
            stack[s_pointer++] = {pnode.args[1], run->nodes[pnode.args[1]].pointer};
            stack[s_pointer++] = {pnode.args[0], run->nodes[pnode.args[0]].pointer};
        }
        break;
        case 81:
            reg = read_memory(run, NULL);
            break;
        case 82: // write memory
        {
            Node pnode = run->nodes[node.node_pointer];
            stack[s_pointer++] = {0, 1003};

            // Add read value node
            // Node p0node = run->nodes[node.node_pointer];
            stack[s_pointer++] = {pnode.args[0], run->nodes[pnode.args[0]].pointer};
        }
        break;
        case 83: // loop
        {
            run->inner_loop++;

            // Add operation node n-times
            Node pnode = run->nodes[node.node_pointer];
            stack[s_pointer++] = {pnode.args[1], 1000};

            // Add read value node
            stack[s_pointer++] = {pnode.args[0], run->nodes[pnode.args[0]].pointer};
        }
        break;
        case 1000:
            if (run->inner_loop < 5 && reg > 0 && reg <= 30)
            {
                // Read register
                stack[s_pointer++] = {0, 10001};
                for (int i = 0; i < reg; i++)
                {
                    // Loop operation
                    stack[s_pointer++] = {node.node_pointer, run->nodes[node.node_pointer].pointer};
                }
            }
            else
            {
                run->status = -2;
            }

            break;
        case 10001:
            run->inner_loop--;
            break;
        case 1001: // testing_output_write_previous
            run->output[run->output_y][run->output_x - 1] = reg;
            break;
        case 1002: // testing_output_write
            run->output[run->output_y][run->output_x] = reg;
            break;
        case 1003: // write memory
            run->memory = reg;
            break;
        case 1004: // comparison
        {
            Node pnode = run->nodes[node.node_pointer];
            if (reg)
            {
                stack[s_pointer++] = {pnode.args[0], run->nodes[pnode.args[0]].pointer};
            }
            else
            {
                stack[s_pointer++] = {pnode.args[1], run->nodes[pnode.args[1]].pointer};
            }
        }
        break;
        case 1005: // no
            reg = !reg;
            break;
        case 1006: // equal
        {
            // copy register
            reg1 = reg;
            // final check
            stack[s_pointer++] = {node.node_pointer, 1007};
            // obtain second value
            Node pnode = run->nodes[node.node_pointer];
            stack[s_pointer++] = {pnode.args[1], run->nodes[pnode.args[1]].pointer};
        }
        break;
        case 1007: // equal
            reg = (reg == reg1);
            break;
        case 1008: // bigger_than
        {
            // copy register
            reg1 = reg;
            // final check
            stack[s_pointer++] = {node.node_pointer, 1009};
            // obtain second value
            Node pnode = run->nodes[node.node_pointer];
            stack[s_pointer++] = {pnode.args[1], run->nodes[pnode.args[1]].pointer};
        }
        break;
        case 1009:
            reg = (reg1 > reg);
            break;
        default:
            run->status = -1;
            break;
        }
    }

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
    pfuncs[10] = input_end;
    pfuncs[11] = input_beginning;
    pfuncs[12] = input_down_end;
    pfuncs[13] = output_end;
    pfuncs[14] = output_beginning;
    pfuncs[15] = output_down_end;
    pfuncs[16] = output_move_left;
    pfuncs[17] = output_move_right;
    pfuncs[18] = output_move_down;
    pfuncs[19] = output_move_up;
    pfuncs[20] = get_input_position_x;
    pfuncs[21] = get_input_position_y;
    pfuncs[22] = get_output_position_x;
    pfuncs[23] = get_output_position_y;
    pfuncs[24] = get_length_input_x;
    pfuncs[25] = get_length_input_y;
    pfuncs[26] = get_length_output_x;
    pfuncs[27] = get_length_output_y;
    pfuncs[28] = input_next;
    pfuncs[29] = input_previous;
    pfuncs[30] = input_move_left;
    pfuncs[31] = input_move_right;
    pfuncs[32] = input_move_down;
    pfuncs[33] = input_move_up;
    pfuncs[34] = reset_input_position;
    pfuncs[35] = reset_input_down_position;
    pfuncs[36] = input_max;
    pfuncs[37] = input_min;
    pfuncs[38] = input_read;
    pfuncs[39] = output_read;
    pfuncs[40] = reset_output_position;
    pfuncs[41] = reset_output_down_position;
    pfuncs[42] = get_testing_length_input_x;
    pfuncs[43] = get_testing_length_input_y;
    pfuncs[44] = get_testing_length_output_x;
    pfuncs[45] = get_testing_length_output_y;
    pfuncs[46] = get_testing_input_position_y;
    pfuncs[47] = get_testing_input_position_x;
    pfuncs[48] = get_testing_output_position_y;
    pfuncs[49] = get_testing_output_position_x;
    pfuncs[50] = testing_input_max;
    pfuncs[51] = testing_input_min;
    pfuncs[52] = testing_input_read;
    pfuncs[53] = testing_output_read_previous;
    pfuncs[54] = testing_output_read;
    pfuncs[55] = testing_reset_input_position;
    pfuncs[56] = testing_reset_input_down_position;
    pfuncs[57] = testing_output_write_previous;
    pfuncs[58] = testing_output_write;
    pfuncs[59] = testing_reset_output_position;
    pfuncs[60] = testing_reset_output_down_position;
    pfuncs[61] = testing_output_move_left;
    pfuncs[62] = testing_output_move_right;
    pfuncs[63] = testing_output_move_down;
    pfuncs[64] = testing_output_move_up;
    pfuncs[65] = testing_is_output_end;
    pfuncs[66] = testing_is_output_down;
    pfuncs[67] = testing_input_move_left;
    pfuncs[68] = testing_input_move_right;
    pfuncs[69] = testing_input_move_down;
    pfuncs[70] = testing_input_move_up;
    pfuncs[71] = comparison;
    pfuncs[72] = bigger_than_output_next;
    pfuncs[73] = bigger_than_testing_output_next;
    pfuncs[74] = swap_testing_output_next;
    pfuncs[75] = bigger_than;
    pfuncs[76] = equal;
    pfuncs[77] = no;
    pfuncs[78] = prog2;
    pfuncs[79] = prog3;
    pfuncs[80] = dowhile;
    pfuncs[81] = read_memory;
    pfuncs[82] = write_memory;
    pfuncs[83] = loop;
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
    map["input_end"] = 10;
    map["input_beginning"] = 11;
    map["input_down_end"] = 12;
    map["output_end"] = 13;
    map["output_beginning"] = 14;
    map["output_down_end"] = 15;
    map["output_move_left"] = 16;
    map["output_move_right"] = 17;
    map["output_move_down"] = 18;
    map["output_move_up"] = 19;
    map["get_input_position_x"] = 20;
    map["get_input_position_y"] = 21;
    map["get_output_position_x"] = 22;
    map["get_output_position_y"] = 23;
    map["get_length_input_x"] = 24;
    map["get_length_input_y"] = 25;
    map["get_length_output_x"] = 26;
    map["get_length_output_y"] = 27;
    map["input_next"] = 28;
    map["input_previous"] = 29;
    map["input_move_left"] = 30;
    map["input_move_right"] = 31;
    map["input_move_down"] = 32;
    map["input_move_up"] = 33;
    map["reset_input_position"] = 34;
    map["reset_input_down_position"] = 35;
    map["input_max"] = 36;
    map["input_min"] = 37;
    map["input_read"] = 38;
    map["output_read"] = 39;
    map["reset_output_position"] = 40;
    map["reset_output_down_position"] = 41;
    map["get_testing_length_input_x"] = 42;
    map["get_testing_length_input_y"] = 43;
    map["get_testing_length_output_x"] = 44;
    map["get_testing_length_output_y"] = 45;
    map["get_testing_input_position_y"] = 46;
    map["get_testing_input_position_x"] = 47;
    map["get_testing_output_position_y"] = 48;
    map["get_testing_output_position_x"] = 49;
    map["testing_input_max"] = 50;
    map["testing_input_min"] = 51;
    map["testing_input_read"] = 52;
    map["testing_output_read_previous"] = 53;
    map["testing_output_read"] = 54;
    map["testing_reset_input_position"] = 55;
    map["testing_reset_input_down_position"] = 56;
    map["testing_output_write_previous"] = 57;
    map["testing_output_write"] = 58;
    map["testing_reset_output_position"] = 59;
    map["testing_reset_output_down_position"] = 60;
    map["testing_output_move_left"] = 61;
    map["testing_output_move_right"] = 62;
    map["testing_output_move_down"] = 63;
    map["testing_output_move_up"] = 64;
    map["testing_is_output_end"] = 65;
    map["testing_is_output_down"] = 66;
    map["testing_input_move_left"] = 67;
    map["testing_input_move_right"] = 68;
    map["testing_input_move_down"] = 69;
    map["testing_input_move_up"] = 70;
    map["comparison"] = 71;
    map["bigger_than_output_next"] = 72;
    map["bigger_than_testing_output_next"] = 73;
    map["swap_testing_output_next"] = 74;
    map["bigger_than"] = 75;
    map["bigger_thanR"] = 75;
    map["bigger_thanW"] = 75;
    map["equal"] = 76;
    map["equalR"] = 76;
    map["equalW"] = 76;
    map["no"] = 77;
    map["prog2"] = 78;
    map["prog3"] = 79;
    map["dowhile"] = 80;
    map["read_memory"] = 81;
    map["write_memory"] = 82;
    map["loop"] = 83;

    return map;
}

int getProgram(std::string string, MAP_INSTRUCTIONS map, std::map<int, Node> *nodes, int &position)
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

            nodes->insert({program, {pointer, 0, {0, 0, 0}}});

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

void getProgram(std::string string, MAP_INSTRUCTIONS map, std::map<int, Node> *nodes)
{
    int position = 0;
    getProgram(string, map, nodes, position);
}

void copy_program(int start_index, int end_index, std::vector<int> *programs, std::vector<Node> *nodes, std::string *code, MAP_INSTRUCTIONS map)
{
    for (int i = start_index; i < end_index; ++i)
    {
        std::map<int, Node> subnodes;
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
    cudaMallocManaged(&d_sprograms, sizeof(Programs));

    int n_threads = std::min(n_programs, 20);
    int chunk_size = n_programs / n_threads;

    std::vector<std::thread> threads;

    std::vector<std::vector<int>> programs(n_threads);
    std::vector<std::vector<Node>> nodes(n_threads);
    for (int i = 0; i < n_threads; ++i)
    {
        int start_index = i * chunk_size;
        int end_index = (i == n_threads - 1) ? n_programs : (i + 1) * chunk_size;

        threads.emplace_back(copy_program, start_index, end_index, &programs[i], &nodes[i], code, map);
    }

    for (auto &t : threads)
    {
        t.join();
    }

    size_t total_nodes = 0;
    size_t total_programs = 0;
    for (int i = 0; i < n_threads; i++)
    {
        total_nodes += nodes[i].size();
        total_programs += programs[i].size();
    }

    d_sprograms->n_nodes = total_nodes;
    d_sprograms->n_programs = total_programs;

    cudaMallocManaged(&d_sprograms->nodes, d_sprograms->n_nodes * sizeof(Node));

    cudaMallocManaged(&d_sprograms->programs, d_sprograms->n_programs * sizeof(int));

    size_t offset_nodes = 0;
    size_t offset_programs = 0;
    for (int i = 0; i < n_threads; i++)
    {
        std::copy(nodes[i].begin(), nodes[i].end(), d_sprograms->nodes + offset_nodes);

        for (int j = 0; j < programs[i].size(); j++)
        {
            programs[i][j] += offset_nodes;
        }
        std::copy(programs[i].begin(), programs[i].end(), d_sprograms->programs + offset_programs);

        offset_nodes += nodes[i].size();
        offset_programs += programs[i].size();
    }
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