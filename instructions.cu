#ifndef PROGRAM_INSTRUCTIONS_C
#define PROGRAM_INSTRUCTIONS_C

#include "types.cuh"
#include <map>
#include <thread>
#include <unordered_map>
#include <vector>

#ifndef SETUP_BUILDING_CPU
#define FUNCTION_DEFINITION __forceinline__ __device__ int
#else
#define FUNCTION_DEFINITION inline int
#endif

FUNCTION_DEFINITION function_switch(int pointer, Run *run);

FUNCTION_DEFINITION get0(Run *run)
{
    return 0;
}
FUNCTION_DEFINITION get1(Run *run)
{
    return 1;
}
FUNCTION_DEFINITION get2(Run *run)
{
    return 2;
}
FUNCTION_DEFINITION get3(Run *run)
{
    return 3;
}
FUNCTION_DEFINITION get4(Run *run)
{
    return 4;
}
FUNCTION_DEFINITION get5(Run *run)
{
    return 5;
}
FUNCTION_DEFINITION get6(Run *run)
{
    return 6;
}
FUNCTION_DEFINITION get7(Run *run)
{
    return 7;
}
FUNCTION_DEFINITION get8(Run *run)
{
    return 8;
}
FUNCTION_DEFINITION get9(Run *run)
{
    return 9;
}

// Training
FUNCTION_DEFINITION input_end(Run *run)
{
    if (run->problem.n_training > 0)
        return run->training_input_x == (run->problem.training[run->training_id][0].x - 1);

    return 0;
}

FUNCTION_DEFINITION input_beginning(Run *run)
{
    return run->training_input_x == 0;
}

FUNCTION_DEFINITION input_down_end(Run *run)
{
    if (run->problem.n_training > 0)
        return run->training_input_y == (run->problem.training[run->training_id][0].y - 1);
    return 0;
}

FUNCTION_DEFINITION output_end(Run *run)
{
    if (run->problem.n_training > 0)
        return run->training_output_x == (run->problem.training[run->training_id][1].x - 1);

    return 0;
}

FUNCTION_DEFINITION output_beginning(Run *run)
{
    return run->training_output_x == 0;
}

FUNCTION_DEFINITION output_down_end(Run *run)
{
    if (run->problem.n_training > 0)
        return run->training_output_y == (run->problem.training[run->training_id][1].y - 1);

    return 0;
}

FUNCTION_DEFINITION output_move_left(Run *run)
{
    if (run->training_output_x > 0)
        run->training_output_x--;
    else
        run->status = -1;

    return 0;
}

FUNCTION_DEFINITION output_move_right(Run *run)
{
    if (run->problem.n_training > 0)
    {
        if (run->training_output_x < (run->problem.training[run->training_id][1].x - 1))
            run->training_output_x++;
        else
            run->status = -1;
    }
    return 0;
}

FUNCTION_DEFINITION output_move_down(Run *run)
{
    if (run->problem.n_training > 0)
    {
        if (run->training_output_y < (run->problem.training[run->training_id][1].y - 1))
            run->training_output_y++;
        else
            run->status = -1;
    }
    return 0;
}

FUNCTION_DEFINITION output_move_up(Run *run)
{
    if (run->training_output_y > 0)
        run->training_output_y--;
    else
        run->status = -1;

    return 0;
}

FUNCTION_DEFINITION get_input_position_x(Run *run)
{
    return run->training_input_x;
}

FUNCTION_DEFINITION get_input_position_y(Run *run)
{
    return run->training_input_y;
}

FUNCTION_DEFINITION get_output_position_x(Run *run)
{
    return run->training_output_x;
}

FUNCTION_DEFINITION get_output_position_y(Run *run)
{
    return run->training_output_y;
}

FUNCTION_DEFINITION get_length_input_x(Run *run)
{
    if (run->problem.n_training > 0)
    {
        Array m = run->problem.training[run->training_id][0];
        return m.x;
    }
    return 0;
}

FUNCTION_DEFINITION get_length_input_y(Run *run)
{
    if (run->problem.n_training > 0)
    {
        Array m = run->problem.training[run->training_id][0];
        return m.y;
    }
    return 0;
}

FUNCTION_DEFINITION get_length_output_x(Run *run)
{
    if (run->problem.n_training > 0)
    {
        Array m = run->problem.training[run->training_id][1];
        return m.x;
    }
    return 0;
}

FUNCTION_DEFINITION get_length_output_y(Run *run)
{
    if (run->problem.n_training > 0)
    {
        Array m = run->problem.training[run->training_id][1];
        return m.y;
    }
    return 0;
}

// Training Input
FUNCTION_DEFINITION input_next(Run *run)
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

FUNCTION_DEFINITION input_previous(Run *run)
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

FUNCTION_DEFINITION input_move_left(Run *run)
{
    if (run->training_input_x > 0)
        run->training_input_x--;
    else
        run->status = -1;

    return 0;
}

FUNCTION_DEFINITION input_move_right(Run *run)
{
    if (run->problem.n_training > 0)
    {
        if (run->training_input_x < (run->problem.training[run->training_id][0].x - 1))
            run->training_input_x++;
        else
            run->status = -1;
    }
    return 0;
}

FUNCTION_DEFINITION input_move_down(Run *run)
{
    if (run->problem.n_training > 0)
    {
        if (run->training_input_y < (run->problem.training[run->training_id][0].y - 1))
            run->training_input_y++;
        else
            run->status = -1;
    }
    return 0;
}

FUNCTION_DEFINITION input_move_up(Run *run)
{
    if (run->training_input_y > 0)
        run->training_input_y--;
    else
        run->status = -1;

    return 0;
}

FUNCTION_DEFINITION reset_input_position(Run *run)
{
    run->training_input_x = 0;
    return 0;
}

FUNCTION_DEFINITION reset_input_down_position(Run *run)
{
    run->training_input_y = 0;
    return 0;
}

FUNCTION_DEFINITION input_max(Run *run)
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

FUNCTION_DEFINITION input_min(Run *run)
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

FUNCTION_DEFINITION input_read(Run *run)
{
    if (run->problem.n_training > 0)
        return run->problem.training[run->training_id][0].array[run->training_input_y][run->training_input_x];

    return 0;
}

FUNCTION_DEFINITION output_read(Run *run)
{
    if (run->problem.n_training > 0)
        return run->problem.training[run->training_id][1].array[run->training_output_y][run->training_output_x];

    return 0;
}

FUNCTION_DEFINITION reset_output_position(Run *run)
{
    run->training_output_x = 0;
    return 0;
}

FUNCTION_DEFINITION reset_output_down_position(Run *run)
{
    run->training_output_y = 0;
    return 0;
}

// Testing
FUNCTION_DEFINITION get_testing_length_input_x(Run *run)
{
    return run->problem.input.x;
}

FUNCTION_DEFINITION get_testing_length_input_y(Run *run)
{
    return run->problem.input.y;
}

FUNCTION_DEFINITION get_testing_length_output_x(Run *run)
{
    return run->problem.output.x;
}

FUNCTION_DEFINITION get_testing_length_output_y(Run *run)
{
    return run->problem.output.y;
}

FUNCTION_DEFINITION get_testing_input_position_y(Run *run)
{
    return run->input_y;
}

FUNCTION_DEFINITION get_testing_input_position_x(Run *run)
{
    return run->input_x;
}

FUNCTION_DEFINITION get_testing_output_position_y(Run *run)
{
    return run->output_y;
}

FUNCTION_DEFINITION get_testing_output_position_x(Run *run)
{
    return run->output_x;
}

FUNCTION_DEFINITION testing_input_max(Run *run)
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

FUNCTION_DEFINITION testing_input_min(Run *run)
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

FUNCTION_DEFINITION testing_input_read(Run *run)
{
    return run->problem.input.array[run->input_y][run->input_x];
}

FUNCTION_DEFINITION testing_output_read_previous(Run *run)
{
    if (run->output_x > 0)
        return run->output[run->output_y][run->output_x - 1];

    run->status = -1;

    return 0;
}

FUNCTION_DEFINITION testing_output_read(Run *run)
{
    return run->output[run->output_y][run->output_x];

    return 0;
}

FUNCTION_DEFINITION testing_reset_input_position(Run *run)
{
    run->input_x = 0;
    return 0;
}

FUNCTION_DEFINITION testing_reset_input_down_position(Run *run)
{
    run->input_y = 0;
    return 0;
}

FUNCTION_DEFINITION testing_reset_output_position(Run *run)
{
    run->output_x = 0;
    return 0;
}

FUNCTION_DEFINITION testing_reset_output_down_position(Run *run)
{
    run->output_y = 0;
    return 0;
}

FUNCTION_DEFINITION testing_output_move_left(Run *run)
{
    if (run->output_x > 0)
        run->output_x--;
    else
        run->status = -1;

    return 0;
}

FUNCTION_DEFINITION testing_output_move_right(Run *run)
{
    if (run->output_x < (run->problem.output.x - 1))
        run->output_x++;
    else
        run->status = -1;

    return 0;
}

FUNCTION_DEFINITION testing_output_move_down(Run *run)
{
    if (run->output_y < (run->problem.output.y - 1))
        run->output_y++;
    else
        run->status = -1;
    return 0;
}

FUNCTION_DEFINITION testing_output_move_up(Run *run)
{
    if (run->output_y > 0)
        run->output_y--;
    else
        run->status = -1;

    return 0;
}

FUNCTION_DEFINITION testing_is_output_end(Run *run)
{
    return run->output_x == (run->problem.output.x - 1);
}

FUNCTION_DEFINITION testing_is_output_down(Run *run)
{
    return run->output_y == (run->problem.output.y - 1);
}

FUNCTION_DEFINITION testing_input_move_left(Run *run)
{
    if (run->input_x > 0)
        run->input_x--;
    else
        run->status = -1;

    return 0;
}

FUNCTION_DEFINITION testing_input_move_right(Run *run)
{
    if (run->input_x < (run->problem.input.x - 1))
        run->input_x++;
    else
        run->status = -1;

    return 0;
}

FUNCTION_DEFINITION testing_input_move_down(Run *run)
{
    if (run->input_y < (run->problem.input.y - 1))
        run->input_y++;
    else
        run->status = -1;

    return 0;
}

FUNCTION_DEFINITION testing_input_move_up(Run *run)
{
    if (run->input_y > 0)
        run->input_y--;
    else
        run->status = -1;

    return 0;
}

FUNCTION_DEFINITION bigger_than_output_next(Run *run)
{
    if (run->problem.n_training > 0)
    {
        if (run->training_output_x < (run->problem.training[run->training_id][1].x - 1))
        {
            return run->problem.training[run->training_id][1].array[run->training_output_y][run->training_output_x] >
                   run->problem.training[run->training_id][1].array[run->training_output_y][run->training_output_x + 1];
        }
        else
        {
            run->status = -1;
        }
    }
    return 0;
}

FUNCTION_DEFINITION bigger_than_testing_output_next(Run *run)
{
    if (run->output_x < (run->problem.output.x - 1))
    {
        return run->output[run->output_y][run->output_x] >
               run->output[run->output_y][run->output_x + 1];
    }
    else
    {
        run->status = -1;
    }

    return 0;
}

FUNCTION_DEFINITION swap_testing_output_next(Run *run)
{
    if (run->output_x < (run->problem.output.x - 1))
    {
        int tmp = run->output[run->output_y][run->output_x];
        run->output[run->output_y][run->output_x] = run->output[run->output_y][run->output_x + 1];
        run->output[run->output_y][run->output_x + 1] = tmp;
    }
    else
    {
        run->status = -1;
    }

    return 0;
}

FUNCTION_DEFINITION read_memory(Run *run)
{
    return run->memory;
}

struct SNode
{
    int node_pointer;
    int case_operation;
};

FUNCTION_DEFINITION function_switch(int pointer, Run *run)
{
    SNode stack[1000];
    int s_pointer = 1;

    // Root of the program tree
    stack[0] = {pointer, run->nodes[pointer].pointer};

    int reg = 0;
    int reg1 = 0;
    int reg2 = 0;

    // Till no more nodes are available
    while (s_pointer > 0 && run->status == 0)
    {
        SNode *node = &stack[--s_pointer];

        switch (node->case_operation)
        {
        case 0:
            reg = get0(NULL);
            break;
        case 1:
            reg = get1(NULL);
            break;
        case 2:
            reg = get2(NULL);
            break;
        case 3:
            reg = get3(NULL);
            break;
        case 4:
            reg = get4(NULL);
            break;
        case 5:
            reg = get5(NULL);
            break;
        case 6:
            reg = get6(NULL);
            break;
        case 7:
            reg = get7(NULL);
            break;
        case 8:
            reg = get8(NULL);
            break;
        case 9:
            reg = get9(NULL);
            break;
        case 10:
            reg = input_end(run);
            break;
        case 11:
            reg = input_beginning(run);
            break;
        case 12:
            reg = input_down_end(run);
            break;
        case 13:
            reg = output_end(run);
            break;
        case 14:
            reg = output_beginning(run);
            break;
        case 15:
            reg = output_down_end(run);
            break;
        case 16:
            reg = output_move_left(run);
            break;
        case 17:
            reg = output_move_right(run);
            break;
        case 18:
            reg = output_move_down(run);
            break;
        case 19:
            reg = output_move_up(run);
            break;
        case 20:
            reg = get_input_position_x(run);
            break;
        case 21:
            reg = get_input_position_y(run);
            break;
        case 22:
            reg = get_output_position_x(run);
            break;
        case 23:
            reg = get_output_position_y(run);
            break;
        case 24:
            reg = get_length_input_x(run);
            break;
        case 25:
            reg = get_length_input_y(run);
            break;
        case 26:
            reg = get_length_output_x(run);
            break;
        case 27:
            reg = get_length_output_y(run);
            break;
        case 28:
            reg = input_next(run);
            break;
        case 29:
            reg = input_previous(run);
            break;
        case 30:
            reg = input_move_left(run);
            break;
        case 31:
            reg = input_move_right(run);
            break;
        case 32:
            reg = input_move_down(run);
            break;
        case 33:
            reg = input_move_up(run);
            break;
        case 34:
            reg = reset_input_position(run);
            break;
        case 35:
            reg = reset_input_down_position(run);
            break;
        case 36:
            reg = run->inputInstance.max;
            break;
        case 37:
            reg = run->inputInstance.min;
            break;
        case 38:
            reg = input_read(run);
            break;
        case 39:
            reg = output_read(run);
            break;
        case 40:
            reg = reset_output_position(run);
            break;
        case 41:
            reg = reset_output_down_position(run);
            break;
        case 42:
            reg = get_testing_length_input_x(run);
            break;
        case 43:
            reg = get_testing_length_input_y(run);
            break;
        case 44:
            reg = get_testing_length_output_x(run);
            break;
        case 45:
            reg = get_testing_length_output_y(run);
            break;
        case 46:
            reg = get_testing_input_position_y(run);
            break;
        case 47:
            reg = get_testing_input_position_x(run);
            break;
        case 48:
            reg = get_testing_output_position_y(run);
            break;
        case 49:
            reg = get_testing_output_position_x(run);
            break;
        case 50:
            reg = testing_input_max(run);
            break;
        case 51:
            reg = testing_input_min(run);
            break;
        case 52:
            reg = testing_input_read(run);
            break;
        case 53:
            reg = testing_output_read_previous(run);
            break;
        case 54:
            reg = testing_output_read(run);
            break;
        case 55:
            reg = testing_reset_input_position(run);
            break;
        case 56:
            reg = testing_reset_input_down_position(run);
            break;
        // Write operations missing
        case 57: // testing_output_write_previous
            if (run->output_x > 0)
            {

                Node *pnode = &run->nodes[node->node_pointer];
                stack[s_pointer++] = {0, 101};

                // Add read value node
                stack[s_pointer++] = {pnode->args[0], run->nodes[pnode->args[0]].pointer};
            }
            else
            {
                run->status = -1;
            }

            break;
        case 58: // testing_output_write
        {
            Node *pnode = &run->nodes[node->node_pointer];
            stack[s_pointer++] = {0, 102};

            // Add read value node
            stack[s_pointer++] = {pnode->args[0], run->nodes[pnode->args[0]].pointer};
        }
        break;
        case 59:
            reg = testing_reset_output_position(run);
            break;
        case 60:
            reg = testing_reset_output_down_position(run);
            break;
        case 61:
            reg = testing_output_move_left(run);
            break;
        case 62:
            reg = testing_output_move_right(run);
            break;
        case 63:
            reg = testing_output_move_down(run);
            break;
        case 64:
            reg = testing_output_move_up(run);
            break;
        case 65:
            reg = testing_is_output_end(run);
            break;
        case 66:
            reg = testing_is_output_down(run);
            break;
        case 67:
            reg = testing_input_move_left(run);
            break;
        case 68:
            reg = testing_input_move_right(run);
            break;
        case 69:
            reg = testing_input_move_down(run);
            break;
        case 70:
            reg = testing_input_move_up(run);
            break;
        case 71: // comparison
        {
            Node *pnode = &run->nodes[node->node_pointer];
            stack[s_pointer++] = {node->node_pointer, 104};

            // Add read value node
            stack[s_pointer++] = {pnode->args[0], run->nodes[pnode->args[0]].pointer};
        }
        break;
        case 72:
            reg = bigger_than_output_next(run);
            break;

        case 73:
            reg = bigger_than_testing_output_next(run);
            break;
        case 74:
            reg = swap_testing_output_next(run);
            break;
        case 75: // bigger_than
        {
            Node *pnode = &run->nodes[node->node_pointer];
            stack[s_pointer++] = {node->node_pointer, 108};

            // Add read value node
            stack[s_pointer++] = {pnode->args[0], run->nodes[pnode->args[0]].pointer};
        }
        break;
        case 76: // equal
        {
            Node *pnode = &run->nodes[node->node_pointer];
            stack[s_pointer++] = {node->node_pointer, 106};

            // Add read value node
            stack[s_pointer++] = {pnode->args[0], run->nodes[pnode->args[0]].pointer};
        }
        break;
        case 77: // no
        {
            Node *pnode = &run->nodes[node->node_pointer];
            stack[s_pointer++] = {node->node_pointer, 105};

            // Add read value node
            stack[s_pointer++] = {pnode->args[0], run->nodes[pnode->args[0]].pointer};
        }
        break;
        case 78: // prog2
        {
            Node *pnode = &run->nodes[node->node_pointer];
            stack[s_pointer++] = {pnode->args[1], run->nodes[pnode->args[1]].pointer};
            stack[s_pointer++] = {pnode->args[0], run->nodes[pnode->args[0]].pointer};
        }
        break;
        case 79: // prog3
        {
            Node *pnode = &run->nodes[node->node_pointer];
            stack[s_pointer++] = {pnode->args[2], run->nodes[pnode->args[2]].pointer};
            stack[s_pointer++] = {pnode->args[1], run->nodes[pnode->args[1]].pointer};
            stack[s_pointer++] = {pnode->args[0], run->nodes[pnode->args[0]].pointer};
        }
        break;
        case 81:
            reg = read_memory(run);
            break;
        case 82: // write memory
        {
            Node *pnode = &run->nodes[node->node_pointer];
            stack[s_pointer++] = {0, 103};

            // Add read value node
            // Node p0node = run->nodes[node.node_pointer];
            stack[s_pointer++] = {pnode->args[0], run->nodes[pnode->args[0]].pointer};
        }
        break;
        case 83: // loop
        {
            run->inner_loop++;

            // Add operation node n-times
            Node *pnode = &run->nodes[node->node_pointer];

            stack[s_pointer++] = {pnode->args[1], 100};

            // Add read value node
            stack[s_pointer++] = {pnode->args[0], run->nodes[pnode->args[0]].pointer};
        }
        break;
        case 100:
            if (run->inner_loop < 5 && reg > 0 && reg <= 30)
            {
                // Read register
                stack[s_pointer++] = {0, 110};
                for (int i = 0; i < reg; i++)
                {
                    // Loop operation
                    stack[s_pointer++] = {node->node_pointer, run->nodes[node->node_pointer].pointer};
                }
            }
            else
            {
                run->status = -2;
            }

            break;
        case 110:
            run->inner_loop--;
            break;
        case 101: // testing_output_write_previous
            run->output[run->output_y][run->output_x - 1] = reg;
            break;
        case 102: // testing_output_write
            run->output[run->output_y][run->output_x] = reg;
            break;
        case 1003: // write memory
            run->memory = reg;
            break;
        case 104: // comparison
        {
            Node *pnode = &run->nodes[node->node_pointer];
            if (reg)
            {
                stack[s_pointer++] = {pnode->args[0], run->nodes[pnode->args[0]].pointer};
            }
            else
            {
                stack[s_pointer++] = {pnode->args[1], run->nodes[pnode->args[1]].pointer};
            }
        }
        break;
        case 105: // no
            reg = !reg;
            break;
        case 106: // equal
        {
            // copy register
            reg1 = reg;
            // final check
            stack[s_pointer++] = {node->node_pointer, 107};
            // obtain second value
            Node *pnode = &run->nodes[node->node_pointer];
            stack[s_pointer++] = {pnode->args[1], run->nodes[pnode->args[1]].pointer};
        }
        break;
        case 107: // equal
            reg = (reg == reg1);
            break;
        case 108: // bigger_than
        {
            // copy register
            reg1 = reg;
            // final check
            stack[s_pointer++] = {node->node_pointer, 109};
            // obtain second value
            Node *pnode = &run->nodes[node->node_pointer];
            stack[s_pointer++] = {pnode->args[1], run->nodes[pnode->args[1]].pointer};
        }
        break;
        case 109:
            reg = (reg1 > reg);
            break;
        case 84: // testing_set_input_position
        {
            Node *pnode = &run->nodes[node->node_pointer];
            stack[s_pointer++] = {node->node_pointer, 120};

            stack[s_pointer++] = {pnode->args[0], run->nodes[pnode->args[0]].pointer};
        }
        break;
        case 120:
        {
            reg1 = reg;
            Node *pnode = &run->nodes[node->node_pointer];
            stack[s_pointer++] = {node->node_pointer, 121};

            stack[s_pointer++] = {pnode->args[1], run->nodes[pnode->args[1]].pointer};
        }
        break;
        case 121:
            if (reg >= 0 && reg < get_testing_length_input_x(run) && reg1 >= 0 && reg1 < get_testing_length_input_y(run))
            {
                run->input_x = reg;
                run->input_y = reg1;
            }
            else
            {
                run->status = -1;
            }

            break;
        case 85: // testing_set_output_position
        {
            Node *pnode = &run->nodes[node->node_pointer];
            stack[s_pointer++] = {node->node_pointer, 122};

            stack[s_pointer++] = {pnode->args[0], run->nodes[pnode->args[0]].pointer};
        }
        break;
        case 122:
        {
            reg1 = reg;
            Node *pnode = &run->nodes[node->node_pointer];
            stack[s_pointer++] = {node->node_pointer, 123};

            stack[s_pointer++] = {pnode->args[1], run->nodes[pnode->args[1]].pointer};
        }
        break;
        case 123:
            if (reg >= 0 && reg < get_testing_length_output_x(run) && reg1 >= 0 && reg1 < get_testing_length_output_y(run))
            {
                run->output_x = reg;
                run->output_y = reg1;
            }
            else
            {
                run->status = -1;
            }

            break;
        case 86: // testing_set_output_value
        {
            Node *pnode = &run->nodes[node->node_pointer];
            stack[s_pointer++] = {node->node_pointer, 124};

            stack[s_pointer++] = {pnode->args[0], run->nodes[pnode->args[0]].pointer};
        }
        break;
        case 124:
        {
            reg1 = reg;
            Node *pnode = &run->nodes[node->node_pointer];
            stack[s_pointer++] = {node->node_pointer, 125};

            stack[s_pointer++] = {pnode->args[1], run->nodes[pnode->args[1]].pointer};
        }
        break;
        case 125:
        {
            reg2 = reg;
            Node *pnode = &run->nodes[node->node_pointer];
            stack[s_pointer++] = {node->node_pointer, 126};

            stack[s_pointer++] = {pnode->args[2], run->nodes[pnode->args[2]].pointer};
        }
        break;
        case 126:
            if (reg1 >= 0 && reg1 < get_testing_length_output_x(run) && reg2 >= 0 && reg2 < get_testing_length_output_y(run))
            {
                run->output[reg2][reg1] = reg;
            }
            else
            {
                run->status = -1;
            }

            break;
        case 87: // testing_get_input_value
        {
            Node *pnode = &run->nodes[node->node_pointer];
            stack[s_pointer++] = {node->node_pointer, 127};

            stack[s_pointer++] = {pnode->args[0], run->nodes[pnode->args[0]].pointer};
        }
        break;
        case 127:
        {
            reg1 = reg;
            Node *pnode = &run->nodes[node->node_pointer];
            stack[s_pointer++] = {node->node_pointer, 128};

            stack[s_pointer++] = {pnode->args[1], run->nodes[pnode->args[1]].pointer};
        }
        break;
        case 128:
            if (reg >= 0 && reg < get_testing_length_input_x(run) && reg1 >= 0 && reg1 < get_testing_length_input_y(run))
            {
                reg = run->problem.input.array[reg1][reg];
            }
            else
            {
                run->status = -1;
            }

            break;
        case 88: // testing_get_output_value
        {
            Node *pnode = &run->nodes[node->node_pointer];
            stack[s_pointer++] = {node->node_pointer, 129};

            stack[s_pointer++] = {pnode->args[0], run->nodes[pnode->args[0]].pointer};
        }
        break;
        case 129:
        {
            reg1 = reg;
            Node *pnode = &run->nodes[node->node_pointer];
            stack[s_pointer++] = {node->node_pointer, 130};

            stack[s_pointer++] = {pnode->args[1], run->nodes[pnode->args[1]].pointer};
        }
        break;
        case 130:
            if (reg >= 0 && reg < get_testing_length_output_x(run) && reg1 >= 0 && reg1 < get_testing_length_output_y(run))
            {
                reg = run->output[reg1][reg];
            }
            else
            {
                run->status = -1;
            }

            break; 
        default:
            run->status = -1;
            break;
        }
    }

    return 0;
}

#define MAP_INSTRUCTIONS std::unordered_map<STRING, int>

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

    map["testing_set_input_position"] = 84;
    map["testing_set_output_position"] = 85;
    map["testing_set_output_value"] = 86;
    map["testing_get_input_value"] = 87;
    map["testing_get_output_value"] = 88;


    return map;
}

inline int getProgram(const STRING *string, MAP_INSTRUCTIONS &map, std::vector<Node> *nodes, int &position)
{
    int program = -1;

    int initial_position = position;
    const int length = string->length();
    const char *cstr = string->data();

    while (initial_position < length && cstr[initial_position] == ' ')
        initial_position++;

    // Traverse string into tree
    while (position < length)
    {
        switch (cstr[position])
        {
        case '(':
        {
            // Create new program entry
            program = nodes->size();
            int pointer = map[string->substr(initial_position, position - initial_position)];

            nodes->push_back({pointer, 0, {0, 0, 0}});

            position++;

            if (cstr[position] == ')')
            {
                position++;
                return program;
            }
            else
            {
                int sub = getProgram(string, map, nodes, position);
                nodes->at(program).args[nodes->at(program).n_args++] = sub;
            }
        }
        break;
        case ')':
            position++;
            return program;
            break;
        case ',':
        {
            position++;

            int sub = getProgram(string, map, nodes, position);
            nodes->at(program).args[nodes->at(program).n_args++] = sub;
        }
        break;
        default:
            position++;
            break;
        }
    }

    return program;
}

inline void getProgram(const STRING *string, MAP_INSTRUCTIONS &map, std::vector<Node> *nodes)
{
    int position = 0;
    getProgram(string, map, nodes, position);
}

inline void copy_program(int start_index, int end_index, std::vector<int> *programs, std::vector<Node> *nodes, STRING **code, MAP_INSTRUCTIONS map)
{
    for (int i = start_index; i < end_index; ++i)
    {
        std::vector<Node> subnodes;
        getProgram(code[i], map, &subnodes);

        programs->push_back(nodes->size());

        for (long unsigned int i = 0; i < subnodes.size(); i++)
        {
            nodes->push_back({subnodes[i].pointer, subnodes[i].n_args, {subnodes[i].args[0], subnodes[i].args[1], subnodes[i].args[2]}});
        }
    }
}

Programs *copy_programs_to_gpu(int n_programs, STRING **code)
{
    MAP_INSTRUCTIONS map = get_map();

    // Create array of programs in host memory
    Programs *d_sprograms;
    // cudaMallocManaged(&d_sprograms, sizeof(Programs));
    allocate_memory((void **)&d_sprograms, sizeof(Programs));

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

    // cudaMallocManaged(&d_sprograms->nodes, d_sprograms->n_nodes * sizeof(Node));
    allocate_memory((void **)&d_sprograms->programs, d_sprograms->n_programs * sizeof(int));

    // cudaMallocManaged(&d_sprograms->programs, d_sprograms->n_programs * sizeof(int));
    allocate_memory((void **)&d_sprograms->nodes, d_sprograms->n_nodes * sizeof(Node));

    size_t offset_nodes = 0;
    size_t offset_programs = 0;
    for (int i = 0; i < n_threads; i++)
    {
        std::copy(nodes[i].begin(), nodes[i].end(), d_sprograms->nodes + offset_nodes);

        for (long unsigned int j = 0; j < programs[i].size(); j++)
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
    // cudaFree(programs->nodes);
    free_memory(programs->programs);
    // cudaFree(programs->programs);
    free_memory(programs->nodes);
    // cudaFree(programs);
    free_memory(programs);
    return 0;
}

#endif