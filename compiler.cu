#ifndef PROGRAM_COMPILER_C
#define PROGRAM_COMPILER_C

#include "instructions.cu"
#include "types.cuh"
#include <iostream>
#include <thread>
#include <unordered_map>
#include <vector>

#define MAP_INSTRUCTIONS std::unordered_map<std::string, int (*)(Run *, Program **)>

MAP_INSTRUCTIONS get_map()
{

    MAP_INSTRUCTIONS map;

    map["get0"] = &get0;
    map["testing_output_write"] = &testing_output_write;

    return map;
}

Program *getProgram(std::string string, MAP_INSTRUCTIONS map, int &position)
{
    struct Program *program = NULL;

    std::vector<struct Program *> subprograms;

    int initial_position = position;

    while (initial_position < string.length() && string[initial_position] == ' ')
        initial_position++;

    // Traverse string into tree
    while (position < string.length())
    {
        if (string[position] == '(')
        {
            // Create new program entry
            program = new struct Program();
            program->pointer = map[string.substr(initial_position, position - initial_position)];

            position++;

            if (string[position] == ')')
            {
                position++;
                return program;
            }
            else
            {
                Program *sub = getProgram(string, map, position);
                subprograms.push_back(sub);
            }
        }
        else if (string[position] == ')')
        {
            // Close and return
            position++;
            program->n_args = subprograms.size();
            program->args = new struct Program *[subprograms.size()];
            std::copy(subprograms.begin(), subprograms.end(), program->args);

            return program;
        }
        else if (string[position] == ',')
        {
            // After this, there is a new program
            position++;
            subprograms.push_back(getProgram(string, map, position));
        }
        else
        {
            position++;
        }
    }

    return program;
}

Program *getProgram(std::string string, MAP_INSTRUCTIONS map)
{
    int position = 0;
    return getProgram(string, map, position);
}

#endif

int main()
{
    std::string c("testing_output_write(get0())");

    MAP_INSTRUCTIONS map = get_map();

    for (int i = 0; i < 30000000; i++)
    {
        getProgram(c, map);
    }

    Program * p = getProgram(c, map);

    std::cout << p->pointer << std::endl;

    return 0;
}