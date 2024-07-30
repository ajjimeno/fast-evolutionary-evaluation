#ifndef PROGRAM_COMPILER_C
#define PROGRAM_COMPILER_C

// #include "instructions.cu"
#include "types.cuh"
#include <iostream>
#include <thread>
#include <unordered_map>
#include <vector>

#define MAP_INSTRUCTIONS std::unordered_map<std::string, int>

MAP_INSTRUCTIONS get_map()
{

    MAP_INSTRUCTIONS map;

    map["get0"] = 0;
    map["testing_output_write"] = 1;

    return map;
};

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

#endif

int main()
{
    std::string c("testing_output_write(get0())");

    MAP_INSTRUCTIONS map = get_map();

    for (int i = 0; i < 3000000; i++)
    {
        std::vector<Node> nodes;
        getProgram(c, map, &nodes);
    }

    std::vector<Node> nodes;
    getProgram(c, map, &nodes);

    for (int i = 0; i < nodes.size(); i++)
    {
        std::cout << nodes[i].pointer << std::endl;
        std::cout << nodes[i].n_args << std::endl;
        std::cout << nodes[i].args[0] << std::endl;
        std::cout << nodes[i].args[1] << std::endl;
        std::cout << nodes[i].args[2] << std::endl;
        std::cout << "====" << std::endl;
    }

    return 0;
}