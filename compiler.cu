#include "instructions.cu"
#include <iostream>
#include <map>
#include <thread>
#include <unordered_map>
#include <vector>



int main()
{
    //std::string c("prog2(prog2(comparison(testing_is_output_down(), swap_testing_output_next(), testing_input_move_right()), comparison(testing_is_output_end(), prog2(comparison(testing_is_output_end(), prog2(comparison(testing_is_output_end(), prog2(comparison(testing_is_output_down(), comparison(testing_is_output_end(), prog2(testing_reset_output_down_position(), testing_input_move_down()), testing_reset_output_position()), testing_reset_output_position()), comparison(testing_is_output_end(), testing_input_move_up(), testing_output_write_previous(testing_output_read()))), testing_output_write_previous(testing_output_read())), comparison(testing_is_output_end(), testing_input_move_up(), testing_output_write_previous(testing_output_read()))), testing_output_write_previous(testing_output_read())), comparison(testing_is_output_end(), testing_input_move_up(), testing_output_write_previous(testing_output_read()))), testing_output_write_previous(testing_output_read()))), testing_output_move_right())");

    std::string c("equalW(input_read(), output_read())");

    MAP_INSTRUCTIONS map = get_map();

    std::map<int, Node> nodes;
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