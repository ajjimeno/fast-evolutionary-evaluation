#include "instructions.cu"
#include <iostream>
#include <map>
#include <thread>
#include <unordered_map>
#include <vector>

// nvcc compiler.cu -o compiler --compiler-options -g --compiler-options -pg

int main()
{
    //std::string c("prog2(prog2(comparison(testing_is_output_down(), swap_testing_output_next(), testing_input_move_right()), comparison(testing_is_output_end(), prog2(comparison(testing_is_output_end(), prog2(comparison(testing_is_output_end(), prog2(comparison(testing_is_output_down(), comparison(testing_is_output_end(), prog2(testing_reset_output_down_position(), testing_input_move_down()), testing_reset_output_position()), testing_reset_output_position()), comparison(testing_is_output_end(), testing_input_move_up(), testing_output_write_previous(testing_output_read()))), testing_output_write_previous(testing_output_read())), comparison(testing_is_output_end(), testing_input_move_up(), testing_output_write_previous(testing_output_read()))), testing_output_write_previous(testing_output_read())), comparison(testing_is_output_end(), testing_input_move_up(), testing_output_write_previous(testing_output_read()))), testing_output_write_previous(testing_output_read()))), testing_output_move_right())");

    std::string c("prog2(testing_output_write(testing_input_max()), comparison(bigger_than_testing_output_next(), prog2(prog2(testing_output_write_previous(testing_input_min()), comparison(bigger_than_testing_output_next(), testing_output_move_right(), testing_reset_input_position())), comparison(bigger_than_testing_output_next(), testing_input_move_right(), prog2(testing_output_write_previous(testing_input_min()), comparison(bigger_than_testing_output_next(), prog2(testing_input_move_up(), comparison(testing_is_output_down(), testing_reset_input_position(), testing_output_move_down())), testing_reset_input_position())))), prog2(testing_input_move_down(), comparison(bigger_than_testing_output_next(), testing_reset_input_down_position(), prog2(testing_output_write(testing_input_max()), comparison(bigger_than_testing_output_next(), prog2(prog2(testing_output_move_right(), comparison(bigger_thanR(testing_input_read(), get_testing_output_position_x()), testing_input_move_up(), prog2(testing_input_move_up(), comparison(testing_is_output_down(), testing_output_move_down(), testing_reset_output_down_position())))), swap_testing_output_next()), prog2(testing_reset_output_position(), comparison(bigger_than_testing_output_next(), swap_testing_output_next(), prog2(testing_output_move_right(), comparison(equalR(testing_input_read(), testing_input_max()), testing_output_move_right(), prog2(testing_output_write_previous(testing_input_max()), comparison(testing_is_output_end(), testing_input_move_up(), prog2(testing_input_move_up(), comparison(testing_is_output_down(), testing_reset_input_position(), testing_output_move_down()))))))))))))))");

    //std::string c("equalW(input_read(), output_read())");

    MAP_INSTRUCTIONS map = get_map();

    for (int i=0; i<30000;i++)
{
    std::map<int, Node> nodes;
    getProgram(c, map, &nodes);
}
    /*
    for (int i = 0; i < nodes.size(); i++)
    {
        std::cout << nodes[i].pointer << std::endl;
        std::cout << nodes[i].n_args << std::endl;
        std::cout << nodes[i].args[0] << std::endl;
        std::cout << nodes[i].args[1] << std::endl;
        std::cout << nodes[i].args[2] << std::endl;
        std::cout << "====" << std::endl;
    }*/

    return 0;
}