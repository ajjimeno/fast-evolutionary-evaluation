import SimulatorCPU

#s = SimulatorCPU.Runner("/home/antonio/git/gp/arc_training_tag/00d62c1b")
#s = SimulatorCPU.Runner("/Users/ajimeno/Documents/git/gp/arc_training_tag_test/a3325580")
s = SimulatorCPU.Runner("/Users/ajimeno/Documents/git/gp/arc_training_tag_test/1190e5a7")

from tqdm import tqdm

n=3000

program="comparison(testing_is_output_end(), prog2(testing_set_output_value(get1(), get_testing_output_position_y(), get2()), prog2(testing_set_output_value(aligned_above(), testing_input_max(), get2()), prog2(testing_set_output_value(get1(), get_testing_output_position_y(), get_testing_output_position_y()), prog2(testing_set_output_value(aligned_left(), get7(), get2()), prog2(testing_set_output_value(get2(), get_testing_length_output_x(), get2()), prog2(testing_set_output_value(get1(), get_testing_output_position_y(), get2()), prog2(testing_output_write(testing_get_output_value(get_testing_output_position_y(), testing_output_read())), prog2(comparison(in_input_shape(), prog2(testing_output_move_down(), prog2(testing_set_output_value(get2(), get_testing_output_position_y(), get2()), testing_output_write(get2()))), testing_output_move_right()), testing_output_move_down())))))))), testing_input_move_up())"
program="prog2(testing_set_input_position(get3(), testing_input_max()), prog2(testing_output_write(get6()), prog2(comparison(testing_is_output_end(), testing_output_move_right(), loop(get_testing_input_position_y(), swap_testing_output_next())), prog2(testing_output_write(get4()), prog2(comparison(bigger_thanR(get9(), aligned_below()), testing_input_move_up(), testing_output_write(get7())), prog2(testing_output_write(get4()), prog2(comparison(testing_is_output_end(), testing_output_move_right(), testing_set_input_position(get5(), aligned_above())), prog2(testing_output_write(testing_input_max()), prog2(comparison(testing_is_output_end(), testing_output_move_down(), testing_reset_input_position()), prog2(testing_reset_output_position(), prog2(comparison(testing_is_output_end(), testing_input_move_down(), loop(testing_input_max(), swap_testing_output_next())), prog2(testing_output_write(testing_input_max()), prog2(comparison(bigger_thanR(get_testing_length_output_y(), testing_input_read()), swap_testing_output_next(), loop(get_testing_length_input_y(), swap_testing_output_next())), testing_output_move_down())))))))))))))"

program="prog2(testing_set_output_value(get_testing_input_position_y(), get_testing_output_position_x(), get_max_color()), prog2(prog2(swap_testing_output_next(), prog2(testing_output_write(testing_input_read()), testing_output_move_down())), prog2(testing_output_write(testing_input_read()), testing_set_output_position(testing_output_distance_to_input_x(), get_testing_output_position_y()))))"

program="write_memory(get1(), get2())"

program="prog2(write_memory(get1(), get8()), testing_output_write(read_memory(get1())))"

print(*s.runProgram(program), sep="\n")

"""
for _ in tqdm(range(100)):
     #x=s.run(["prog2(testing_output_write(testing_input_max()), comparison(bigger_than_testing_output_next(), prog2(prog2(testing_output_write_previous(testing_input_min()), comparison(bigger_than_testing_output_next(), testing_output_move_right(), testing_reset_input_position())), comparison(bigger_than_testing_output_next(), testing_input_move_right(), prog2(testing_output_write_previous(testing_input_min()), comparison(bigger_than_testing_output_next(), prog2(testing_input_move_up(), comparison(testing_is_output_down(), testing_reset_input_position(), testing_output_move_down())), testing_reset_input_position())))), prog2(testing_input_move_down(), comparison(bigger_than_testing_output_next(), testing_reset_input_down_position(), prog2(testing_output_write(testing_input_max()), comparison(bigger_than_testing_output_next(), prog2(prog2(testing_output_move_right(), comparison(bigger_thanR(testing_input_read(), get_testing_output_position_x()), testing_input_move_up(), prog2(testing_input_move_up(), comparison(testing_is_output_down(), testing_output_move_down(), testing_reset_output_down_position())))), swap_testing_output_next()), prog2(testing_reset_output_position(), comparison(bigger_than_testing_output_next(), swap_testing_output_next(), prog2(testing_output_move_right(), comparison(equalR(testing_input_read(), testing_input_max()), testing_output_move_right(), prog2(testing_output_write_previous(testing_input_max()), comparison(testing_is_output_end(), testing_input_move_up(), prog2(testing_input_move_up(), comparison(testing_is_output_down(), testing_reset_input_position(), testing_output_move_down()))))))))))))))"]*n)
     x=s.run(["get9()"]*n)
     print(x)

     #x=s.run(["loop(get9(), loop(get9(), loop(get9(), loop(get9(), loop(get9(), loop(get9(), loop(get9(), loop(get9(), loop(get9(), loop(get9(), loop(get9(), loop(get9(), get9()))))))))))))"]*n)

     #x=s.run(["get9()"])

     #x=s.run(["comparison(equal(get9(), get9()), testing_output_write(get9()))"]*n)

     #x=s.run(["prog2(testing_output_write(get9()), testing_output_write(get9()))"]*n)

     #x=s.run(["loop(get9(),loop(get9(),testing_output_write(testing_input_max())))"]*n)

     #x=s.run(["loop(get_testing_length_input_x(), prog2(prog2(testing_input_move_right(), testing_output_move_right()),  testing_output_write(testing_input_read())))"])

     #x=s.run(["prog2(reset_output_position(), prog2(reset_output_down_position(),prog2(testing_output_move_right(),prog2(testing_output_move_right(),prog2(testing_output_move_right(),prog2(testing_output_move_right(),prog2(testing_output_move_right(),testing_output_write(get3()))))))))"])

     #x=s.run(["prog2(testing_reset_output_position(), prog2(testing_reset_output_down_position(),prog2(testing_output_move_down(),prog2(testing_output_move_down(),prog2(testing_output_move_right(),prog2(testing_output_move_right(),prog2(testing_output_move_right(),prog2(testing_output_move_right(),testing_output_write(get3())))))))))"])

     #print(x)
"""