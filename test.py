import SimulatorCPU

#s = SimulatorCPU.Runner("/home/antonio/git/gp/arc_training_tag/00d62c1b")
s = SimulatorCPU.Runner("/Users/ajimeno/Documents/git/gp/arc_training_tag/00d62c1b")

from tqdm import tqdm

n=300000

for _ in tqdm(range(100)):
     x=s.run(["prog2(testing_output_write(testing_input_max()), comparison(bigger_than_testing_output_next(), prog2(prog2(testing_output_write_previous(testing_input_min()), comparison(bigger_than_testing_output_next(), testing_output_move_right(), testing_reset_input_position())), comparison(bigger_than_testing_output_next(), testing_input_move_right(), prog2(testing_output_write_previous(testing_input_min()), comparison(bigger_than_testing_output_next(), prog2(testing_input_move_up(), comparison(testing_is_output_down(), testing_reset_input_position(), testing_output_move_down())), testing_reset_input_position())))), prog2(testing_input_move_down(), comparison(bigger_than_testing_output_next(), testing_reset_input_down_position(), prog2(testing_output_write(testing_input_max()), comparison(bigger_than_testing_output_next(), prog2(prog2(testing_output_move_right(), comparison(bigger_thanR(testing_input_read(), get_testing_output_position_x()), testing_input_move_up(), prog2(testing_input_move_up(), comparison(testing_is_output_down(), testing_output_move_down(), testing_reset_output_down_position())))), swap_testing_output_next()), prog2(testing_reset_output_position(), comparison(bigger_than_testing_output_next(), swap_testing_output_next(), prog2(testing_output_move_right(), comparison(equalR(testing_input_read(), testing_input_max()), testing_output_move_right(), prog2(testing_output_write_previous(testing_input_max()), comparison(testing_is_output_end(), testing_input_move_up(), prog2(testing_input_move_up(), comparison(testing_is_output_down(), testing_reset_input_position(), testing_output_move_down()))))))))))))))"]*n)

     #x=s.run(["loop(get9(), loop(get9(), loop(get9(), loop(get9(), loop(get9(), loop(get9(), loop(get9(), loop(get9(), loop(get9(), loop(get9(), loop(get9(), loop(get9(), get9()))))))))))))"]*n)

     #x=s.run(["get9()"])

     #x=s.run(["comparison(equal(get9(), get9()), testing_output_write(get9()))"]*n)

     #x=s.run(["prog2(testing_output_write(get9()), testing_output_write(get9()))"]*n)

     #x=s.run(["loop(get9(),loop(get9(),testing_output_write(testing_input_max())))"]*n)

     #x=s.run(["loop(get_testing_length_input_x(), prog2(prog2(testing_input_move_right(), testing_output_move_right()),  testing_output_write(testing_input_read())))"])

     #x=s.run(["prog2(reset_output_position(), prog2(reset_output_down_position(),prog2(testing_output_move_right(),prog2(testing_output_move_right(),prog2(testing_output_move_right(),prog2(testing_output_move_right(),prog2(testing_output_move_right(),testing_output_write(get3()))))))))"])

     #x=s.run(["prog2(testing_reset_output_position(), prog2(testing_reset_output_down_position(),prog2(testing_output_move_down(),prog2(testing_output_move_down(),prog2(testing_output_move_right(),prog2(testing_output_move_right(),prog2(testing_output_move_right(),prog2(testing_output_move_right(),testing_output_write(get3())))))))))"])

     #print(x)