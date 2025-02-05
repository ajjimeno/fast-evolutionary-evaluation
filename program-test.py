from programs import getProgram
import SimulatorCPU as simulator

n=300000

#s = simulator.Runner("/home/antonio/git/gp/arc_training_tag/c3f564a4")
s = simulator.Runner("/home/antonio/git/gp/arc_training_tag/1caeab9d")
#s = SimulatorCPU.Runner("/Users/ajimeno/Documents/git/gp/arc_training_tag/00d62c1b")

#program="prog2(write_memory(get3(), get8()),testing_output_write(read_memory(testing_get_output_value(get1(), get1()))))"
#program="prog2(write_memory(get3(), get8()),prog2(testing_output_write(read_memory(testing_get_output_value(get1(), get1()))), testing_output_move_right()))"
"""
operators = [
    # get max number
    "write_memory(get0(), get0())",
    "loop(",
    "get_testing_length_output_y()",
    ",",
    "    loop(",
    "    get_testing_length_output_x()",
    "    ,",
    "        comparison(",
    "        bigger_thanR(testing_output_read(), read_memory(get0()))",
    "        ,",
    "        write_memory(get0(), testing_output_read())",
    "        ,",
    "        write_memory(get0(), read_memory(get0()))",
    "        )",
    "    get0()",
    "    testing_output_move_right()",
    "    )",
    "testing_output_move_down()",
    ")",
    #"testing_output_write(read_memory(get0()))",

    "status_end()"
]
"""

operators = [
    """
    # Identify the y start/end of the block with 1s
    "write_memory(get0(), get0())",
    "write_memory(get1(), get0())",
    "loop(",
    "get_testing_length_output_y()",
    ",",
    "    write_memory(get0(), get0())",
    "    loop(",
    "    get_testing_length_output_x()",
    "    ,",
    "        comparison(",
    "        equalR(read_memory(get0()), get0())",
    "        ,",
    "            comparison(",
    "            equalR(test_output_read(), get1())",
    "            ,",
    "            write_memory(get0(), get1())",
    "            ,",
    "            write_memory(get0(), read_memory(get0()))",
    "            )",
    "        ,",
    "        write_memory(get0(), read_memory(get0()))",
    "        )",
    "    testing_output_move_right()",
    "    )",
    "    comparison(",
    "    equalR(read_memory(get1()), get0())",
    "    ,",
    "        comparison(",
    "        ,",
    "        ,",
    "        )",
    "    ,",
    "    )",
    "testing_set_output_position(get0(), get_testing_output_position_y())",
    "testing_output_move_down()",
    ")",
    """
    # Move block with 2s

    # Move block with 4s

    "status_end()"
]

"""
operators= [
    "testing_set_output_position(get0(), sub(get_testing_length_output_y(), get1()))",
    "loop(",
    "get_testing_length_output_x()",
    ",",
    "    comparison(",
    "    equalR(testing_output_read(), get2())",
    "    ,",
    "        loop(",
    "        get_testing_length_output_y()",
    "        ,",
    "           comparison(",
    "           equalR(testing_output_read(), get5())",
    "           ,",
    "           testing_output_move_down()",
    "           testing_output_move_right()",
    "           testing_output_write(get2())",
    "           testing_output_move_up()",
    "           testing_output_write(get2())",
    "           testing_output_move_up()",
    "           ,",
    "           testing_output_write(get2())",
    "           testing_output_move_up()",
    "           )",
    "        )",
    "    testing_set_output_position(get_testing_output_position_x(), sub(get_testing_length_output_y(), get1()))",
    "    ,",
    "    testing_output_write(testing_output_read())",
    "    )",
    "testing_output_move_right()",
    ")",
    "status_end()"
]
"""

"""
    "testing_set_input_position(get0(), get2())",
    "loop(",
    "get9()",
    ",",
    "    comparison(",
    "    equalR(testing_input_read(), get0())",
    "    ,",
    "    testing_input_move_right()",
    "    ,",
    "    write_memory(get0(), testing_input_read())",
    "    )",
    ")",
    "loop(",
    "get9()",
    ",",
    "testing_output_write(read_memory(get0()))",
    "testing_output_move_right()",
    ")",
    "testing_set_output_position(get0(), get2())",
    "loop(",
    "get9()",
    ",",
    "testing_output_write(read_memory(get0()))",
    "testing_output_move_right()",
    ")",
    "testing_set_output_position(get0(), get0())",
    "loop(",
    "get5()",
    ",",
    "testing_output_write(read_memory(get0()))",
    "testing_output_move_down()",
    ")",
    "testing_set_output_position(get9(), get0())",
    "loop(",
    "get5()",
    ",",
    "testing_output_write(read_memory(get0()))",
    "testing_output_move_down()",
    ")",
    "testing_set_input_position(get0(), get7())",
    "loop(",
    "get9()",
    ",",
    "    comparison(",
    "    equalR(testing_input_read(), get0())",
    "    ,",
    "    testing_input_move_right()",
    "    ,",
    "    write_memory(get0(), testing_input_read())",
    "    )",
    ")",
    "testing_set_output_position(get0(), get7())",
    "loop(",
    "get9()",
    ",",
    "testing_output_write(read_memory(get0()))",
    "testing_output_move_right()",
    ")",
    "testing_set_output_position(get0(), get9())",
    "loop(",
    "get9()",
    ",",
    "testing_output_write(read_memory(get0()))",
    "testing_output_move_right()",
    ")",
    "testing_set_output_position(get0(), get5())",
    "loop(",
    "get5()",
    ",",
    "testing_output_write(read_memory(get0()))",
    "testing_output_move_down()",
    ")",
    "testing_set_output_position(get9(), get5())",
    "loop(",
    "get5()",
    ",",
    "testing_output_write(read_memory(get0()))",
    "testing_output_move_down()",
    ")",
    "status_end()"
]
"""


"""
operators = [
"write_memory(get1(), get6())",
"write_memory(get2(), get7())",
"write_memory(get3(), get8())",
"write_memory(get4(), get9())",
"testing_output_write(read_memory(testing_get_input_value(get1(), get1())))",
"testing_output_move_right()",
"testing_output_write(read_memory(testing_get_input_value(get1(), get1())))",
"testing_output_move_right()",
"testing_output_write(read_memory(testing_get_input_value(get1(), get1())))",
"testing_output_move_down()",
"testing_output_write(read_memory(testing_get_input_value(get1(), get1())))",
"testing_output_move_down()",
"testing_output_write(read_memory(testing_get_input_value(get1(), get1())))",
"testing_output_move_left()",
"testing_output_write(read_memory(testing_get_input_value(get1(), get1())))",
"testing_output_move_left()",
"testing_output_write(read_memory(testing_get_input_value(get1(), get1())))",
"testing_output_move_up()",
"testing_output_write(read_memory(testing_get_input_value(get1(), get1())))",
"testing_output_move_right()",
"testing_output_write(read_memory(testing_get_input_value(get1(), get1())))",
"testing_set_output_position(get0(), get4())",
"testing_output_write(read_memory(testing_get_input_value(get1(), get5())))",
"testing_output_move_right()",
"testing_output_write(read_memory(testing_get_input_value(get1(), get5())))",
"testing_output_move_right()",
"testing_output_write(read_memory(testing_get_input_value(get1(), get5())))",
"testing_output_move_down()",
"testing_output_write(read_memory(testing_get_input_value(get1(), get5())))",
"testing_output_move_down()",
"testing_output_write(read_memory(testing_get_input_value(get1(), get5())))",
"testing_output_move_left()",
"testing_output_write(read_memory(testing_get_input_value(get1(), get5())))",
"testing_output_move_left()",
"testing_output_write(read_memory(testing_get_input_value(get1(), get5())))",
"testing_output_move_up()",
"testing_output_write(read_memory(testing_get_input_value(get1(), get5())))",
"testing_output_move_right()",
"testing_output_write(read_memory(testing_get_input_value(get1(), get5())))",
"testing_set_output_position(get0(), get8())",
"testing_output_write(read_memory(testing_get_input_value(get1(), get9())))",
"testing_output_move_right()",
"testing_output_write(read_memory(testing_get_input_value(get1(), get9())))",
"testing_output_move_right()",
"testing_output_write(read_memory(testing_get_input_value(get1(), get9())))",
"testing_output_move_down()",
"testing_output_write(read_memory(testing_get_input_value(get1(), get9())))",
"testing_output_move_down()",
"testing_output_write(read_memory(testing_get_input_value(get1(), get9())))",
"testing_output_move_left()",
"testing_output_write(read_memory(testing_get_input_value(get1(), get9())))",
"testing_output_move_left()",
"testing_output_write(read_memory(testing_get_input_value(get1(), get9())))",
"testing_output_move_up()",
"testing_output_write(read_memory(testing_get_input_value(get1(), get9())))",
"testing_output_move_right()",
"testing_output_write(read_memory(testing_get_input_value(get1(), get9())))",
"status_end()"
]
"""

program = getProgram(operators)
print(program)

#print(*s.runProgram(program), sep="\n")


from tqdm import tqdm


for _ in tqdm(range(100)):
#     x=s.run(["prog2(testing_output_write(testing_input_max()), comparison(bigger_than_testing_output_next(), prog2(prog2(testing_output_write_previous(testing_input_min()), comparison(bigger_than_testing_output_next(), testing_output_move_right(), testing_reset_input_position())), comparison(bigger_than_testing_output_next(), testing_input_move_right(), prog2(testing_output_write_previous(testing_input_min()), comparison(bigger_than_testing_output_next(), prog2(testing_input_move_up(), comparison(testing_is_output_down(), testing_reset_input_position(), testing_output_move_down())), testing_reset_input_position())))), prog2(testing_input_move_down(), comparison(bigger_than_testing_output_next(), testing_reset_input_down_position(), prog2(testing_output_write(testing_input_max()), comparison(bigger_than_testing_output_next(), prog2(prog2(testing_output_move_right(), comparison(bigger_thanR(testing_input_read(), get_testing_output_position_x()), testing_input_move_up(), prog2(testing_input_move_up(), comparison(testing_is_output_down(), testing_output_move_down(), testing_reset_output_down_position())))), swap_testing_output_next()), prog2(testing_reset_output_position(), comparison(bigger_than_testing_output_next(), swap_testing_output_next(), prog2(testing_output_move_right(), comparison(equalR(testing_input_read(), testing_input_max()), testing_output_move_right(), prog2(testing_output_write_previous(testing_input_max()), comparison(testing_is_output_end(), testing_input_move_up(), prog2(testing_input_move_up(), comparison(testing_is_output_down(), testing_reset_input_position(), testing_output_move_down()))))))))))))))"]*n)

     #x=s.run(["loop(get9(), loop(get9(), loop(get9(), loop(get9(), loop(get9(), loop(get9(), loop(get9(), loop(get9(), loop(get9(), loop(get9(), loop(get9(), loop(get9(), get9()))))))))))))"]*n)

    #x=s.runProgram(["testing_output_write(get9())"]*n)
     #x=s.run([program])
    x=s.run(["get9()"])
    #x=s.run(["loop(get9(),prog2(testing_output_write(get9()),testing_output_move_right()))"])
    print(x)

     #x=s.run(["comparison(equal(get9(), get9()), testing_output_write(get9()))"]*n)

     #x=s.run(["prog2(testing_output_write(get9()), testing_output_write(get9()))"]*n)

     #x=s.run(["loop(get9(),loop(get9(),testing_output_write(testing_input_max())))"]*n)

     #x=s.run(["loop(get_testing_length_input_x(), prog2(prog2(testing_input_move_right(), testing_output_move_right()),  testing_output_write(testing_input_read())))"])

     #x=s.run(["prog2(reset_output_position(), prog2(reset_output_down_position(),prog2(testing_output_move_right(),prog2(testing_output_move_right(),prog2(testing_output_move_right(),prog2(testing_output_move_right(),prog2(testing_output_move_right(),testing_output_write(get3()))))))))"])

     #x=s.run(["prog2(testing_reset_output_position(), prog2(testing_reset_output_down_position(),prog2(testing_output_move_down(),prog2(testing_output_move_down(),prog2(testing_output_move_right(),prog2(testing_output_move_right(),prog2(testing_output_move_right(),prog2(testing_output_move_right(),testing_output_write(get3())))))))))"])

     #print(x)
