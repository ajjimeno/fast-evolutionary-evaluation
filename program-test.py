import SimulatorCPU

s = SimulatorCPU.Runner("/home/antonio/git/gp/arc_training_tag/0520fde7")
#s = SimulatorCPU.Runner("/Users/ajimeno/Documents/git/gp/arc_training_tag/00d62c1b")

#program="prog2(write_memory(get3(), get8()),testing_output_write(read_memory(testing_get_output_value(get1(), get1()))))"
#program="prog2(write_memory(get3(), get8()),prog2(testing_output_write(read_memory(testing_get_output_value(get1(), get1()))), testing_output_move_right()))"


def getProgram(operators):
    if len(operators)>2:
        if operators[0].lstrip() == "comparison(":
            indentation = len(operators[0]) - len(operators[0].lstrip())

            position = 1

            comparison = operators[position]

            position+=2

            operator1 = []

            for i in range(len(operators[position:])):
                if operators[position+i].endswith(",") and len(operators[position+i]) == indentation+1:
                    break
                operator1.append(operators[i+position])

            position+=i+1

            operator2 = []
            for i in range(len(operators[position:])):
                if operators[position+i].endswith(")") and len(operators[position+i]) == indentation+1:
                    break
                operator2.append(operators[i+position])
            
            position+=i+1

            print("Program1:", operator1)
            print("Program2:", operator2)

            comparison = f"comparison({comparison},{getProgram(operator1)},{getProgram(operator2)})"

            if len(operators[position:]) > 0:
                return f"prog2({comparison}, {getProgram(operators[position:])})"
            else:
                return comparison
        elif operators[0].strip() == "loop(":
            indentation = len(operators[0]) - len(operators[0].lstrip())

            position = 1

            counter = operators[position]

            position+=2

            operator = []
            for i in range(len(operators[position:])):
                if operators[position+i].endswith(")") and len(operators[position+i]) == indentation+1:
                    break
                operator.append(operators[i+position])
            
            print("operator", operator)
            operator = getProgram(operator)

            position+=i+1

            loop = f"loop({counter}, {operator})"

            print("remaining:", operators[position:])

            if len(operators[position:]) > 0:
                return f"prog2({loop}, {getProgram(operators[position:])})"
            else:
                return loop
        return f"prog2({operators[0]},{getProgram(operators[1:])})"
    elif len(operators) == 2:
        return f"prog2({operators[0]},{operators[1]})"
    else:
        return operators[0]
    

operators = [
    "comparison(",
    "  get9()",
    ","
    "  testing_output_write(get9())",
    ","
    "  testing_output_write(get1())",
    ")",
    "status_end()"
]

operators = [
    "loop(",
    "get3()",
    ",",
    "testing_set_input_position(get4(), get_testing_output_position_y())",
    "  loop(",
    "  get3()",
    "  ,",
    "    comparison(",
    "    equalR(add(testing_output_read(), testing_input_read()), get2())",
    "    ,",
    "    testing_output_write(get2())",
    "    ,",
    "    testing_output_write(get0())",
    "    )",
    "  testing_output_move_right()",
    "  testing_input_move_right()",
    "  )",
    "testing_output_move_down()",
    "testing_input_move_down()",
    "testing_reset_output_position()",
    "testing_set_input_position(get4(), get_testing_output_position_y())",
    ")",
    "status_end()"
]



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

print(s.runProgram(program))


from tqdm import tqdm

n=300000

#for _ in tqdm(range(100)):
#     x=s.run(["prog2(testing_output_write(testing_input_max()), comparison(bigger_than_testing_output_next(), prog2(prog2(testing_output_write_previous(testing_input_min()), comparison(bigger_than_testing_output_next(), testing_output_move_right(), testing_reset_input_position())), comparison(bigger_than_testing_output_next(), testing_input_move_right(), prog2(testing_output_write_previous(testing_input_min()), comparison(bigger_than_testing_output_next(), prog2(testing_input_move_up(), comparison(testing_is_output_down(), testing_reset_input_position(), testing_output_move_down())), testing_reset_input_position())))), prog2(testing_input_move_down(), comparison(bigger_than_testing_output_next(), testing_reset_input_down_position(), prog2(testing_output_write(testing_input_max()), comparison(bigger_than_testing_output_next(), prog2(prog2(testing_output_move_right(), comparison(bigger_thanR(testing_input_read(), get_testing_output_position_x()), testing_input_move_up(), prog2(testing_input_move_up(), comparison(testing_is_output_down(), testing_output_move_down(), testing_reset_output_down_position())))), swap_testing_output_next()), prog2(testing_reset_output_position(), comparison(bigger_than_testing_output_next(), swap_testing_output_next(), prog2(testing_output_move_right(), comparison(equalR(testing_input_read(), testing_input_max()), testing_output_move_right(), prog2(testing_output_write_previous(testing_input_max()), comparison(testing_is_output_end(), testing_input_move_up(), prog2(testing_input_move_up(), comparison(testing_is_output_down(), testing_reset_input_position(), testing_output_move_down()))))))))))))))"]*n)

     #x=s.run(["loop(get9(), loop(get9(), loop(get9(), loop(get9(), loop(get9(), loop(get9(), loop(get9(), loop(get9(), loop(get9(), loop(get9(), loop(get9(), loop(get9(), get9()))))))))))))"]*n)

     #x=s.run(["get9()"])

     #x=s.run(["comparison(equal(get9(), get9()), testing_output_write(get9()))"]*n)

     #x=s.run(["prog2(testing_output_write(get9()), testing_output_write(get9()))"]*n)

     #x=s.run(["loop(get9(),loop(get9(),testing_output_write(testing_input_max())))"]*n)

     #x=s.run(["loop(get_testing_length_input_x(), prog2(prog2(testing_input_move_right(), testing_output_move_right()),  testing_output_write(testing_input_read())))"])

     #x=s.run(["prog2(reset_output_position(), prog2(reset_output_down_position(),prog2(testing_output_move_right(),prog2(testing_output_move_right(),prog2(testing_output_move_right(),prog2(testing_output_move_right(),prog2(testing_output_move_right(),testing_output_write(get3()))))))))"])

     #x=s.run(["prog2(testing_reset_output_position(), prog2(testing_reset_output_down_position(),prog2(testing_output_move_down(),prog2(testing_output_move_down(),prog2(testing_output_move_right(),prog2(testing_output_move_right(),prog2(testing_output_move_right(),prog2(testing_output_move_right(),testing_output_write(get3())))))))))"])

     #print(x)
