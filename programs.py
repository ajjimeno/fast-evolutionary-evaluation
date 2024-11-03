
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

            #print("Program1:", operator1)
            #print("Program2:", operator2)

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
            
            #print("operator", operator)
            operator = getProgram(operator)

            position+=i+1

            loop = f"loop({counter}, {operator})"

            #print("remaining:", operators[position:])

            if len(operators[position:]) > 0:
                return f"prog2({loop}, {getProgram(operators[position:])})"
            else:
                return loop
        return f"prog2({operators[0]},{getProgram(operators[1:])})"
    elif len(operators) == 2:
        return f"prog2({operators[0]},{operators[1]})"
    else:
        return operators[0]
  

programs = {}

programs["0520fde7"] = [
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

programs["1bfc4729"]=[
    # top number
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

programs["54d9e175"]=[
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
"status_end()"]

programs["a699fb00"] = [
    "loop(",
    "get_testing_length_output_y()",
    ",",
    "   loop(",
    "   get_testing_length_output_x()",
    "   ,",
    "       comparison(",
    "       equalR(get_testing_output_position_x(),get0())",
    "       ,",
    "       testing_input_move_right()",
    "       ,",
    "           comparison(",
    "           equalR(testing_input_read(), get1())",
    "           ,",
    "               comparison(",
    "               equalR(testing_output_read_previous(), get1())",
    "               ,",
    "               testing_output_write(get2())",
    "               ,",
    "               write_memory(get0(), get0())",
    "               )",
    "           ,",
    "           write_memory(get0(), get0())",
    "           )",
    "       )",
    "       testing_output_move_right()",
    "       testing_input_move_right()",
    "   )",
    "testing_output_move_down()",
    "testing_input_move_down()",
    "testing_reset_output_position()",
    "testing_reset_input_position()",
    ")",
    "status_end()"
]

programs["d9f24cd1"]= [
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

programs["f2829549"] = [
    "loop(",
    "get4()",
    ",",
    "testing_set_input_position(get4(), get_testing_output_position_y())",
    "  loop(",
    "  get3()",
    "  ,",
    "    comparison(",
    "    equalR(add(testing_output_read(), testing_input_read()), get0())",
    "    ,",
    "    testing_output_write(get3())",
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

programs["f76d97a5"] = [
    "write_memory(get0(), testing_get_output_value(get0(), get2()))",
    "loop(",
    "get_testing_length_output_y()",
    ",",
    "    loop(",
    "    get_testing_length_output_x()",
    "    ,",
    "        comparison(",
    "        equalR(testing_output_read(), read_memory(get0()))",
    "        ,",
    "        testing_output_write(get0())",
    "        testing_output_move_right()",
    "        ,",
    "        testing_output_write(read_memory(get0()))",
    "        testing_output_move_right()",
    "        )",
    "    )",
    "    testing_output_move_down()",
    "    testing_reset_output_position()",
    ")",
    "status_end()"
]

if __name__ == "__main__":
    for k, v in programs.items():
        print(getProgram(v))