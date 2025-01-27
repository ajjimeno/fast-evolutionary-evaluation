#include <iostream>
#include <Python.h>
#include <vector>

#include "data.cu"
#include "format_string.cpp"
#include "program.cpp"

#if PY_MAJOR_VERSION >= 3
#define PY3K
#endif

#ifndef SETUP_BUILDING_CPU
#define SETUP_BUILDING_CPU
#endif

/*
 *
 * Python wrappers
 *
 *
 */
typedef struct RunnerSimulatorWrapper
{
    PyObject_HEAD Instances *data;

    RunnerSimulatorWrapper() : data(nullptr) {}
} RunnerSimulatorWrapper;

static int wrapRunnerSimulatorConstructor(RunnerSimulatorWrapper *self, PyObject *args, PyObject *kwargs)
{
    PyObject *obj = PyTuple_GetItem(args, 0);
    PyObject *repr = PyObject_Str(obj); // Alternatively use PyObject_Repr, but it adds single quotes
    PyObject *str = PyUnicode_AsEncodedString(repr, "utf-8", "~E~");
    const char *bytes = PyBytes_AS_STRING(str);

    self->data = load_data(bytes);

    return 0;
}

std::vector<uint8_t> bytes_to_vector(const char *bytes, size_t length)
{
    return std::vector<uint8_t>(bytes, bytes + length);
}

inline void string2program(int start_index, int end_index, STRING **programs, std::vector<std::vector<signed char>> *byte_buffers, MAP_TREENODE *nmap)
{
    for (int i = start_index; i < end_index; i++)
    {
        std::string s = toString(byte_buffers->at(i), nmap);

        char *str = s.data();
        char *copy = new char[strlen(str) + 1];
        strcpy(copy, str);
        programs[i] = new STRING(copy);
    }
}

static PyObject *wrapRun(RunnerSimulatorWrapper *self, PyObject *args)
{
    PyObject *py_list;
    if (!PyArg_ParseTuple(args, "O", &py_list))
    {
        std::cout << "Error parsing arguments" << std::endl;
        return NULL;
    }

    // Convert Python list to C++ vector of strings
    std::vector<STRING *> cpp_strings;
    std::vector<std::vector<signed char>> byte_buffers;
    if (!PyList_Check(py_list))
    {
        std::cout << "Argument is not a list" << std::endl;
        return NULL;
    }

    MAP_TREENODE nmap;
    std::cout << "Collecting programs" << std::endl;
    Py_ssize_t len = PyList_Size(py_list);
    for (Py_ssize_t i = 0; i < len; ++i)
    {
        PyObject *py_item = PyList_GetItem(py_list, i);

        if (PyObject_HasAttrString(py_item, "object_ids")) // "__str__"
        {
            if (nmap.size() == 0)
            {
                std::unordered_map<int, STRING> node_name;

                PyObject *pDict = PyObject_GetAttrString(py_item, "object_ids");

                PyObject *key, *value;
                Py_ssize_t pos = 0;
                while (PyDict_Next(pDict, &pos, &key, &value))
                {
                    const char *key_str = PyUnicode_AsUTF8(key);
                    int value_int = PyLong_AsLong(value);

                    node_name[value_int] = STRING(key_str);
                }

                Py_DECREF(pDict);

                pDict = PyObject_GetAttrString(py_item, "ids_objects");
                pos = 0;
                while (PyDict_Next(pDict, &pos, &key, &value))
                {
                    int key_int = PyLong_AsLong(key);
                    PyObject *attribute_name = PyUnicode_FromString("arity");
                    PyObject *attribute_value = PyObject_GetAttr(value, attribute_name);
                    Py_DECREF(attribute_name);

                    nmap.emplace(key_int, TreeNode{(int)PyLong_AsLong(attribute_value), node_name[key_int]});
                }

                Py_DECREF(pDict);
            }

            Py_buffer view;
            PyObject_GetBuffer(py_item, &view, PyBUF_SIMPLE);

            signed char *buffer_address = (signed char *)view.buf;
            Py_ssize_t buffer_length = view.len;

            byte_buffers.push_back(std::vector<signed char>(buffer_address, buffer_address + buffer_length));

            PyBuffer_Release(&view);
        }
        else if (PyUnicode_Check(py_item))
        {
            const char *str = PyUnicode_AsUTF8(py_item);
            char *copy = new char[strlen(str) + 1]; // Allocate memory
            strcpy(copy, str);
            STRING *string = new STRING(copy);
            cpp_strings.push_back(string);
            // PyMem_Free(str);
        }
        else
        {
            // Handle error: list element is not a string
            std::cout << "List element is not a string" << std::endl;
            return NULL;
        }
    }

    std::cout << "Programs collected" << std::endl;

    float *accuracy = NULL;
    int n_programs = 0;

    if (cpp_strings.size() > 0)
    {
        n_programs = cpp_strings.size();
        accuracy = (float *)malloc(n_programs * sizeof(float));
        execute_and_evaluate(n_programs, &cpp_strings[0], accuracy, self->data);
    }
    else if (byte_buffers.size() > 0)
    {
        n_programs = byte_buffers.size();

        STRING **programs = new STRING *[n_programs];

        int n_threads = std::min(n_programs, 20);
        int chunk_size = n_programs / n_threads;

        std::vector<std::thread> threads;

        std::cout << "Starting program writing" << std::endl;

        for (int i = 0; i < n_threads; ++i)
        {
            int start_index = i * chunk_size;
            int end_index = (i == n_threads - 1) ? n_programs : (i + 1) * chunk_size;

            threads.emplace_back(string2program, start_index, end_index, programs, &byte_buffers, &nmap);
        }

        for (auto &t : threads)
        {
            t.join();
        }

        std::cout << n_programs << " programs written" << std::endl;
        // std::cout << programs[0][0] << std::endl;

        std::cout << "Ending program writing" << std::endl;
        accuracy = (float *)malloc(n_programs * sizeof(float));

        execute_and_evaluate(n_programs, &programs[0], accuracy, self->data);

        for (int i = 0; i < n_programs; i++)
        {
            delete programs[i]->data();
            delete programs[i];
        }
    }

    for (auto &s : cpp_strings)
    {
        delete s->data();
        delete s;
    }

    // for (auto &b : byte_buffers)
    //{
    //     delete b;
    // }

    PyObject *list = PyList_New(n_programs);
    if (!list)
    {
        return NULL;
    }

    for (int i = 0; i < n_programs; ++i)
    {
        PyObject *item = PyFloat_FromDouble(accuracy[i]);
        if (!item)
        {
            Py_DECREF(list);
            return NULL;
        }
        PyList_SetItem(list, i, item);
    }

    free(accuracy);

    return list;
}

static PyObject *wrapRunProgram(RunnerSimulatorWrapper *self, PyObject *args)
{

    PyObject *obj = PyTuple_GetItem(args, 0);
    PyObject *repr = PyObject_Str(obj); // Alternatively use PyObject_Repr, but it adds single quotes
    PyObject *str = PyUnicode_AsEncodedString(repr, "utf-8", "~E~");
    const STRING code(PyBytes_AS_STRING(str));

    int myArray[MAX_OUTPUT_SIZE][MAX_OUTPUT_SIZE];
    int *output[MAX_OUTPUT_SIZE];
    for (int i = 0; i < MAX_OUTPUT_SIZE; i++)
    {
        output[i] = myArray[i];
    }

    MAP_INSTRUCTIONS map = get_map();

    std::vector<Node> subnodes;
    getProgram(&code, map, &subnodes);

    Node program[10000];

    for (unsigned long i = 0; i < subnodes.size(); i++)
    {
        program[i] = subnodes[i];
    }

    std::cout << "acc: " << run_problem(program, self->data, 0, output) << std::endl;

    // Return the output as a Python list

    Py_XDECREF(repr);
    Py_XDECREF(str);

    Py_INCREF(Py_None);

    // Create a Python list object
    PyObject *py_list = PyList_New(0);
    if (!py_list)
    {
        return nullptr; // Handle potential errors
    }

    for (int i = 0; i < self->data->instances[0].initial.y; i++)
    {
        PyObject *py_inner_list = PyList_New(self->data->instances[0].initial.x);
        if (!py_inner_list)
        {
            Py_DECREF(py_list);
            return nullptr;
        }

        for (int j = 0; j < self->data->instances[0].initial.x; j++)
        {
            PyObject *py_int = PyLong_FromLong(output[i][j]);
            if (!py_int)
            {
                Py_DECREF(py_list);
                Py_DECREF(py_inner_list);
                return nullptr;
            }
            PyList_SET_ITEM(py_inner_list, j, py_int);
        }

        PyList_Append(py_list, py_inner_list);
    }

    return py_list;

    // return Py_None;
}

// Getters and setters (here only for the 'eaten' attribute)
static PyGetSetDef RunnerSimulatorWrapper_getsets[] = {
    /*{
        (char *)"max_moves",      // attribute name
        (getter)wrapGetNMaxMoves, // C function to get the attribute
        NULL,                     // C function to set the attribute
        NULL,                     // optional doc string
        NULL                      // optional additional data for getter and setter
    },*/
    {NULL, NULL, NULL, NULL, NULL}};

// Class method declarations
static PyMethodDef RunnerSimulatorWrapper_methods[] = {
    {(char *)"run", (PyCFunction)wrapRun, METH_VARARGS, NULL},
    {(char *)"runProgram", (PyCFunction)wrapRunProgram, METH_VARARGS, NULL},
    //{(char *)"compile", (PyCFunction)wrapCompile, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}};

static PyModuleDef RunnerSimulatorWrapper_module = {
    PyModuleDef_HEAD_INIT,
    "RunnerSimulatorWrapper",              // Module name to use with Python import statements
    "Provides some functions, but faster", // Module description
    0,
    RunnerSimulatorWrapper_methods // Structure that defines the methods of the module
};

static void RunnerSimulatorWrapperDealloc(RunnerSimulatorWrapper *self)
{
    // delete self->mInnerClass;
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyTypeObject PyRunnerSimulatorWrapper_Type = {
    PyVarObject_HEAD_INIT(NULL, 0) "SimulatorCPU.Runner" /* tp_name */
};

PyMODINIT_FUNC PyInit_SimulatorCPU()
{
    PyRunnerSimulatorWrapper_Type.tp_new = PyType_GenericNew;
    PyRunnerSimulatorWrapper_Type.tp_basicsize = sizeof(RunnerSimulatorWrapper);
    PyRunnerSimulatorWrapper_Type.tp_dealloc = (destructor)RunnerSimulatorWrapperDealloc;
    PyRunnerSimulatorWrapper_Type.tp_flags = Py_TPFLAGS_DEFAULT;
    PyRunnerSimulatorWrapper_Type.tp_doc = "CPU Simulator";
    PyRunnerSimulatorWrapper_Type.tp_methods = RunnerSimulatorWrapper_methods;
    PyRunnerSimulatorWrapper_Type.tp_getset = RunnerSimulatorWrapper_getsets;

    PyRunnerSimulatorWrapper_Type.tp_init = (initproc)wrapRunnerSimulatorConstructor;

    PyObject *m = PyModule_Create(&RunnerSimulatorWrapper_module);
    if (m == NULL)
    {
        return NULL;
    }
    if (PyType_Ready(&PyRunnerSimulatorWrapper_Type) < 0)
    {
        return NULL;
    }
    PyModule_AddObject(m, (char *)"Runner", (PyObject *)&PyRunnerSimulatorWrapper_Type);
    return m;
}