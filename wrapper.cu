#include <iostream>
#include <Python.h>
#include <vector>

#include "program.cu"

#if PY_MAJOR_VERSION >= 3
#define PY3K
#endif

/*
 *
 * Python wrappers
 *
 *
 */
typedef struct RunnerSimulatorWrapper
{
    // PyObject_HEAD
    //     Runner *mInnerClass;
    // Program *program;
    // std::unordered_map<std::string, int (Runner::*)(Program **)> map;
    // Instance *data;

    // RunnerSimulatorWrapper() : program(NULL){};
} RunnerSimulatorWrapper;

static int wrapRunnerSimulatorConstructor(RunnerSimulatorWrapper *self, PyObject *args, PyObject *kwargs)
{
    // PyObject *obj = PyTuple_GetItem(args, 0);
    // PyObject *repr = PyObject_Str(obj); // Alternatively use PyObject_Repr, but it adds single quotes
    // PyObject *str = PyUnicode_AsEncodedString(repr, "utf-8", "~E~");
    // const char *bytes = PyBytes_AS_STRING(str);

    // self->mInnerClass = new Runner();
    // self->map = getFunctionMap();

    // self->data = read_dir(bytes);

    return 0;
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
    std::vector<std::string> cpp_strings;
    if (!PyList_Check(py_list))
    {
        // Handle error: argument is not a list
        // return -1;
        std::cout << "Argument is not a list" << std::endl;
        return NULL;
    }

    Py_ssize_t len = PyList_Size(py_list);
    for (Py_ssize_t i = 0; i < len; ++i)
    {
        PyObject *py_str = PyList_GetItem(py_list, i);
        if (!PyUnicode_Check(py_str))
        {
            // Handle error: list element is not a string
            std::cout << "List element is not a string" << std::endl;
            return NULL;
        }
        const char *str = PyUnicode_AsUTF8(py_str);
        cpp_strings.push_back(std::string(str));
        // PyMem_Free(str);
    }

    int n_programs = cpp_strings.size();

    float *accuracy = (float *)malloc(n_programs * sizeof(float));

    execute_and_evaluate(n_programs, &cpp_strings[0], accuracy);

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
    PyVarObject_HEAD_INIT(NULL, 0) "Simulator.Runner" /* tp_name */
};

PyMODINIT_FUNC PyInit_SimulatorGPU()
{
    PyRunnerSimulatorWrapper_Type.tp_new = PyType_GenericNew;
    PyRunnerSimulatorWrapper_Type.tp_basicsize = sizeof(RunnerSimulatorWrapper);
    PyRunnerSimulatorWrapper_Type.tp_dealloc = (destructor)RunnerSimulatorWrapperDealloc;
    PyRunnerSimulatorWrapper_Type.tp_flags = Py_TPFLAGS_DEFAULT;
    PyRunnerSimulatorWrapper_Type.tp_doc = "GPU Simulator";
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