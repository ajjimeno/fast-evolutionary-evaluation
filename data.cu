#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <dirent.h>
#include <sys/types.h>
using namespace std;

#include "types.cuh"

// https://stackoverflow.com/questions/14265581/parse-split-a-string-in-c-using-string-delimiter-standard-c
vector<string> split(const string s, const string delimiter)
{
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    string token;
    vector<string> res;

    while ((pos_end = s.find(delimiter, pos_start)) != string::npos)
    {
        token = s.substr(pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back(token);
    }

    res.push_back(s.substr(pos_start));
    return res;
}

Instance read_file(string file_name)
{
    std::cout << "File:" << file_name << std::endl;
    string line;
    ifstream myfile(file_name);

    Instance ins;

    if (myfile.is_open())
    {
        // Training size
        getline(myfile, line);

        int t_examples = stoi(line);

        Array **training = (Array **)malloc(t_examples * sizeof(Array *));
        for (int i = 0; i < t_examples; i++)
        {
            vector<Array> training_example;

            for (int j = 0; j < 2; j++)
            {
                getline(myfile, line);
                vector<string> s = split(line, " ");

                Array a = {
                    stoi(s.at(0)), // y
                    stoi(s.at(1)), // x
                    new int *[stoi(s.at(0))]};

                for (int k = 0; k < a.y; k++)
                {
                    a.array[k] = new int[a.x];
                    getline(myfile, line);

                    vector<string> l = split(line, " ");

                    for (long unsigned int x = 0; x < l.size(); x++)
                    {
                        a.array[k][x] = stoi(l.at(x));
                    }
                }

                training_example.push_back(a);
            }

            training[i] = (Array *)malloc(training_example.size() * sizeof(Array));

            std::copy(training_example.begin(), training_example.end(), training[i]);
        }

        // Copy training
        ins.n_training = t_examples;
        ins.training = training;

        // Test input
        {
            getline(myfile, line);
            vector<string> s = split(line, " ");

            ins.input.y = stoi(s.at(0));
            ins.input.x = stoi(s.at(1));
            ins.input.array = new int *[ins.input.y];

            for (int k = 0; k < ins.input.y; k++)
            {
                ins.input.array[k] = new int[ins.input.x];

                getline(myfile, line);
                vector<string> l = split(line, " ");

                for (long unsigned int x = 0; x < l.size(); x++)
                {
                    ins.input.array[k][x] = stoi(l.at(x));
                }
            }
        }

        // Test output
        {
            getline(myfile, line);
            vector<string> s = split(line, " ");

            ins.output.y = stoi(s.at(0));
            ins.output.x = stoi(s.at(1));
            ins.output.array = new int *[ins.output.y];

            ins.gt.y = ins.output.y;
            ins.gt.x = ins.output.x;
            ins.gt.array = new int *[ins.gt.y];

            for (int k = 0; k < ins.output.y; k++)
            {
                ins.output.array[k] = new int[ins.output.x];
                ins.gt.array[k] = new int[ins.output.x];

                getline(myfile, line);
                vector<string> l = split(line, " ");

                for (long unsigned int x = 0; x < l.size(); x++)
                {
                    ins.output.array[k][x] = stoi(l.at(x));
                    ins.gt.array[k][x] = stoi(l.at(x));
                }
            }
        }

        {
            getline(myfile, line);
            vector<string> s = split(line, " ");

            ins.initial.y = stoi(s.at(0));
            ins.initial.x = stoi(s.at(1));
            ins.initial.array = new int *[ins.initial.y];

            for (int k = 0; k < ins.initial.y; k++)
            {
                ins.initial.array[k] = new int[ins.initial.x];

                getline(myfile, line);
                vector<string> l = split(line, " ");

                for (long unsigned int x = 0; x < l.size(); x++)
                {
                    ins.initial.array[k][x] = stoi(l.at(x));
                }
            }
        }

        myfile.close();
    }
    else
        cout << "Unable to open file";

    return ins;
}

Shape *read_shapes(string file_name, Instance *instance)
{
    std::cout << "File:" << file_name << std::endl;
    string line;
    ifstream myfile(file_name);

    if (myfile.is_open())
    {
        // Training size
        getline(myfile, line);

        int t_shapes = stoi(line);

        instance->n_shapes = t_shapes;

        Shape *shapes = (Shape *)malloc(t_shapes * sizeof(Shape));

        for (int i = 0; i < t_shapes; i++)
        {
            getline(myfile, line);

            shapes[i].type = line[0];

            getline(myfile, line);

            vector<string> l = split(line, " ");

            shapes[i].x = stoi(l[0]);
            shapes[i].y = stoi(l[1]);

            getline(myfile, line);

            shapes[i].area = stof(line);

            getline(myfile, line);

            l = split(line, " ");

            shapes[i].box.left = stoi(l[0]);
            shapes[i].box.top = stoi(l[1]);
            shapes[i].box.width = stoi(l[2]);
            shapes[i].box.height = stoi(l[3]);
        }

        return shapes;
    }
    else
    {
        instance->n_shapes = 0;
        cout << "Unable to open file";
    }

    return NULL;
}

Instances *read_dir(const char *path)
{
    Instances *output;

    vector<Instance> ins;

    struct dirent *entry;
    DIR *dir = opendir(path);
    if (dir == NULL)
        return NULL;

    string s = path;
    s += "/";

    while ((entry = readdir(dir)) != NULL)
    {
        string st(entry->d_name);
        if (entry->d_name[0] != '.' && st.find(".txt") == st.length() - 4)
        {
            Instance instance = read_file(s + entry->d_name);

            instance.shapes = read_shapes(s + st + ".s", &instance);

            ins.push_back(instance);
        }
    }

    closedir(dir);

    output = (Instances *)malloc(sizeof(Instances));
    output->n_instances = ins.size();
    output->instances = (Instance *)malloc(ins.size() * sizeof(Instance));

    std::copy(ins.begin(), ins.end(), output->instances);

    return output;
}

int **push_array(int **array, int y, int x)
{
    int **output;

    // cudaMallocManaged(&output, y * sizeof(int *));
    allocate_memory((void **)&output, y * sizeof(int *));

    for (int i = 0; i < y; i++)
    {
        // cudaMallocManaged(&output[i], x * sizeof(int));
        allocate_memory((void **)&output[i], x * sizeof(int));

        for (int j = 0; j < x; j++)
        {
            output[i][j] = array[i][j];
        }
    }

    return output;
}

Instances *load_data(const char *dir)
{
    Instances *instances = read_dir(dir);

    Instances *output;

    // cudaMallocManaged(&output, sizeof(Instances));
    allocate_memory((void **)&output, sizeof(Instances));

    output->n_instances = instances->n_instances;

    // cudaMallocManaged(&output->instances, instances->n_instances * sizeof(Instance));
    allocate_memory((void **)&output->instances, instances->n_instances * sizeof(Instance));

    for (int i = 0; i < instances->n_instances; i++)
    {
        output->instances[i].n_shapes = instances->instances[i].n_shapes;


        allocate_memory((void **)&output->instances[i].shapes, instances->instances[i].n_shapes*sizeof(Shape));

        for (int j = 0; j < instances->instances[i].n_shapes; j++)
        { 
            output->instances[i].shapes[j].area = instances->instances[i].shapes[j].area;
            output->instances[i].shapes[j].x = instances->instances[i].shapes[j].x;
            output->instances[i].shapes[j].y = instances->instances[i].shapes[j].y;
            output->instances[i].shapes[j].type = instances->instances[i].shapes[j].type;
            output->instances[i].shapes[j].box.height = instances->instances[i].shapes[j].box.height;
            output->instances[i].shapes[j].box.width = instances->instances[i].shapes[j].box.width;
            output->instances[i].shapes[j].box.left = instances->instances[i].shapes[j].box.left;
            output->instances[i].shapes[j].box.top = instances->instances[i].shapes[j].box.top;
        }

        output->instances[i].n_training = instances->instances[i].n_training;

        output->instances[i].input.x = instances->instances[i].input.x;
        output->instances[i].input.y = instances->instances[i].input.y;

        output->instances[i].input.array =
            push_array(instances->instances[i].input.array,
                       instances->instances[i].input.y,
                       instances->instances[i].input.x);

        output->instances[i].output.x = instances->instances[i].output.x;
        output->instances[i].output.y = instances->instances[i].output.y;

        output->instances[i].output.array =
            push_array(instances->instances[i].output.array,
                       instances->instances[i].output.y,
                       instances->instances[i].output.x);

        output->instances[i].gt.x = instances->instances[i].gt.x;
        output->instances[i].gt.y = instances->instances[i].gt.y;

        output->instances[i].gt.array =
            push_array(instances->instances[i].gt.array,
                       instances->instances[i].gt.y,
                       instances->instances[i].gt.x);

        output->instances[i].initial.x = instances->instances[i].initial.x;
        output->instances[i].initial.y = instances->instances[i].initial.y;

        output->instances[i].initial.array =
            push_array(instances->instances[i].initial.array,
                       instances->instances[i].initial.y,
                       instances->instances[i].initial.x);

        // cudaMallocManaged(&output->instances[i].training, output->instances[i].n_training * sizeof(Array *));
        allocate_memory((void **)&output->instances[i].training, output->instances[i].n_training * sizeof(Array *));

        for (int j = 0; j < output->instances[i].n_training; j++)
        {
            // cudaMallocManaged(&output->instances[i].training[j], 2*sizeof(Array));
            allocate_memory((void **)&output->instances[i].training[j], 2 * sizeof(Array));

            for (int k = 0; k < 2; k++)
            {
                output->instances[i].training[j][k].x = instances->instances[i].training[j][k].x;
                output->instances[i].training[j][k].y = instances->instances[i].training[j][k].y;

                output->instances[i].training[j][k].array =
                    push_array(instances->instances[i].training[j][k].array,
                               instances->instances[i].training[j][k].y,
                               instances->instances[i].training[j][k].x);
            }
        }
    }

    return output;
}

int main()
{
    load_data("../data/experiments/count/training");
}
