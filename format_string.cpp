#include <iostream>
#include <unordered_map>
#include <string>
#include <string_view>
#include <stack>
#include <vector>

//#define STRING std::string

const std::vector<int> ids = {0, 1, 2};

struct TreeNode
{
    int arity;
    STRING name;
    TreeNode(int arity, STRING name) : arity(arity), name(name) {}
};

#define MAP_TREENODE std::unordered_map<int, TreeNode>

MAP_TREENODE getTreeNodeMap()
{
    MAP_TREENODE nodes;

    nodes.emplace(0, TreeNode{2, "prog2"});
    nodes.emplace(1, TreeNode{0, "get9"});
    nodes.emplace(2, TreeNode{0, "get0"});

    return nodes;
}

std::string concatenate_arguments(const std::vector<std::string*>* args)
{
    std::string str = "(";
    for (int i = 0; i < args->size(); i++)
    {
        if (str.size() != 1)
        {
            str += ",";
        }

        str += *args->at(i);
    }

    return str + ")";
}

std::string format(const STRING &name, const std::vector<std::string*>* args)
{
    return std::string(name) + concatenate_arguments(args);
}

std::string toString(const std::vector<int> &nodes)
{
    std::string string;
    std::stack<std::pair<int, std::vector<std::string*>*>> stack;

    MAP_TREENODE nmap = getTreeNodeMap();

    for (const int &node : nodes)
    {
        std::vector<std::string*> * v = new std::vector<std::string*>();
        stack.push({node, v});

        while (!stack.empty() && stack.top().second->size() == nmap.at(stack.top().first).arity)
        {
            auto &[prim, args] = stack.top();
            stack.pop();

            string = format(nmap.at(prim).name, args);

            for (int i = 0; i < args->size(); i++)
            { delete args->at(i); }
            delete args;

            if (stack.empty())
            {
                break;
            }

            stack.top().second->push_back(new std::string(string));
        }
    }

    return string;
}

/*
int main()
{
    for (int i = 0; i < 1000000;i++)
        toString(ids);
    return 0;
}*/