using System;

namespace ProgrammingTest;
class Program
{
    static void Main(string[] args)
    {
        Network net = new Network(2, 2, 1);
        double[,] input = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
        double[,] expected_output = { {1}, {1}, {1}, {0} };
        int epochs = 1000;
        for (int i = 0; i < epochs; i++)
        {
            Console.WriteLine(net.Fit(input, expected_output));   
        }
        double[,] test = { {1, 1} };
        net.Predict(test).WriteMatrix();
    }
}
