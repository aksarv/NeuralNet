using System;

class Program
{
    static void Main(string[] args)
    {
        Network net = new Network(2, 2, 1);
        int epochs = 100000;
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double[,] inputs = new double[1, 2];
            double[,] outputs = new double[1, 1];
            Random rnd = new Random();
            for (int i = 0; i < 1; i++)
            {
                int n = rnd.Next(0, 3);
                int[] a = new int[2];
                int b = 0;
                if (n == 0)
                {
                    (a[0], a[1]) = (0, 0);
                    b = 0;
                }
                else if (n == 1)
                {
                    (a[0], a[1]) = (0, 1);
                    b = 1;
                }
                else if (n == 2)
                {
                    (a[0], a[1]) = (1, 0);
                    b = 1;
                }
                else if (n == 3)
                {
                    (a[0], a[1]) = (1, 1);
                    b = 0;
                }
                (inputs[i, 0], inputs[i, 1]) = (a[0], a[1]);
                outputs[i, 0] = b;
            }
            double network_loss = net.Step(inputs, outputs);
            // System.Console.WriteLine(network_loss);
        }
        net.WriteNetwork();
        double[,] trial = { { 1, 1 } };
        double[,] prediction = net.Predict(trial);
        System.Console.WriteLine(prediction[0, 0]);
        net.Write2D(prediction);
        System.Console.WriteLine(prediction.Length);
        net.Write2D(net.bias_input_hidden);
        net.Write2D(net.bias_hidden_output);
    }
}
