// Not complete

class Network
{
    public Random rnd = new Random();
    public int no_input;
    public int no_hidden;
    public int no_output;
    public double[,] weights_input_hidden;
    public double[,] weights_hidden_output;
    public double[,] bias_input_hidden;
    public double[,] bias_hidden_output;
    public double learning_rate;

    public double NextGaussian(int mean=0, int stdDev=1)
    {
        double u1 = 1.0 - rnd.NextDouble();
        double u2 = 1.0 - rnd.NextDouble();
        double rndStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        return mean + stdDev * rndStdNormal;
    }

    public Network(int inp, int hid, int ou, double lr=0.01)
    {
        no_input = inp;
        no_hidden = hid;
        no_output = ou;
        weights_input_hidden = new double[inp, hid];
        weights_hidden_output = new double[hid, ou];
        for (int i = 0; i < inp; i++)
        {
            for (int j = 0; j < hid; j++)
            {
                weights_input_hidden[i, j] = NextGaussian();
            }
        }
        for (int i = 0; i < hid; i++)
        {
            for (int j = 0; j < ou; j++)
            {
                weights_hidden_output[i, j] = NextGaussian();
            }
        }
        bias_input_hidden = new double[1, hid];
        bias_hidden_output = new double[1, ou];
        double init_input_hidden_bias = NextGaussian();
        double init_hidden_output_bias = NextGaussian();
        for (int i = 0; i < hid; i++)
        {
            bias_input_hidden[0, i] = init_input_hidden_bias;
        }
        for (int i = 0; i < ou; i++)
        {
            bias_hidden_output[0, i] = init_hidden_output_bias;
        }
        learning_rate = lr;
    }

    public double Dot(double[] v1, double[] v2)
    {
        if (v1.Length != v2.Length)
        {
            throw new Exception("Dot product failed as vectors are not the same length");
        }
        double result = 0;
        for (int i = 0; i < v1.Length; i++)
        {
            result += v1[i] * v2[i];
        }
        return result;
    }

    public double[,] MatMul(double[,] m1, double[,] m2)
    {
        if (m1.GetLength(1) != m2.GetLength(0))
        {
            throw new Exception("Matrix multiplication failed as matrices are not comforable");
        }
        double[,] result = new double[m1.GetLength(0), m2.GetLength(1)];
        for (int i = 0; i < m1.GetLength(0); i++)
        {
            for (int j = 0; j < m2.GetLength(1); j++)
            {
                double[] row = new double[m1.GetLength(1)];
                for (int k = 0; k < m1.GetLength(1); k++)
                {
                    row[k] = m1[i, k];
                }
                double[] col = new double[m2.GetLength(0)];
                for (int k = 0; k < m2.GetLength(0); k++)
                {
                    col[k] = m2[k, j];
                }
                result[i, j] = Dot(row, col);
            }
        }
        return result;
    }

    public double Sigmoid(double input)
    {
        return 1.0 / (1.0 + Math.Exp(-input));
    }

    public double[,] T(double[,] a)
    {
        double[,] result = new double[a.GetLength(1), a.GetLength(0)];
        for (int i = 0; i < a.GetLength(0); i++)
        {
            for (int j = 0; j < a.GetLength(1); j++)
            {
                result[j, i] = a[i, j];
            }
        }
        return result;
    }

    public double MSELoss(double[,] output, double[,] expected)
    {
        double total_loss = 0.0;
        for (int i = 0; i < output.GetLength(0); i++)
        {
            for (int j = 0; j < output.GetLength(0); j++)
            {
                total_loss += 0.5 * Math.Pow(expected[i, j] - output[i, j], 2);
            }
        }
        return total_loss;
    }

    public double[,] ForwardInputHidden(double[,] input)
    {
        double[,] hidden_layer_output = MatMul(input, weights_input_hidden);
        for (int i = 0; i < hidden_layer_output.GetLength(0); i++)
        {
            for (int j = 0; j < hidden_layer_output.GetLength(1); j++)
            {
                hidden_layer_output[i, j] = Sigmoid(hidden_layer_output[i, j] + bias_input_hidden[i, j]);
            }
        }
        return hidden_layer_output;
    }
    
    public double[,] ForwardHiddenOutput(double[,] hidden_layer_output)
    {
        double[,] output_layer_output = MatMul(hidden_layer_output, weights_hidden_output);
        for (int i = 0; i < output_layer_output.GetLength(0); i++)
        {
            for (int j = 0; j < output_layer_output.GetLength(1); j++)
            {
                output_layer_output[i, j] = Sigmoid(output_layer_output[i, j] + bias_input_hidden[i, j]);
            }
        }
        return output_layer_output;
    }

    public double[,] Predict(double[,] input)
    {
        double[,] hidden_layer_output = ForwardInputHidden(input);
        double[,] output_layer_output = ForwardHiddenOutput(hidden_layer_output);
        return output_layer_output;
    }

    public void Step(double[,] input, double[,] expected)
    {
        double[,] hidden_layer_output = ForwardInputHidden(input);
        double[,] output_layer_output = ForwardHiddenOutput(hidden_layer_output);
        // Loss
        double loss = MSELoss(output_layer_output, expected);
        // Backward Pass
        double[,] d_loss_d_weights_hidden_output = new double[no_hidden, no_output];
        double[,] d_loss_d_inputs_hidden_output = new double[1, no_hidden];
        double[,] d_loss_d_bias_hidden_output = new double[1, no_output];
        for (int i = 0; i < no_hidden; i++)
        {
            for (int j = 0; j < no_output; j++)
            {
                double d_loss_d_output = output_layer_output[0, j] - expected[0, j];
                double d_output_d_preactivated = output_layer_output[0, j] * (1 - output_layer_output[0, j]);
                double d_preactivated_d_weight = hidden_layer_output[0, i];
                d_loss_d_weights_hidden_output[i, j] = d_loss_d_output * d_output_d_preactivated * d_preactivated_d_weight;
                d_loss_d_bias_hidden_output[0, j] = d_loss_d_output * d_output_d_preactivated;
                d_loss_d_inputs_hidden_output[0, i] += d_loss_d_output * d_output_d_preactivated * weights_hidden_output[i, j];
            }
        }
        double[,] d_loss_d_weights_inputs_hidden = new double[no_input, no_hidden];
        double[,] d_loss_d_bias_inputs_hidden = new double[1, no_hidden];
        for (int i = 0; i < no_input; i++)
        {
            for (int j = 0; j < no_hidden; j++)
            {
                double d_loss_d_output = d_loss_d_inputs_hidden_output[0, j];
                double d_output_d_preactivated = hidden_layer_output[0, j] * (1 - hidden_layer_output[0, j]);
                double d_preactivated_d_weight = input[0, i];
                d_loss_d_weights_inputs_hidden[i, j] = d_loss_d_output * d_output_d_preactivated * d_preactivated_d_weight;
                d_loss_d_bias_inputs_hidden[0, j] = d_loss_d_output * d_output_d_preactivated;
            }
        }
        // Update Weights and Biases
        for (int i = 0; i < no_hidden; i++)
        {
            for (int j = 0; j < no_output; j++)
            {
                weights_hidden_output[i, j] -= learning_rate * d_loss_d_weights_hidden_output[i, j];
            }
        }
        for (int j = 0; j < no_output; j++)
        {
            bias_hidden_output[0, j] -= learning_rate * d_loss_d_bias_hidden_output[0, j];
        }
        for (int i = 0; i < no_input; i++)
        {
            for (int j = 0; j < no_hidden; j++)
            {
                weights_input_hidden[i, j] -= learning_rate * d_loss_d_weights_inputs_hidden[i, j];
            }
        }
        for (int j = 0; j < no_hidden; j++)
        {
            bias_input_hidden[0, j] -= learning_rate * d_loss_d_weights_inputs_hidden[0, j];
        }
    }
}
