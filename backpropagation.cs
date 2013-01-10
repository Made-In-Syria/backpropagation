using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
namespace WindowsFormsApplication3
{


    class Layer
    {
        public int linkedNeuronsNumber;
        public int neuronsNumber;
        public double[,] linkedNeuronsWeight;
        public double[,] prevWeight;
        public double[] neuronsOutput;
        public double[] deltaMatrix;
        public Layer(int linkedNneuronsNumber, int neuronsNumber)
        {
            this.linkedNeuronsNumber = linkedNneuronsNumber + 1;
            this.neuronsNumber = neuronsNumber;
            linkedNeuronsWeight = new double[neuronsNumber, linkedNneuronsNumber + 1];
            prevWeight = new double[neuronsNumber, linkedNneuronsNumber + 1];
            neuronsOutput = new double[neuronsNumber];
            deltaMatrix = new double[neuronsNumber];
            for (int neuron = 0; neuron < neuronsNumber; neuron++)
                for (int linkedNeuron = 0; linkedNeuron < linkedNeuronsNumber; linkedNeuron++)
                    prevWeight[neuron, linkedNeuron] = 0;
        }

    }
    public class backpropagation
    {
        private int inputNeuronsNumber;
        private int outputNeuronsNumber;
        private int hiddenLayersNumber;
        private double[] desierdValues;
        private double[] inputValues;
        bool withMomentum;
        private double momentum;
        private double maxError;
        private double learningRate;
        int epoch = 0;
        int maxEpoch;
        private Layer[] Net;
        public int patternNumber;
        public double[,] inputValuesMatrix;
        public double[,] desierdValuesMatrix;
        public backpropagation(int inputNeuronsNumber, int outputNeuronsNumber, int hiddenLayersNumber, int patternNumber, double momentum, bool withMomentum, double learningRate, int[] hiddenNeuronsNumber, double MaxError, double[,] inputValuesMatrix, double[,] desierdValuesMatrix,int maxEpoch)
        {
            this.inputNeuronsNumber = inputNeuronsNumber;
            this.outputNeuronsNumber = outputNeuronsNumber;
            this.hiddenLayersNumber = hiddenLayersNumber;
            this.patternNumber = patternNumber;
            this.learningRate = learningRate;
            this.momentum = momentum;
            this.withMomentum = withMomentum;
            this.maxError = MaxError;
            this.maxEpoch = maxEpoch;
            Net = new Layer[hiddenLayersNumber + 2];
            desierdValues = new double[outputNeuronsNumber];
            inputValues = new double[inputNeuronsNumber];
            this.inputValuesMatrix = inputValuesMatrix;
            this.desierdValuesMatrix = desierdValuesMatrix;
            for (int layer = 0; layer < hiddenLayersNumber + 2; layer++)
            {
                if(layer == 0)
                    Net[layer] = new Layer(0,inputNeuronsNumber);
                else if (layer == hiddenLayersNumber +1)
                    Net[layer] = new Layer(Net[layer - 1].neuronsNumber, outputNeuronsNumber);
                else
                    Net[layer] = new Layer(Net[layer - 1].neuronsNumber, hiddenNeuronsNumber[layer - 1]);
            }
        }
        private double output(int layer, int neuron)
        {
            double val = 0;
            double linkedNeuronOutput;
            if (layer == 0)
            {
                return inputValues[neuron];
            }
            else
                for (int linkedNeuron = 0; linkedNeuron < Net[layer].linkedNeuronsNumber; linkedNeuron++)
                {
                    if (linkedNeuron == Net[layer].linkedNeuronsNumber - 1)
                        linkedNeuronOutput = 1;
                    else
                        linkedNeuronOutput = Net[layer - 1].neuronsOutput[linkedNeuron];

                    val += linkedNeuronOutput * Net[layer].linkedNeuronsWeight[neuron, linkedNeuron];
                }

            return output(val);
        }

        private double output(double input)
        {
            return (1 / (1 + Math.Exp(-input)));
        }

        private double error(int layer, int neuron)
        {
            double val = 0;
            if (layer == hiddenLayersNumber + 1)
            {
                return desierdValues[neuron] - Net[layer].neuronsOutput[neuron];
            }
            else if (layer > 0)
            {
                for (int i = 0; i < Net[layer + 1].neuronsNumber; i++)
                {
                    val += Net[layer + 1].deltaMatrix[i] * Net[layer + 1].linkedNeuronsWeight[i, neuron];
                }
            }

            return val;
        }

        private double delta(double errorValue, double neuronOutput)
        {
            return errorValue * neuronOutput * (1 - neuronOutput);
        }

        public double newWeight(int layer, int neuron, int linkedNeuron)
        {
            double oldWeight = Net[layer].linkedNeuronsWeight[neuron, linkedNeuron];
            double linkedNeuronOutput;
            if (linkedNeuron == Net[layer].linkedNeuronsNumber - 1)
                linkedNeuronOutput = 1;
            else
                linkedNeuronOutput = Net[layer - 1].neuronsOutput[linkedNeuron];

            if (withMomentum)
            {
                double weightDiff = Net[layer].linkedNeuronsWeight[neuron, linkedNeuron] - Net[layer].prevWeight[neuron, linkedNeuron];
                return oldWeight + (learningRate * linkedNeuronOutput * Net[layer].deltaMatrix[neuron]) + momentum * weightDiff;
            }
            else
                return oldWeight + (learningRate * linkedNeuronOutput * Net[layer].deltaMatrix[neuron]);
        }

        public void setOutputOfEveryNeuron()
        {
            for (int layer = 0; layer < hiddenLayersNumber + 2; layer++)
            {
                for (int neuron = 0; neuron < Net[layer].neuronsNumber; neuron++)
                {
                    Net[layer].neuronsOutput[neuron] = output(layer, neuron);
                }
            }
        }
        private void setDeltaMatrix()
        {
            for (int layer = hiddenLayersNumber + 1; layer >= 0; layer--)
            {
                for (int neuron = 0; neuron < Net[layer].neuronsNumber; neuron++)
                {
                    Net[layer].deltaMatrix[neuron] = delta(error(layer, neuron), Net[layer].neuronsOutput[neuron]);
                }
            }
        }

        double NextDouble(Random rng, double min, double max)
        {
            return min + (rng.NextDouble() * (max - min));
        }

        private void setPrimaryWeights()
        {
            Random rand = new Random();
            for (int layer = hiddenLayersNumber + 1; layer > 0; layer--)
            {
                for (int neuron = 0; neuron < Net[layer].neuronsNumber; neuron++)
                {
                    for (int linkedNeuron = 0; linkedNeuron < Net[layer].linkedNeuronsNumber; linkedNeuron++)
                    {
                        Net[layer].linkedNeuronsWeight[neuron, linkedNeuron] = NextDouble(rand, -1, 1); ;
                    }
                }
            }
        }



        public double errorValue()
        {
            double val = 0;
            for (int neuron = 0; neuron < Net[hiddenLayersNumber + 1].neuronsNumber; neuron++)
                val += (desierdValues[neuron] - Net[hiddenLayersNumber + 1].neuronsOutput[neuron]) * (desierdValues[neuron] - Net[hiddenLayersNumber + 1].neuronsOutput[neuron]);//sum( sqr(error))
            return val/2;
        }

        public void readPattern(int pattern)
        {
            for (int i = 0; i < inputNeuronsNumber; i++)
            {
                inputValues[i] = inputValuesMatrix[i, pattern];
            }
            for (int i = 0; i < outputNeuronsNumber; i++)
            {
                desierdValues[i] = desierdValuesMatrix[i, pattern];
            }
        }

        public void updateWeights(Form1 form1)
        {
            
            setPrimaryWeights();
            do
            {
                int pattern = 0;
                while (pattern < patternNumber)
                {
                    readPattern(pattern);
                    setOutputOfEveryNeuron();
                    setDeltaMatrix();
                    for (int layer = hiddenLayersNumber + 1; layer > 0; layer--)
                    {
                        for (int neuron = 0; neuron < Net[layer].neuronsNumber; neuron++)
                        {
                            for (int linkedNeuron = 0; linkedNeuron < Net[layer].linkedNeuronsNumber; linkedNeuron++)
                            {
                                Net[layer].prevWeight[neuron, linkedNeuron] = Net[layer].linkedNeuronsWeight[neuron, linkedNeuron];
                                Net[layer].linkedNeuronsWeight[neuron, linkedNeuron] = newWeight(layer, neuron, linkedNeuron);
                            }
                        }
                    }
                    pattern++;
                }
                epoch++;
            } while (errorValue() > maxError && epoch < maxEpoch);
        }
    }
}
