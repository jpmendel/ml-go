package ml

import (
	"math/rand"
	"os"
	"testing"
)

type TrainingData struct {
	Inputs  [][][]float32
	Targets [][][]float32
}

func TestNeuralNetworkCopy(t *testing.T) {
	neuralNetwork := NewNeuralNetwork()
	neuralNetwork.Add(NewDenseLayer(2, 2, ActivationRELU))

	shallow := neuralNetwork
	deep := neuralNetwork.Copy()

	neuralNetwork.Add(NewDenseLayer(2, 1, ActivationRELU))

	shallowLayers := shallow.LayerCount()
	if shallowLayers != 2 {
		t.Errorf("Shallow copy layer count shoud be 2, is: %d", shallowLayers)
	}

	deepLayers := deep.LayerCount()
	if deepLayers != 1 {
		t.Errorf("Deep copy layer count should be 1, is: %d", deepLayers)
	}
}

func TestNeuralNetworkAddGetLayers(t *testing.T) {
	neuralNetwork := NewNeuralNetwork()
	neuralNetwork.Add(
		NewDenseLayer(2, 3, ActivationSigmoid),
		NewDenseLayer(3, 1, ActivationSigmoid),
	)

	err := neuralNetwork.Add(NewDenseLayer(2, 2, ActivationSigmoid))
	if err == nil {
		t.Errorf("Did not trigger error on incorrect layer shape")
	}

	firstLayer := neuralNetwork.LayerAt(0)
	if firstLayer.InputShape().Cols != 2 || firstLayer.OutputShape().Cols != 3 {
		t.Errorf("Incorrect inputs and outputs: (%d, %d) != (%d, %d)", firstLayer.InputShape().Cols, firstLayer.OutputShape().Cols, 2, 3)
	}

	outOfBoundsLayer := neuralNetwork.LayerAt(2)
	if outOfBoundsLayer != nil {
		t.Errorf("Found layer at index where one does not exist")
	}
}

func TestNeuralNetworkXOR(t *testing.T) {
	trainingData := []TrainingData{
		{
			Inputs:  [][][]float32{{{0, 0}}},
			Targets: [][][]float32{{{0}}},
		},
		{
			Inputs:  [][][]float32{{{0, 1}}},
			Targets: [][][]float32{{{1}}},
		},
		{
			Inputs:  [][][]float32{{{1, 0}}},
			Targets: [][][]float32{{{1}}},
		},
		{
			Inputs:  [][][]float32{{{1, 1}}},
			Targets: [][][]float32{{{0}}},
		},
	}

	neuralNetwork := NewNeuralNetwork()
	neuralNetwork.Add(
		NewDenseLayer(2, 2, ActivationSigmoid),
		NewDenseLayer(2, 1, ActivationSigmoid),
	)

	for i := 0; i < 100000; i++ {
		index := rand.Intn(len(trainingData))
		data := trainingData[index]
		err := neuralNetwork.Train(data.Inputs, data.Targets, 0.3, 0.5)
		if err != nil {
			t.Fatalf("Error in Train: %s", err.Error())
		}
	}

	result1, err := neuralNetwork.Predict(trainingData[0].Inputs)
	if err != nil {
		t.Fatalf("Error in Predict: %s", err.Error())
	}
	if result1[0][0][0] > 0.1 {
		t.Errorf("Incorrect prediction for [0, 0]: %.3f", result1[0][0][0])
	}

	result2, err := neuralNetwork.Predict(trainingData[1].Inputs)
	if err != nil {
		t.Fatalf("Error in Predict: %s", err.Error())
	}
	if result2[0][0][0] < 0.9 {
		t.Errorf("Incorrect prediction for [0, 1]: %.3f", result2[0][0][0])
	}

	result3, err := neuralNetwork.Predict(trainingData[2].Inputs)
	if err != nil {
		t.Fatalf("Error in Predict: %s", err.Error())
	}
	if result3[0][0][0] < 0.9 {
		t.Errorf("Incorrect prediction for [1, 0]: %.3f", result3[0][0][0])
	}

	result4, err := neuralNetwork.Predict(trainingData[3].Inputs)
	if err != nil {
		t.Fatalf("Error in Predict: %s", err.Error())
	}
	if result4[0][0][0] > 0.1 {
		t.Errorf("Incorrect prediction for [1, 1]: %.3f", result4[0][0][0])
	}
}

func TestNeuralNetworkSaveLoad(t *testing.T) {
	neuralNetwork := NewNeuralNetwork()
	conv := NewConvolutionLayer(16, 16, 1, []*Tensor{FilterVerticalEdges, FilterHorizontalEdges}, ActivationRELU)
	pool := NewPoolingLayer(conv.OutputShape().Rows, conv.OutputShape().Cols, conv.OutputShape().Frames, 2, PoolingMax)
	flat := NewFlattenLayer(pool.OutputShape().Rows, pool.OutputShape().Cols, pool.OutputShape().Frames)
	dense1 := NewDenseLayer(flat.OutputShape().Cols, 16, ActivationRELU)
	dense2 := NewDenseLayer(dense1.OutputShape().Cols, 8, ActivationSoftmax)
	neuralNetwork.Add(conv, pool, flat, dense1, dense2)

	err := neuralNetwork.SaveToFile("neuralNetwork.json")
	if err != nil {
		t.Fatalf("Error in SaveNeuralNetwork: %s", err.Error())
	}

	loadedNeuralNetwork := NewNeuralNetwork()
	err = loadedNeuralNetwork.LoadFromFile("neuralNetwork.json")
	if err != nil {
		t.Fatalf("Error in LoadNeuralNetwork: %s", err.Error())
	}

	if loadedNeuralNetwork.LayerCount() != neuralNetwork.LayerCount() {
		t.Errorf("Loaded neural network layers do not match original: %d != %d", loadedNeuralNetwork.LayerCount(), neuralNetwork.LayerCount())
	}

	err = os.Remove("neuralNetwork.json")
	if err != nil {
		t.Fatalf("Error removing test file: %s", err.Error())
	}
}
