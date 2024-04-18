package ml

import (
	"math/rand"
	"testing"
)

func TestAutoEncoderCopy(t *testing.T) {
	autoEncoder := NewAutoEncoder(4)
	autoEncoder.AddCodingLayer(3, ActivationSigmoid)

	shallow := autoEncoder
	deep := autoEncoder.Copy()

	autoEncoder.AddCodingLayer(2, ActivationSigmoid)

	shallowLayers := shallow.LayerCount()
	if shallowLayers != 4 {
		t.Errorf("Shallow copy layer count shoud be 2, is: %d", shallowLayers)
	}

	deepLayers := deep.LayerCount()
	if deepLayers != 2 {
		t.Errorf("Deep copy layer count should be 1, is: %d", deepLayers)
	}
}

func TestAutoEncoderAddGetLayers(t *testing.T) {
	autoEncoder := NewAutoEncoder(4)
	autoEncoder.AddCodingLayer(2, ActivationSigmoid)

	firstLayer := autoEncoder.LayerAt(0)
	if firstLayer.OutputShape().Cols != 2 {
		t.Errorf("Incorrect first layer")
	}

	outOfBoundsLayer := autoEncoder.LayerAt(2)
	if outOfBoundsLayer != nil {
		t.Errorf("Found layer at index where one does not exist")
	}
}

func TestAutoEncoderEncodeDecode(t *testing.T) {
	inputs := [][]float32{
		{0.0, 0.0, 1.0, 1.0},
		{0.0, 1.0, 1.0, 0.0},
	}

	autoEncoder := NewAutoEncoder(4)
	autoEncoder.AddCodingLayer(2, ActivationSigmoid)

	for i := 0; i < 10000; i++ {
		index := rand.Intn(len(inputs))
		data := inputs[index]
		err := autoEncoder.Train(data, 0.3, 0.2)
		if err != nil {
			t.Fatalf("Error in Train: %s", err.Error())
		}
	}
	for i, input := range inputs {
		encoded, err := autoEncoder.Encode(input)
		if err != nil {
			t.Fatalf("Error in Encode: %s", err.Error())
		}

		decoded, err := autoEncoder.Decode(encoded)
		if err != nil {
			t.Fatalf("Error in Decode: %s", err.Error())
		}

		for j := 0; j < len(decoded); j++ {
			low := input[j] - 0.1
			high := input[j] + 0.1
			if decoded[j] < low || decoded[j] > high {
				t.Errorf("Incorrect decode for input %d at index %d: %.3f", i, j, decoded[j])
			}
		}
	}
}
