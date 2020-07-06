package nn

import "testing"

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
