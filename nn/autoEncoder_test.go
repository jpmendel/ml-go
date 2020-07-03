package nn

import "testing"

func TestAutoEncoderCopy(t *testing.T) {
	autoEncoder := NewAutoEncoder(4)
	autoEncoder.AddCodingLayer(2, ActivationSigmoid)

	shallow := autoEncoder
	deep := autoEncoder.Copy()

	autoEncoder.AddDecodingLayer(ActivationSoftmax)

	shallowLayers := shallow.LayerCount()
	if shallowLayers != 2 {
		t.Errorf("Shallow copy layer count shoud be 2, is: %d", shallowLayers)
	}

	deepLayers := deep.LayerCount()
	if deepLayers != 1 {
		t.Errorf("Deep copy layer count should be 1, is: %d", deepLayers)
	}
}

func TestAutoEncoderAddGetLayers(t *testing.T) {
	autoEncoder := NewAutoEncoder(4)
	autoEncoder.AddCodingLayer(2, ActivationSigmoid)
	autoEncoder.AddDecodingLayer(ActivationSoftmax)

	err := autoEncoder.AddCodingLayer(2, ActivationSigmoid)
	if err == nil {
		t.Errorf("Did not trigger error on coding layer after decoding layer")
	}

	err = autoEncoder.AddDecodingLayer(ActivationSoftmax)
	if err == nil {
		t.Errorf("Did not trigger error on multiple decoding layers")
	}

	firstLayer := autoEncoder.LayerAt(0)
	if firstLayer.OutputShape().Cols != 2 {
		t.Errorf("Incorrect first layer")
	}

	outOfBoundsLayer := autoEncoder.LayerAt(2)
	if outOfBoundsLayer != nil {
		t.Errorf("Found layer at index where one does not exist")
	}
}
