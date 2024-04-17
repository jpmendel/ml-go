package nn

import (
	"testing"

	tsr "github.com/jpmendel/ml-go/tensor"
)

func TestPoolingLayer(t *testing.T) {
	layer := NewPoolingLayer(4, 4, 1, 2, PoolingMax)

	inputs := tsr.NewValueTensor2D([][]float32{
		{4, 2, 6, 5},
		{1, 3, 8, 7},
		{6, 9, 3, 4},
		{8, 7, 2, 5},
	})

	solution := tsr.NewValueTensor2D([][]float32{
		{4, 8},
		{9, 5},
	})

	pooled, err := layer.FeedForward(inputs)
	if err != nil {
		t.Fatalf("Error in FeedForward: %s", err.Error())
	}
	if pooled.Frames != solution.Frames {
		t.Fatalf("Pooled outputs have incorrect frame length: %d != %d", pooled.Frames, solution.Frames)
	}

	if !pooled.Equals(solution) {
		t.Errorf("Matrix after feed forward should be:\n%swhen result is:\n%s", solution.String(), pooled.String())
	}

	unpooled, err := layer.BackPropagate(pooled, 0.0, 0.0)
	if err != nil {
		t.Fatalf("Error in BackPropagate: %s", err.Error())
	}
	if unpooled.Frames != inputs.Frames {
		t.Fatalf("Unpooled outputs have incorrect frame length: %d != %d", unpooled.Frames, inputs.Frames)
	}

	if !unpooled.Equals(inputs) {
		t.Errorf("Matrix after back propagate should be:\n%swhen result is:\n%s", inputs.String(), unpooled.String())
	}
}
