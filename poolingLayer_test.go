package ml

import "testing"

func TestPoolingLayer(t *testing.T) {
	layer := NewPoolingLayer(4, 4, 1, 2, PoolingMax)

	inputs := []*Matrix{
		NewMatrixWithValues([][]float32{
			{4, 2, 6, 5},
			{1, 3, 8, 7},
			{6, 9, 3, 4},
			{8, 7, 2, 5},
		}),
	}

	solution := []*Matrix{
		NewMatrixWithValues([][]float32{
			{4, 8},
			{9, 5},
		}),
	}

	pooled, err := layer.FeedForward(inputs)
	if err != nil {
		t.Fatalf("Error in FeedForward: %s", err.Error())
	}
	if len(pooled) != len(solution) {
		t.Fatalf("Pooled outputs have incorrect length: %d != %d", len(pooled), len(solution))
	}

	for i, output := range pooled {
		if !output.Equals(solution[i]) {
			t.Errorf("Matrix at index %d after feed forward should be:\n%swhen result is:\n%s", i, solution[i].String(), output.String())
		}
	}

	unpooled, err := layer.BackPropagate(pooled, 0.0, 0.0)
	if err != nil {
		t.Fatalf("Error in BackPropagate: %s", err.Error())
	}
	if len(unpooled) != len(inputs) {
		t.Fatalf("Unpooled outputs have incorrect length: %d != %d", len(unpooled), len(inputs))
	}

	for i, input := range unpooled {
		if !input.Equals(inputs[i]) {
			t.Errorf("Matrix at index %d after back propagate should be:\n%swhen result is:\n%s", i, inputs[i].String(), input.String())
		}
	}
}
