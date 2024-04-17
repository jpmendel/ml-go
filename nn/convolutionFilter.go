package nn

import tsr "github.com/jpmendel/ml-go/tensor"

// FilterVerticalEdges emphasizes vertical edges in data.
var FilterVerticalEdges = tsr.NewValueTensor2D([][]float32{
	{-1, 0, 1},
	{-1, 0, 1},
	{-1, 0, 1},
})

// FilterHorizontalEdges emphasizes horizontal edges in data.
var FilterHorizontalEdges = tsr.NewValueTensor2D([][]float32{
	{-1, -1, -1},
	{0, 0, 0},
	{1, 1, 1},
})
