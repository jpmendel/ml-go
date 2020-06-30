package nn

import "../mat"

// FilterVerticalEdges emphasizes vertical edges in data.
var FilterVerticalEdges = mat.NewMatrixWithValues([][]float32{
	{-1, 0, 1},
	{-1, 0, 1},
	{-1, 0, 1},
})

// FilterHorizontalEdges emphasizes horizontal edges in data.
var FilterHorizontalEdges = mat.NewMatrixWithValues([][]float32{
	{-1, -1, -1},
	{0, 0, 0},
	{1, 1, 1},
})
