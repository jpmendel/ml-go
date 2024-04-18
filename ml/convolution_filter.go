package ml

// FilterVerticalEdges emphasizes vertical edges in data.
var FilterVerticalEdges = NewValueTensor2D([][]float32{
	{-1, 0, 1},
	{-1, 0, 1},
	{-1, 0, 1},
})

// FilterHorizontalEdges emphasizes horizontal edges in data.
var FilterHorizontalEdges = NewValueTensor2D([][]float32{
	{-1, -1, -1},
	{0, 0, 0},
	{1, 1, 1},
})
