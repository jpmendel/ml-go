package nn

import (
	"math"

	"../mat"
)

// ActivationFunction represents a function used to activate neural network outputs.
type ActivationFunction struct {
	Type       ActivationType
	Function   func(*mat.Matrix) *mat.Matrix
	Derivative func(*mat.Matrix) *mat.Matrix
}

// ActivationType is the identifying type of the activation function.
type ActivationType string

const (
	// ActivationTypeRELU is the type for a rectified linear unit activation function.
	ActivationTypeRELU = ActivationType("relu")

	// ActivationTypeSigmoid is the type for a sigmoid activation function.
	ActivationTypeSigmoid = ActivationType("sigmoid")

	// ActivationTypeTanh is the type for a hyperbolic tangent activation function.
	ActivationTypeTanh = ActivationType("tanh")

	// ActivationTypeSoftmax is the type for a soft max activation function.
	ActivationTypeSoftmax = ActivationType("softmax")
)

// ActivationRELU is the rectified linear unit activation function.
var ActivationRELU = ActivationFunction{
	Type: ActivationTypeRELU,
	Function: func(matrix *mat.Matrix) *mat.Matrix {
		matrix.ApplyFunction(func(current float32, row int, col int) float32 {
			if current > 0 {
				return current
			}
			return 0
		})
		return matrix
	},
	Derivative: func(matrix *mat.Matrix) *mat.Matrix {
		matrix.ApplyFunction(func(current float32, row int, col int) float32 {
			if current > 0 {
				return 1
			}
			return 0
		})
		return matrix
	},
}

// ActivationSigmoid is the sigmoid activation function.
var ActivationSigmoid = ActivationFunction{
	Type: ActivationTypeSigmoid,
	Function: func(matrix *mat.Matrix) *mat.Matrix {
		matrix.ApplyFunction(func(current float32, row int, col int) float32 {
			return 1 / (1 + float32(math.Exp(-float64(current))))
		})
		return matrix
	},
	Derivative: func(matrix *mat.Matrix) *mat.Matrix {
		matrix.ApplyFunction(func(current float32, row int, col int) float32 {
			return current * (1 - current)
		})
		return matrix
	},
}

// ActivationTanh is the hyperbolic tangent activation function.
var ActivationTanh = ActivationFunction{
	Type: ActivationTypeTanh,
	Function: func(matrix *mat.Matrix) *mat.Matrix {
		matrix.ApplyFunction(func(current float32, row int, col int) float32 {
			return float32(math.Tanh(float64(current)))
		})
		return matrix
	},
	Derivative: func(matrix *mat.Matrix) *mat.Matrix {
		matrix.ApplyFunction(func(current float32, row int, col int) float32 {
			return 1 - float32(math.Pow(float64(current), 2))
		})
		return matrix
	},
}

// ActivationSoftmax is the softmax activation function.
var ActivationSoftmax = ActivationFunction{
	Type: ActivationTypeSoftmax,
	Function: func(matrix *mat.Matrix) *mat.Matrix {
		matrix.ApplyFunction(func(current float32, row int, col int) float32 {
			return float32(math.Exp(float64(current)))
		})
		sum := matrix.Sum()
		matrix.ApplyFunction(func(current float32, row int, col int) float32 {
			return current / sum
		})
		return matrix
	},
	Derivative: func(matrix *mat.Matrix) *mat.Matrix {
		newMatrix := matrix.Copy()
		newMatrix.ApplyFunction(func(current float32, row int, col int) float32 {
			sum := float32(0.0)
			for r := 0; r < matrix.Rows; r++ {
				for c := 0; c < matrix.Cols; c++ {
					if col == c {
						sum += matrix.Get(r, c) * (1 - current)
					}
					sum += matrix.Get(r, c) * -current
				}
			}
			return sum
		})
		return newMatrix
	},
}

func activationFunctionOfType(activationType ActivationType) ActivationFunction {
	switch activationType {
	case ActivationTypeRELU:
		return ActivationRELU
	case ActivationTypeSigmoid:
		return ActivationSigmoid
	case ActivationTypeTanh:
		return ActivationTanh
	case ActivationTypeSoftmax:
		return ActivationSoftmax
	default:
		return ActivationRELU
	}
}
