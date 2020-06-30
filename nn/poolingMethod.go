package nn

import "../mat"

// PoolingFunction represents a function used to find a pooled value.
type PoolingFunction struct {
	Method          PoolingMethod
	FindPooledValue func(*mat.Matrix, int, int, int) float32
}

// PoolingMethod is the identifying type of the pooling function.
type PoolingMethod string

const (
	// PoolingMethodMax returns the maximum value as the pooled value.
	PoolingMethodMax = PoolingMethod("max")

	// PoolingMethodAvg returns the average value as the pooled value.
	PoolingMethodAvg = PoolingMethod("avg")
)

// PoolingMax finds the maximum value in the pool.
var PoolingMax = PoolingFunction{
	Method: PoolingMethodMax,
	FindPooledValue: func(matrix *mat.Matrix, row int, col int, poolSize int) float32 {
		max := matrix.Get(row, col)
		for or := 0; or < poolSize; or++ {
			for oc := 0; oc < poolSize; oc++ {
				if or == 0 && oc == 0 {
					continue
				}
				value := matrix.Get(row+or, col+oc)
				if value > max {
					max = value
				}
			}
		}
		return max
	},
}

// PoolingAvg finds the average value of the pool.
var PoolingAvg = PoolingFunction{
	Method: PoolingMethodAvg,
	FindPooledValue: func(matrix *mat.Matrix, poolSize int, row int, col int) float32 {
		total := float32(0.0)
		for or := 0; or < poolSize; or++ {
			for oc := 0; oc < poolSize; oc++ {
				total += matrix.Get(row+or, col+oc)
			}
		}
		return total / float32(poolSize*poolSize)
	},
}

func poolingFunctionOfMethod(poolingMethod PoolingMethod) PoolingFunction {
	switch poolingMethod {
	case PoolingMethodMax:
		return PoolingMax
	case PoolingMethodAvg:
		return PoolingAvg
	default:
		return PoolingMax
	}
}
