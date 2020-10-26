package main

import (
	"fmt"
	"log"

	"gonum.org/v1/gonum/mat"
)

func main() {
	model, err := NewNetwork(2, 1, 2)
	check(err)

	numbers := []float64{
		0, 0,
		1, 1,
		1, 0,
		0, 1,
	}

	answers := []float64{
		0,
		0,
		1,
		1,
	}

	input := mat.NewDense(4, 2, numbers)
	ans := mat.NewDense(4, 1, answers)

	model.Train(input, ans, 0.5, 100000)

	first := []float64{
		0, 0,
	}
	second := []float64{
		1, 1,
	}
	third := []float64{
		1, 0,
	}
	fourth := []float64{
		0, 1,
	}

	t1 := mat.NewDense(1, 2, first)
	t2 := mat.NewDense(1, 2, second)
	t3 := mat.NewDense(1, 2, third)
	t4 := mat.NewDense(1, 2, fourth)

	fmt.Print("outputs:")
	Print(model.Predict(t1))
	Print(model.Predict(t2))
	Print(model.Predict(t3))
	Print(model.Predict(t4))
}

func check(e error) {
	if e != nil {
		log.Fatal(e)
	}
}
