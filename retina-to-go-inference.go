package main

import (
	"fmt"
	_ "image/png"
	"io/ioutil"
	"log"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

func main() {
	// Load a frozen graph to use for queries
	modelpath := "retinago.pb"
	model, err := ioutil.ReadFile(modelpath)
	if err != nil {
		log.Fatal(err)
	}

	// Construct an in-memory graph from the serialized form.
	graph := tf.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		log.Fatal(err)
	}

	// Create a session for inference over graph.
	session, err := tf.NewSession(graph, nil)

	if err != nil {
		log.Fatal(err)
	}

	defer session.Close()
	tensor2, _ := makeTensorFromImage("205_right.png")

	final, err := session.Run(
		map[tf.Output]*tf.Tensor{
			graph.Operation("densenet121_input").Output(0): tensor2,
		},
		[]tf.Output{
			graph.Operation("outputLayer/Sigmoid").Output(0),
		},
		nil)
	if err != nil {
		log.Fatal(err)
		fmt.Println("failed here")
	}
	//fmt.Println("ok===================================================")
	fmt.Printf("Result value: %v \n", final[0].Value().([][]float32)[0])

}

func makeTensorFromImage(filename string) (*tf.Tensor, error) {
	bytes, err := ioutil.ReadFile(filename)
	if err != nil {
		fmt.Println("prob1-------------------------------------------------------------------------")
	}

	tensor, err := tf.NewTensor(string(bytes))
	if err != nil {
		fmt.Println("prob2")
	}

	// Construct a graph to normalize the image
	graph2, input2, output2, err := constructGraphToNormalizeImage()
	if err != nil {
		fmt.Println("prob3")
	}

	// Execute that graph to normalize this one image

	session, err := tf.NewSession(graph2, nil)
	if err != nil {
		fmt.Println("prob4")
	}
	defer session.Close()
	normalized, err := session.Run(
		map[tf.Output]*tf.Tensor{input2: tensor},
		[]tf.Output{output2},
		nil)
	if err != nil {
		fmt.Println("prob5")
	}
	return normalized[0], nil
}

func constructGraphToNormalizeImage() (graph3 *tf.Graph, input3, output3 tf.Output, err error) {

	const (
		H, W  = 224, 224
		Mean  = float32(0)
		Scale = float32(1)
	)

	s := op.NewScope()
	input3 = op.Placeholder(s, tf.String)

	output3 = op.Div(s,
		op.Sub(s,
			op.ResizeBilinear(s,
				op.ExpandDims(s,
					op.Cast(s,
						op.DecodeJpeg(s, input3, op.DecodeJpegChannels(3)), tf.Float),
					op.Const(s.SubScope("make_batch"), int32(0))),
				op.Const(s.SubScope("size"), []int32{H, W})),
			op.Const(s.SubScope("mean"), Mean)),
		op.Const(s.SubScope("scale"), Scale))
	//op.Const(s.SubScope("reshape"), []int32{0, 3, 1, 2}))

	graph3, err = s.Finalize()

	return graph3, input3, output3, err
}
