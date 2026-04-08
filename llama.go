package main

import (
	"fmt"
	"time"
	"math/big"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/circuits/ckks/polynomial"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

type LlamaSize struct {
	hidDim   int
	expDim   int
	numHeads int
	seqLen   int
}

type SiluFunc struct {
	scalar   *big.Float
	constant *big.Float
	siluEval *polynomial.Evaluator
	siluPoly polynomial.PolynomialVector
}

type NormFunc struct {
	config	  LayerNormConfig
	gamma     *rlwe.Ciphertext
	beta 	  *rlwe.Ciphertext
}

type LlamaInference struct {
	params    *ckks.Parameters
	size      *LlamaSize
	eval      []*ckks.Evaluator
	btpEval   *bootstrapping.Evaluator
	siluFunc  *SiluFunc
	normFunc  *NormFunc
	helper    *TestHelper
	w         map[string][]*rlwe.Plaintext
	wMsg	  map[string][][]complex128
	cache     map[string][]*rlwe.Ciphertext
	cacheMsg  map[string][][]complex128
	mask      map[string][]*rlwe.Plaintext
}

func (llama *LlamaInference) DecoderMoai(x *rlwe.Ciphertext) (y *rlwe.Ciphertext) {
	eval := llama.eval[0]
	inLevel := x.Level()

	fmt.Print("Computing Decoder...\n")
	decoderStart := time.Now()

	q, k, v := llama.QKV(x)
	q, k = llama.RoPE(q, k)
	llama.Cache(k, v)
	s := llama.QK_T(q)
	s = llama.BootTo(s, 12)
	s = llama.Softmax(s, 14, 0)
	o := llama.AttnV(s)
	o = llama.Out(o)
	eval.Add(o, x, x)
	x = llama.BootTo(x, 15)
	x = llama.Norm(x, 9)

	up, gate := llama.UpGate(x)
	gate = llama.BootTo(gate, 9)
	gate = llama.SiLU(gate)
	y = x
	eval.MulRelin(up, gate, y)
	eval.Rescale(y, y)
	y = llama.Down(y)
	eval.Add(x, y, y)
	y = llama.BootTo(y, 15)
	y = llama.Norm(y, 11)

	decElapsed := time.Since(decoderStart)
	fmt.Print("Decoder completed!\n")
	fmt.Printf("Consumed %f seconds with input level %d and output level %d\n", decElapsed.Seconds(), inLevel, y.Level())

	return y
}

func (llama *LlamaInference) DecoderThor(x *rlwe.Ciphertext) (y *rlwe.Ciphertext) {
	eval := llama.eval[0]
	inLevel := x.Level()

	fmt.Print("Computing Decoder...\n")
	decoderStart := time.Now()

	q, k, _ := llama.QKV(x)
	llama.helper.Dec(q, 10)
	q, k = llama.RoPE(q, k)
	llama.helper.Dec(q, 10)
	// llama.Cache(k, v)
	s := llama.QK_T(q)
	s = llama.BootTo(s, 12)
	llama.helper.Dec(s, 10)
	// s = llama.Softmax(s, 14, 0)
	// llama.helper.Dec(s, 10)
	o := llama.AttnV(s)
	llama.helper.Dec(o, 10)
	o = llama.Out(o)
	llama.helper.Dec(o, 10)
	eval.Add(o, x, x)
	x = llama.BootTo(x, 13)
	x = llama.NormThor(x, 0)
	llama.helper.Dec(x, 10)

	x = llama.BootTo(x, 13)
	up, gate := llama.UpGate(x)
	llama.helper.Dec(gate, 10)
	gate = llama.SiLU(gate)
	llama.helper.Dec(gate, 10)
	y = x
	eval.MulRelin(up, gate, y)
	eval.Rescale(y, y)
	y = llama.BootTo(y, 13)
	y = llama.Down(y)
	eval.Add(x, y, y)
	llama.helper.Dec(y, 10)
	y = llama.NormThor(y, 11)
	llama.helper.Dec(y, 10)

	decElapsed := time.Since(decoderStart)
	fmt.Print("Decoder completed!\n")
	fmt.Printf("Consumed %f seconds with input level %d and output level %d\n", decElapsed.Seconds(), inLevel, y.Level())

	return y
}

// decryptThenEncrypt decrypts, applies plaintext function, then re-encrypts
func (llama *LlamaInference) decryptThenEncrypt(
	ct *rlwe.Ciphertext,
	plaintextFn func([]complex128) []complex128,
	targetLevel int,
) *rlwe.Ciphertext {
	// Decrypt
	decoded := llama.helper.Dec(ct, 0)

	// Apply plaintext function
	result := plaintextFn(decoded)

	// Re-encode and re-encrypt at target level
	pt := ckks.NewPlaintext(*llama.params, targetLevel)
	if err := llama.helper.GetEncoder().Encode(result, pt); err != nil {
		panic(err)
	}

	newCt, err := llama.helper.GetEncryptor().EncryptNew(pt)
	if err != nil {
		panic(err)
	}

	return newCt
}

// decryptThenEncryptWithSlots decrypts, applies fn, then re-encrypts for a specific dimension
func (llama *LlamaInference) decryptThenEncryptWithSlots(
	ct *rlwe.Ciphertext,
	dim, stride int,
	plaintextFn func([]complex128) []complex128,
	targetLevel int,
) *rlwe.Ciphertext {
	// Decrypt full slots
	decoded := llama.helper.Dec(ct, 0)

	// Extract actual values by stride
	values := make([]complex128, dim)
	for i := 0; i < dim; i++ {
		values[i] = decoded[i*stride]
	}

	// Apply plaintext function
	resultValues := plaintextFn(values)

	// Pack back into full slots
	slots := llama.params.MaxSlots()
	resultSlots := make([]complex128, slots)
	for i := 0; i < dim; i++ {
		resultSlots[i*stride] = resultValues[i]
	}

	// Re-encode and re-encrypt at target level
	pt := ckks.NewPlaintext(*llama.params, targetLevel)
	if err := llama.helper.GetEncoder().Encode(resultSlots, pt); err != nil {
		panic(err)
	}

	newCt, err := llama.helper.GetEncryptor().EncryptNew(pt)
	if err != nil {
		panic(err)
	}

	return newCt
}

// DecoderPlaintext runs the full decoder with non-linear operations in plaintext
func (llama *LlamaInference) DecoderPlaintext(x *rlwe.Ciphertext) (y *rlwe.Ciphertext) {
	eval := llama.eval[0]
	inLevel := x.Level()
	slots := llama.params.MaxSlots()
	hidStr := slots / llama.size.hidDim
	expStr := slots / llama.size.expDim

	fmt.Print("Computing Decoder with plaintext non-linear operations...\n")
	decoderStart := time.Now()

	// ========== Attention Part ==========
	// 1. QKV projections (linear)
	q, k, v := llama.QKV(x)
	llama.Cache(k, v)

	// 2. RoPE (linear rotations)
	q, k = llama.RoPE(q, k)

	// 3. QK^T (linear)
	s := llama.QK_T(q)

	// 4. Softmax (non-linear -> plaintext)
	s = llama.decryptThenEncrypt(s, llama.SoftmaxPlaintext, 6)

	// 5. AttnV (linear)
	o := llama.AttnV(s)

	// 6. Out projection (linear)
	o = llama.Out(o)

	// 7. Residual add
	eval.Add(o, x, o)

	// 8. Norm (non-linear -> plaintext)
	o = llama.decryptThenEncryptWithSlots(o, llama.size.hidDim, hidStr, llama.NormPlaintext, 6)

	// ========== FFN Part ==========
	// 9. UpGate projections (linear)
	up, gate := llama.UpGate(o)

	// 10. SiLU on gate (non-linear -> plaintext)
	gate = llama.decryptThenEncryptWithSlots(gate, llama.size.expDim, expStr, llama.SiLUPlaintext, 6)

	// 11. Element-wise multiply (linear)
	y, _ = eval.MulRelinNew(up, gate)
	eval.Rescale(y, y)

	// 12. Down projection (linear)
	y = llama.Down(y)

	// 13. Residual add
	eval.Add(o, y, y)

	// 14. Final Norm (non-linear -> plaintext)
	y = llama.decryptThenEncryptWithSlots(y, llama.size.hidDim, hidStr, llama.NormPlaintext, 6)

	decElapsed := time.Since(decoderStart)
	fmt.Print("Decoder completed!\n")
	fmt.Printf("Consumed %f seconds with input level %d and output level %d\n", decElapsed.Seconds(), inLevel, y.Level())

	return y
}

func (llama *LlamaInference) DecoderMsg(x []complex128) (y []complex128) {
	vecLog := func(v []complex128, n int, name ...string) {
		if len(name) > 0 {
			fmt.Printf("Vector %s: \n", name[0])
		}
		for _, x := range v[0:n] {
			fmt.Printf("%.4f ", real(x))
		}
		fmt.Println()
	}

	q := llama.LinearMsg(x, llama.wMsg["q"])
	vecLog(q, 10, "Q")
	k := llama.LinearMsg(x, llama.wMsg["k"])
	// v := llama.LinearMsg(x, llama.wMsg["v"])
	q, k = llama.RoPEMsg(q, k)
	vecLog(q, 10, "Q after RoPE")
	// llama.CacheMsg(k, v)
	s := llama.LinearMsg(q, llama.cacheMsg["k"], 2)
	vecLog(s, 10, "S after QK_T")
	// s = llama.SoftmaxPlaintext(s)
	// vecLog(s, 10)
	o := llama.AttnVMsg(s)
	vecLog(o, 10, "O after AttnV")
	o = llama.LinearMsg(o, llama.wMsg["out"])
	vecLog(o, 10, "O after Out")
	for i := 0; i < len(o); i++ {
		o[i] += x[i]
	}
	o = llama.NormPlaintext(o)
	vecLog(o, 10, "O after Norm")

	up := llama.LinearMsg(o, llama.wMsg["up"])
	gate := llama.LinearMsg(o, llama.wMsg["gate"])
	vecLog(gate, 10, "Gate before SiLU")
	gate = llama.SiLUPlaintext(gate)
	vecLog(gate, 10, "Gate after SiLU")
	y = make([]complex128, len(up))
	for i := 0; i < len(up); i++ {
		y[i] = up[i] * gate[i]
	}
	y = llama.LinearMsg(y, llama.wMsg["down"])
	for i := 0; i < len(y); i++ {
		y[i] += o[i]
	}
	vecLog(y, 10, "Y before Norm")
	y = llama.NormPlaintext(y)
	vecLog(y, 10, "Y after Norm")

	return y
}

func (llama *LlamaInference) Model(x *rlwe.Ciphertext) (y *rlwe.Ciphertext) {
	modelStart := time.Now()
	for i := 0; i < 32; i++ {
		fmt.Printf("Computing the %d-th decoder...", i)
		x = llama.DecoderMoai(x)
	}
	y = x
	modelElapsed := time.Since(modelStart)
	fmt.Print("Model completed!\n")
	fmt.Printf("Consumed %f seconds for the whole model.\n", modelElapsed.Seconds())

	return y
}
