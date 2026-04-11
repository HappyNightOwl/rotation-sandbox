package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"testing"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

const stage1Seed int64 = 42

var (
	Stage1SeqLen   = 29
	Stage1HidDim   = 32
	Stage1ExpDim   = 64
	Stage1NumHeads = 8
)

const defaultMSETol = 1e-5

type stage1Context struct {
	params ckks.Parameters
	bpLit  bootstrapping.ParametersLiteral
	llama  *LlamaInference
	helper *TestHelper
	size   *LlamaSize
	slots  int
	hidStr int
	expStr int
	level  int
	logN   int
	tol    float64
	assert bool
}

func stage1MSE(got, want []complex128) (float64, error) {
	if len(got) != len(want) {
		return 0, fmt.Errorf("mse: length mismatch %d vs %d", len(got), len(want))
	}
	if len(got) == 0 {
		return 0, nil
	}
	var sum float64
	for i := range got {
		d := real(got[i]) - real(want[i])
		sum += d * d
	}
	return sum / float64(len(got)), nil
}

func precisionBitsFromMSE(mse float64) float64 {
	if mse <= 0 {
		return 64
	}
	return -math.Log2(math.Sqrt(mse))
}

func setupStage1Context(t *testing.T) *stage1Context {
	t.Helper()

	// 检查是否通过环境变量指定了配置文件（覆盖默认）
	configPath := os.Getenv("CACHEMIR_CONFIG")
	if configPath == "" {
		configPath = "config.json" // 默认使用 config.json
	}
	
	// 加载配置，如果文件不存在则使用默认配置
	config, err := LoadConfig(configPath)
	if err != nil {
		t.Fatalf("failed to load config from %s: %v", configPath, err)
	}
	
	// 如果配置文件不存在，使用测试特定的默认值
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		config.Runtime.TestModule = "Decoder"
		config.Runtime.Level = 6
		config.Model.HiddenDim = Stage1HidDim
		config.Model.ExpandedDim = Stage1ExpDim
		config.Model.SeqLen = Stage1SeqLen
		config.Model.NumHeads = Stage1NumHeads
		t.Logf("Config file not found, using default test values: hidDim=%d, expDim=%d, numHeads=%d, seqLen=%d",
			config.Model.HiddenDim, config.Model.ExpandedDim, config.Model.NumHeads, config.Model.SeqLen)
	} else {
		t.Logf("Using config from %s: hidDim=%d, expDim=%d, numHeads=%d, seqLen=%d",
			configPath, config.Model.HiddenDim, config.Model.ExpandedDim, config.Model.NumHeads, config.Model.SeqLen)
	}

	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            config.Crypto.LogN,
		LogQ:            config.Crypto.LogQ,
		LogP:            config.Crypto.LogP,
		LogDefaultScale: config.Crypto.LogDefaultScale,
		Xs:              ring.Ternary{H: config.Crypto.XsH},
	})
	if err != nil {
		t.Fatalf("failed to create CKKS parameters: %v", err)
	}
	bpLit := bootstrapping.ParametersLiteral{
		LogN: &config.Crypto.LogN,
		LogP: config.Bootstrapping.LogP,
		Xs:   params.Xs(),
	}

	llama, helper, size, _ := PrepareContextWithConfig(params, bpLit, config)
	helper.PrepareWeights(size, []string{"q", "k", "v", "out", "up", "gate", "down", "RoPE"}, llama)
	helper.PrepareCache(size, []string{"k", "v"}, llama)

	slots := params.MaxSlots()
	hidStr := slots / size.hidDim
	if hidStr <= 0 || hidStr*size.hidDim != slots {
		t.Fatalf("invalid hid stride: slots=%d hidDim=%d stride=%d", slots, size.hidDim, hidStr)
	}
	expStr := slots / size.expDim
	if expStr <= 0 || expStr*size.expDim != slots {
		t.Fatalf("invalid exp stride: slots=%d expDim=%d stride=%d", slots, size.expDim, expStr)
	}

	tol := defaultMSETol
	if v := os.Getenv("CACHEMIR_TOL"); v != "" {
		if parsed, e := strconv.ParseFloat(v, 64); e == nil && parsed > 0 {
			tol = parsed
		}
	}
	assertMode := os.Getenv("CACHEMIR_ASSERT") != "0"

	return &stage1Context{
		params: params,
		bpLit:  bpLit,
		llama:  llama,
		helper: helper,
		size:   size,
		slots:  slots,
		hidStr: hidStr,
		expStr: expStr,
		level:  6,
		logN:   8,
		tol:    tol,
		assert: assertMode,
	}
}

func makeStage1Input(t *testing.T, c *stage1Context, trial int, dim, stride int) (*rlwe.Ciphertext, []complex128) {
	t.Helper()
	rng := rand.New(rand.NewSource(stage1Seed + int64(trial)))
	msgSlots := make([]float64, c.slots)
	for i := range msgSlots {
		if i%stride == 0 {
			msgSlots[i] = -10 + 20*rng.Float64()
		}
	}
	pt := ckks.NewPlaintext(c.params, c.level)
	if err := c.helper.GetEncoder().Encode(msgSlots, pt); err != nil {
		t.Fatalf("encode plaintext failed: %v", err)
	}
	xCt, err := c.helper.GetEncryptor().EncryptNew(pt)
	if err != nil {
		t.Fatalf("encrypt failed: %v", err)
	}
	xMsg := make([]complex128, dim)
	for i := range dim {
		xMsg[i] = complex(msgSlots[i*stride], 0)
	}
	return xCt, xMsg
}

func takeByStride(decoded []complex128, outDim, stride int) []complex128 {
	out := make([]complex128, outDim)
	for i := 0; i < outDim; i++ {
		out[i] = decoded[i*stride]
	}
	return out
}

// decryptThenEncrypt decrypts a ciphertext, applies a plaintext function, then re-encrypts at targetLevel
// This is used to simulate non-linear operations in plaintext for testing linear layers
func decryptThenEncrypt(
	t *testing.T,
	c *stage1Context,
	ct *rlwe.Ciphertext,
	plaintextFn func([]complex128) []complex128,
	targetLevel int,
) *rlwe.Ciphertext {
	t.Helper()

	// Decrypt
	decoded := c.helper.Dec(ct, 0)

	// Apply plaintext function
	result := plaintextFn(decoded)

	// Re-encode and re-encrypt at target level (to ensure enough levels for subsequent ops)
	pt := ckks.NewPlaintext(c.params, targetLevel)
	if err := c.helper.GetEncoder().Encode(result, pt); err != nil {
		t.Fatalf("encode failed in decrypt-then-encrypt: %v", err)
	}

	newCt, err := c.helper.GetEncryptor().EncryptNew(pt)
	if err != nil {
		t.Fatalf("encrypt failed in decrypt-then-encrypt: %v", err)
	}

	return newCt
}

// decryptThenEncryptWithSlots decrypts, applies fn, then re-encrypts for a specific dimension
// This version properly handles the strided packing used in the project
func decryptThenEncryptWithSlots(
	t *testing.T,
	c *stage1Context,
	ct *rlwe.Ciphertext,
	dim, stride int,
	plaintextFn func([]complex128) []complex128,
	targetLevel int,
) *rlwe.Ciphertext {
	t.Helper()

	// Decrypt full slots
	decoded := c.helper.Dec(ct, 0)

	// Extract actual values by stride
	values := takeByStride(decoded, dim, stride)

	// Apply plaintext function
	resultValues := plaintextFn(values)

	// Pack back into full slots
	resultSlots := make([]complex128, c.slots)
	for i := 0; i < dim; i++ {
		resultSlots[i*stride] = resultValues[i]
	}

	// Re-encode and re-encrypt at target level
	pt := ckks.NewPlaintext(c.params, targetLevel)
	if err := c.helper.GetEncoder().Encode(resultSlots, pt); err != nil {
		t.Fatalf("encode failed in decrypt-then-encrypt: %v", err)
	}

	newCt, err := c.helper.GetEncryptor().EncryptNew(pt)
	if err != nil {
		t.Fatalf("encrypt failed in decrypt-then-encrypt: %v", err)
	}

	return newCt
}

func assertMSE(t *testing.T, c *stage1Context, block string, got, want []complex128) {
	t.Helper()
	mse, err := stage1MSE(got, want)
	if err != nil {
		t.Fatalf("%s mse failed: %v", block, err)
	}
	t.Logf("Stage1 block=%s | seqLen=%d hidDim=%d expDim=%d | MSE=%.3e | precision=%.2f bits",
		block, c.size.seqLen, c.size.hidDim, c.size.expDim, mse, precisionBitsFromMSE(mse))
	if c.assert && mse > c.tol {
		t.Fatalf("%s mse too large: got %.3e want <= %.3e", block, mse, c.tol)
	}
}

func TestStage1_LinearPipelineMSE(t *testing.T) {
	c := setupStage1Context(t)
	xCt, xMsg := makeStage1Input(t, c, 0, c.size.hidDim, c.hidStr)

	t.Run("QProjection", func(t *testing.T) {
		qCt, _, _ := c.llama.QKV(xCt.CopyNew())
		qPad := c.helper.Dec(qCt, 0)
		q := takeByStride(qPad, c.size.hidDim, c.hidStr)
		qMsg := c.llama.LinearMsg(xMsg, c.llama.wMsg["q"])
		assertMSE(t, c, "QProjection", q, qMsg)
	})

	t.Run("KProjection", func(t *testing.T) {
		_, kCt, _ := c.llama.QKV(xCt.CopyNew())
		kPad := c.helper.Dec(kCt, 0)
		k := takeByStride(kPad, c.size.hidDim, c.hidStr)
		kMsg := c.llama.LinearMsg(xMsg, c.llama.wMsg["k"])
		assertMSE(t, c, "KProjection", k, kMsg)
	})

	t.Run("VProjection", func(t *testing.T) {
		_, _, vCt := c.llama.QKV(xCt.CopyNew())
		vPad := c.helper.Dec(vCt, 0)
		v := takeByStride(vPad, c.size.hidDim, c.hidStr)
		vMsg := c.llama.LinearMsg(xMsg, c.llama.wMsg["v"])
		assertMSE(t, c, "VProjection", v, vMsg)
	})

	t.Run("OutProjection", func(t *testing.T) {
		oCt := c.llama.Out(xCt.CopyNew())
		oPad := c.helper.Dec(oCt, 0)
		o := takeByStride(oPad, c.size.hidDim, c.hidStr)
		oMsg := c.llama.LinearMsg(xMsg, c.llama.wMsg["out"])
		assertMSE(t, c, "OutProjection", o, oMsg)
	})

	t.Run("UpGateProjection", func(t *testing.T) {
		upCt, gateCt := c.llama.UpGate(xCt.CopyNew())
		upPad := c.helper.Dec(upCt, 0)
		gatePad := c.helper.Dec(gateCt, 0)
		up := takeByStride(upPad, c.size.expDim, c.expStr)
		gate := takeByStride(gatePad, c.size.expDim, c.expStr)
		upMsg := c.llama.LinearMsg(xMsg, c.llama.wMsg["up"])
		gateMsg := c.llama.LinearMsg(xMsg, c.llama.wMsg["gate"])
		assertMSE(t, c, "UpProjection", up, upMsg)
		assertMSE(t, c, "GateProjection", gate, gateMsg)
	})

	t.Run("DownProjection", func(t *testing.T) {
		// First generate expDim input via Up projection
		upCt, _ := c.llama.UpGate(xCt.CopyNew())
		upMsg := c.llama.LinearMsg(xMsg, c.llama.wMsg["up"])
		// Then apply Down projection
		dCt := c.llama.Down(upCt)
		dPad := c.helper.Dec(dCt, 0)
		d := takeByStride(dPad, c.size.hidDim, c.hidStr)
		dMsg := c.llama.LinearMsg(upMsg, c.llama.wMsg["down"])
		assertMSE(t, c, "DownProjection", d, dMsg)
	})

}

// TestStage1_NonlinearInPlaintext tests the full pipeline with non-linear operations
// done in plaintext (decrypt-compute-encrypt) to isolate linear layer testing
func TestStage1_NonlinearInPlaintext(t *testing.T) {
	c := setupStage1Context(t)
	xCt, xMsg := makeStage1Input(t, c, 0, c.size.hidDim, c.hidStr)

	t.Run("FullDecoderWithPlaintextNonlinear", func(t *testing.T) {
		// This test runs a complete decoder layer with:
		// - Linear ops (QKV, RoPE, QK^T, AttnV, Out, UpGate, Down) in ciphertext
		// - Non-linear ops (Softmax, SiLU, Norm) in plaintext (decrypt-compute-encrypt)

		eval := c.llama.eval[0]
		xCopy := xCt.CopyNew()

		// ========== Attention Part ==========
		// 1. QKV projections (linear)
		qCt, kCt, vCt := c.llama.QKV(xCopy)
		qMsg := c.llama.LinearMsg(xMsg, c.llama.wMsg["q"])
		kMsg := c.llama.LinearMsg(xMsg, c.llama.wMsg["k"])
		vMsg := c.llama.LinearMsg(xMsg, c.llama.wMsg["v"])

		// 2. RoPE (linear rotations)
		qCt, kCt = c.llama.RoPE(qCt, kCt)
		qMsg, kMsg = c.llama.RoPEMsg(qMsg, kMsg)

		// 3. Cache K, V
		c.llama.Cache(kCt, vCt)
		c.llama.CacheMsg(kMsg, vMsg)

		// 4. QK^T (linear)
		sCt := c.llama.QK_T(qCt)
		sMsg := c.llama.LinearMsg(qMsg, c.llama.cacheMsg["k"], 2)

		// 5. Softmax (non-linear -> plaintext)
		sCt = decryptThenEncrypt(t, c, sCt, c.llama.SoftmaxPlaintext, 6)

		// 6. AttnV (linear)
		oCt := c.llama.AttnV(sCt)
		oMsg := c.llama.AttnVMsg(sMsg)

		// 7. Out projection (linear)
		oCt = c.llama.Out(oCt)
		oMsg = c.llama.LinearMsg(oMsg, c.llama.wMsg["out"])

		// 8. Residual add
		eval.Add(oCt, xCt.CopyNew(), oCt)
		for i := range oMsg {
			oMsg[i] += xMsg[i]
		}

		// 9. Norm (non-linear -> plaintext)
		oCt = decryptThenEncryptWithSlots(t, c, oCt, c.size.hidDim, c.hidStr, c.llama.NormPlaintext, 6)
		oMsg = c.llama.NormPlaintext(oMsg)

		// ========== FFN Part ==========
		// 10. UpGate projections (linear)
		upCt, gateCt := c.llama.UpGate(oCt)
		upMsg := c.llama.LinearMsg(oMsg, c.llama.wMsg["up"])
		gateMsg := c.llama.LinearMsg(oMsg, c.llama.wMsg["gate"])

		// 11. SiLU on gate (non-linear -> plaintext)
		gateCt = decryptThenEncryptWithSlots(t, c, gateCt, c.size.expDim, c.expStr, c.llama.SiLUPlaintext, 6)
		gateMsg = c.llama.SiLUPlaintext(gateMsg)

		// 12. Element-wise multiply (linear)
		yCt, _ := eval.MulRelinNew(upCt, gateCt)
		eval.Rescale(yCt, yCt)
		yMsg := make([]complex128, len(upMsg))
		for i := range upMsg {
			yMsg[i] = upMsg[i] * gateMsg[i]
		}

		// 13. Down projection (linear)
		yCt = c.llama.Down(yCt)
		yMsg = c.llama.LinearMsg(yMsg, c.llama.wMsg["down"])

		// 14. Residual add
		eval.Add(oCt, yCt, yCt)
		for i := range yMsg {
			yMsg[i] += oMsg[i]
		}

		// 15. Final Norm (non-linear -> plaintext)
		yCt = decryptThenEncryptWithSlots(t, c, yCt, c.size.hidDim, c.hidStr, c.llama.NormPlaintext, 6)
		yMsg = c.llama.NormPlaintext(yMsg)

		// Compare final result
		yResult := takeByStride(c.helper.Dec(yCt, 0), c.size.hidDim, c.hidStr)
		assertMSE(t, c, "FullDecoderWithPlaintextNonlinear", yResult, yMsg)
	})

	t.Run("SiLUInPlaintext", func(t *testing.T) {
		// First do UpGate projection
		_, gateCt := c.llama.UpGate(xCt.CopyNew())
		gateCt = c.llama.BootTo(gateCt, 13)

		// Decrypt, apply SiLU in plaintext, re-encrypt
		gatePlain := decryptThenEncryptWithSlots(t, c, gateCt, c.size.expDim, c.expStr,
			c.llama.SiLUPlaintext, 6)

		// Compare with pure plaintext computation
		gateMsg := c.llama.LinearMsg(xMsg, c.llama.wMsg["gate"])
		expected := c.llama.SiLUPlaintext(gateMsg)

		resultPad := c.helper.Dec(gatePlain, 0)
		result := takeByStride(resultPad, c.size.expDim, c.expStr)
		assertMSE(t, c, "SiLUInPlaintext", result, expected)
	})

	t.Run("NormInPlaintext", func(t *testing.T) {
		// Apply Norm via plaintext computation
		// Use lower level since we don't need high levels for plaintext computation
		xCopy := xCt.CopyNew()
		xCopy = c.llama.BootTo(xCopy, 6) // Use a reasonable level

		normPlain := decryptThenEncryptWithSlots(t, c, xCopy, c.size.hidDim, c.hidStr,
			c.llama.NormPlaintext, 6)

		// Compare with pure plaintext
		expected := c.llama.NormPlaintext(xMsg)

		resultPad := c.helper.Dec(normPlain, 0)
		result := takeByStride(resultPad, c.size.hidDim, c.hidStr)
		assertMSE(t, c, "NormInPlaintext", result, expected)
	})
}

// TestStage1_DecoderPlaintext tests the new DecoderPlaintext method
func TestStage1_DecoderPlaintext(t *testing.T) {
	c := setupStage1Context(t)
	xCt, xMsg := makeStage1Input(t, c, 0, c.size.hidDim, c.hidStr)

	t.Run("FullDecoderPlaintextMethod", func(t *testing.T) {
		// Run the complete decoder using the DecoderPlaintext method
		yCt := c.llama.DecoderPlaintext(xCt.CopyNew())
		yResult := takeByStride(c.helper.Dec(yCt, 0), c.size.hidDim, c.hidStr)

		// Compare with pure plaintext reference
		yMsg := c.llama.DecoderMsg(xMsg)
		assertMSE(t, c, "DecoderPlaintextMethod", yResult, yMsg)
	})
}

// TestStage1_AttentionOnly tests only the attention part with plaintext Softmax
func TestStage1_AttentionOnly(t *testing.T) {
	c := setupStage1Context(t)
	xCt, xMsg := makeStage1Input(t, c, 0, c.size.hidDim, c.hidStr)

	t.Run("AttentionWithPlaintextSoftmax", func(t *testing.T) {
		xCopy := xCt.CopyNew()

		// Attention part only
		qCt, kCt, vCt := c.llama.QKV(xCopy)
		qMsg := c.llama.LinearMsg(xMsg, c.llama.wMsg["q"])
		kMsg := c.llama.LinearMsg(xMsg, c.llama.wMsg["k"])
		vMsg := c.llama.LinearMsg(xMsg, c.llama.wMsg["v"])

		qCt, kCt = c.llama.RoPE(qCt, kCt)
		qMsg, kMsg = c.llama.RoPEMsg(qMsg, kMsg)

		c.llama.Cache(kCt, vCt)
		c.llama.CacheMsg(kMsg, vMsg)

		sCt := c.llama.QK_T(qCt)
		sMsg := c.llama.LinearMsg(qMsg, c.llama.cacheMsg["k"], 2)

		// Softmax in plaintext
		sCt = decryptThenEncrypt(t, c, sCt, c.llama.SoftmaxPlaintext, 6)

		oCt := c.llama.AttnV(sCt)
		oMsg := c.llama.AttnVMsg(sMsg)

		oCt = c.llama.Out(oCt)
		oMsg = c.llama.LinearMsg(oMsg, c.llama.wMsg["out"])

		c.llama.eval[0].Add(oCt, xCt.CopyNew(), oCt)
		for i := range oMsg {
			oMsg[i] += xMsg[i]
		}

		oResult := takeByStride(c.helper.Dec(oCt, 0), c.size.hidDim, c.hidStr)
		assertMSE(t, c, "AttentionWithPlaintextSoftmax", oResult, oMsg)
	})
}

// TestStage1_FFNOnly tests only the FFN part with plaintext SiLU and Norm
func TestStage1_FFNOnly(t *testing.T) {
	c := setupStage1Context(t)

	// Create input for FFN (simulating output from attention + norm)
	xCt, xMsg := makeStage1Input(t, c, 0, c.size.hidDim, c.hidStr)

	t.Run("FFNWithPlaintextNonlinear", func(t *testing.T) {
		xCopy := xCt.CopyNew()

		// UpGate
		upCt, gateCt := c.llama.UpGate(xCopy)
		upMsg := c.llama.LinearMsg(xMsg, c.llama.wMsg["up"])
		gateMsg := c.llama.LinearMsg(xMsg, c.llama.wMsg["gate"])

		// SiLU in plaintext
		gateCt = decryptThenEncryptWithSlots(t, c, gateCt, c.size.expDim, c.expStr,
			c.llama.SiLUPlaintext, 6)
		gateMsg = c.llama.SiLUPlaintext(gateMsg)

		// Multiply
		yCt, _ := c.llama.eval[0].MulRelinNew(upCt, gateCt)
		c.llama.eval[0].Rescale(yCt, yCt)
		yMsg := make([]complex128, len(upMsg))
		for i := range upMsg {
			yMsg[i] = upMsg[i] * gateMsg[i]
		}

		// Down
		yCt = c.llama.Down(yCt)
		yMsg = c.llama.LinearMsg(yMsg, c.llama.wMsg["down"])

		// Residual
		c.llama.eval[0].Add(xCt.CopyNew(), yCt, yCt)
		for i := range yMsg {
			yMsg[i] += xMsg[i]
		}

		// Final Norm in plaintext
		yCt = decryptThenEncryptWithSlots(t, c, yCt, c.size.hidDim, c.hidStr,
			c.llama.NormPlaintext, 6)
		yMsg = c.llama.NormPlaintext(yMsg)

		yResult := takeByStride(c.helper.Dec(yCt, 0), c.size.hidDim, c.hidStr)
		assertMSE(t, c, "FFNWithPlaintextNonlinear", yResult, yMsg)
	})
}

// expandToSlots expands a dimension-sized vector to full slots by replication
func expandToSlots(values []complex128, slots, stride int) []complex128 {
	out := make([]complex128, slots)
	for i := range values {
		out[i*stride] = values[i]
	}
	return out
}

// encryptFromSlots encrypts values that are already in slot format
func encryptFromSlots(t *testing.T, c *stage1Context, values []complex128, dim, level int) *rlwe.Ciphertext {
	t.Helper()
	pt := ckks.NewPlaintext(c.params, level)
	if err := c.helper.GetEncoder().Encode(values, pt); err != nil {
		t.Fatalf("encode failed: %v", err)
	}
	ct, err := c.helper.GetEncryptor().EncryptNew(pt)
	if err != nil {
		t.Fatalf("encrypt failed: %v", err)
	}
	return ct
}
