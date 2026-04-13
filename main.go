package main

import (
	"flag"
	"fmt"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

var (
	// 原有的 flag 现在用于覆盖配置文件中的值
	logN     = flag.Int("logN", 8, "logarithm of polynomial degree (overrides config)")
	test     = flag.String("test", "", "the module to test (overrides config)")
	level    = flag.Int("level", 16, "input level of the module (overrides config)")
	btpLevel = flag.Int("btpLevel", 15, "bootstrap level of the module, limited to Norm and Softmax (overrides config)")
	hidDim   = flag.Int("hidDim", 32, "hidden dimension of the model (overrides config)")
	expDim   = flag.Int("expDim", 64, "expanded hidden dimension of the model (overrides config)")
	seqLen   = flag.Int("seqLen", 29, "input sequence length (overrides config)")
	numHeads = flag.Int("numHeads", 2, "number of heads (overrides config)")
	parallel = flag.Bool("parallel", false, "use parallel computing or not")

	// 新增：配置文件路径
	configPath = flag.String("config", "config.json", "path to configuration file")
	// 新增：生成默认配置文件模板
	generateConfig = flag.String("gen-config", "", "generate default config template to specified path and exit")
	// 新增：Phase 2 模式开关
	phase2 = flag.Bool("phase2", false, "enable Phase 2 Zero-Padding mode (default: Phase 1 align mode)")
)

func main() {
	flag.Parse()

	// 如果指定了生成配置文件，生成后退出
	if *generateConfig != "" {
		if err := SaveConfig(*generateConfig, DefaultConfig()); err != nil {
			panic(fmt.Sprintf("Failed to generate config: %v", err))
		}
		fmt.Printf("Default configuration template generated at: %s\n", *generateConfig)
		return
	}

	// 加载配置（配置文件不存在则使用默认值）
	config, err := LoadConfig(*configPath)
	if err != nil {
		panic(fmt.Sprintf("Failed to load config: %v", err))
	}

	// 应用命令行 flag 覆盖（如果显式设置）
	overrides := &FlagOverrides{
		LogN:     logN,
		HidDim:   hidDim,
		ExpDim:   expDim,
		SeqLen:   seqLen,
		NumHeads: numHeads,
		Level:    level,
		BtpLevel: btpLevel,
		Test:     test,
		Parallel: parallel,
	}
	config.ApplyOverrides(overrides)

	// ===== 维度兼容性处理 =====
	numSlots := 1 << config.Crypto.LogN
	var origSize LlamaSize

	if *phase2 {
		fmt.Println(">>> Phase 2: Zero-Padding Mode <<<")
		compat := NewDimensionCompat(numSlots)
		compat.SetMode(ModePadding)
		origSize = LlamaSize{
			hidDim:   config.Model.HiddenDim,
			expDim:   config.Model.ExpandedDim,
			seqLen:   config.Model.SeqLen,
			numHeads: config.Model.NumHeads,
		}
		handler := compat.NewPaddingHandler(&origSize)
		effective := handler.GetEffectiveDimensions()
		origHid, padHid, origExp, padExp := handler.GetPaddingInfo()

		fmt.Printf("Phase 2 Zero-Padding:\n")
		fmt.Printf("  hidDim: %d → %d (padded)\n", origHid, padHid)
		fmt.Printf("  expDim: %d → %d (padded)\n", origExp, padExp)
		fmt.Printf("  effective dimensions will be used for FHE operations\n")

		config.Model.HiddenDim = effective.hidDim
		config.Model.ExpandedDim = effective.expDim
	} else {
		fmt.Println(">>> Phase 1: Auto-Align Mode <<<")
		compat := NewDimensionCompat(numSlots)
		aligned := compat.AlignLlamaSize(LlamaSize{
			hidDim:   config.Model.HiddenDim,
			expDim:   config.Model.ExpandedDim,
			seqLen:   config.Model.SeqLen,
			numHeads: config.Model.NumHeads,
		})
		config.Model.HiddenDim = aligned.HidDim.Aligned
		config.Model.ExpandedDim = aligned.ExpDim.Aligned
		compat.PrintAlignmentInfo(aligned)
	}

	// 打印当前配置（便于调试）
	fmt.Printf("=== Running with Configuration ===\n")
	fmt.Printf("Model: hidDim=%d, expDim=%d, numHeads=%d, seqLen=%d\n",
		config.Model.HiddenDim, config.Model.ExpandedDim, config.Model.NumHeads, config.Model.SeqLen)
	fmt.Printf("Crypto: logN=%d, logDefaultScale=%d\n",
		config.Crypto.LogN, config.Crypto.LogDefaultScale)
	fmt.Printf("Runtime: test=%s, level=%d, btpLevel=%d, parallel=%v\n",
		config.Runtime.TestModule, config.Runtime.Level, config.Runtime.BtpLevel, config.Runtime.Parallel)
	fmt.Printf("==================================\n\n")

	// 使用配置创建 CKKS 参数
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            config.Crypto.LogN,
		LogQ:            config.Crypto.LogQ,
		LogP:            config.Crypto.LogP,
		LogDefaultScale: config.Crypto.LogDefaultScale,
		Xs:              ring.Ternary{H: config.Crypto.XsH},
	})
	if err != nil {
		panic(err)
	}

	btpParametersLit := bootstrapping.ParametersLiteral{
		LogN: &config.Crypto.LogN,
		LogP: config.Bootstrapping.LogP,
		Xs:   params.Xs(),
	}

	var phase2Wrapper *Phase2Wrapper
	if *phase2 {
		phase2Wrapper = NewPhase2Wrapper(numSlots, &origSize)
	}

	llama, helper, size, opeval := PrepareContextWithConfig(params, btpParametersLit, config, phase2Wrapper)

	fmt.Print("Initialization finished!\n")
	x := helper.ctGen(1)[0]
	xDec := make([]complex128, x.Slots())
	helper.encoder.Decode(helper.decryptor.DecryptNew(x), xDec)
	xMsg := make([]complex128, size.hidDim)
	for i := 0; i < size.hidDim; i++ {
		xMsg[i] = xDec[i*x.Slots()/size.hidDim]
	}

	switch config.Runtime.TestModule {
	case "QKV":
		helper.PrepareWeights(size, []string{"q", "k", "v"}, llama)
		qCt, _, _ := llama.QKV(x)
		qPad := llama.helper.Dec(qCt, 0)
		q := make([]complex128, size.hidDim)
		for i := 0; i < size.hidDim; i++ {
			q[i] = qPad[i*x.Slots()/size.hidDim]
		}
		qMsg := llama.LinearMsg(xMsg, llama.wMsg["q"])
		llama.helper.MSE(q, qMsg)
	case "RoPE":
		helper.PrepareWeights(size, []string{"RoPE"}, llama)
		xCt, _ := llama.RoPE(x, x)
		xPad := llama.helper.Dec(xCt, 0)
		xOut := make([]complex128, size.hidDim)
		for i := 0; i < size.hidDim; i++ {
			xOut[i] = xPad[i*x.Slots()/size.hidDim]
		}
		xMsgRot, _ := llama.RoPEMsg(xMsg, xMsg)
		llama.helper.MSE(xOut, xMsgRot)
	case "Cache":
		helper.PrepareCache(size, []string{"k", "v"}, llama)
		llama.Cache(x, x)
	case "QK_T":
		helper.PrepareCache(size, []string{"k"}, llama)
		sCt := llama.QK_T(x)
		sMsg := llama.LinearMsg(xMsg, llama.cacheMsg["k"], 2)
		llama.helper.Dec(sCt, 10)
		fmt.Print(sMsg[:10], "\n")
	case "AttnV":
		helper.PrepareCache(size, []string{"k", "v"}, llama)
		sCt := llama.QK_T(x)
		sMsg := llama.LinearMsg(xMsg, llama.cacheMsg["k"], 2)
		llama.helper.Dec(sCt, 128)
		fmt.Print(sMsg[:58], "\n")
		oCt := llama.AttnV(sCt)
		oMsg := llama.AttnVMsg(sMsg)
		llama.helper.Dec(oCt, 128)
		fmt.Print(oMsg[:32], "\n")
	case "Out":
		helper.PrepareWeights(size, []string{"out"}, llama)
		llama.Out(x)
	case "UpGate":
		helper.PrepareWeights(size, []string{"up", "gate"}, llama)
		upCt, _ := llama.UpGate(x)
		upPad := llama.helper.Dec(upCt, 0)
		up := make([]complex128, size.expDim)
		for i := 0; i < size.expDim; i++ {
			up[i] = upPad[i*x.Slots()/size.expDim]
		}
		upMsg := llama.LinearMsg(xMsg, llama.wMsg["up"])
		llama.helper.MSE(up, upMsg)
	case "Down":
		helper.PrepareWeights(size, []string{"up", "gate"}, llama)
		helper.PrepareWeights(size, []string{"down"}, llama)
		upCt, _ := llama.UpGate(x)
		downCt := llama.Down(upCt)
		downPad := llama.helper.Dec(downCt, 0)
		down := make([]complex128, size.hidDim)
		for i := 0; i < size.hidDim; i++ {
			down[i] = downPad[i*x.Slots()/size.hidDim]
		}
		upMsg := llama.LinearMsg(xMsg, llama.wMsg["up"])
		downMsg := llama.LinearMsg(upMsg, llama.wMsg["down"])
		llama.helper.MSE(down, downMsg)
	case "SiLU":
		y_pt := llama.SiLUPlaintext(llama.helper.Dec(x, 0))
		y := llama.helper.Dec(llama.SiLU(x), 0)
		llama.helper.MSE(y, y_pt)
	case "Softmax":
		y_pt := llama.SoftmaxPlaintext(llama.helper.Dec(x, 0))
		y := llama.helper.Dec(llama.Softmax(x, config.Runtime.BtpLevel, 0), 0)
		llama.helper.MSE(y, y_pt)
	case "Norm":
		y_pt := llama.NormPlaintext(llama.helper.Dec(x, 0))
		y := llama.helper.Dec(llama.Norm(x, config.Runtime.BtpLevel), 0)
		llama.helper.MSE(y, y_pt)
	case "NormThor":
		y_pt := llama.NormPlaintext(llama.helper.Dec(x, 0))
		y := llama.helper.Dec(llama.NormThor(x, 0), 0)
		llama.helper.MSE(y, y_pt)
	case "Argmax":
		// fmt.Printf("Plaintext argmax: %d\n", llama.ArgmaxPlaintext(llama.helper.Dec(x, 0)))
		pt := llama.ArgmaxPlaintext(llama.helper.Dec(x, 0))
		fmt.Print("Plaintext argmax: ")
		for _, val := range pt {
			fmt.Printf("%d+%di; ", val%16, val/16)
		}
		fmt.Print("\n")
		llama.helper.Dec(llama.Argmax(x), 8)
	case "CtMult":
		opeval.ctCtMult()
	case "Ops":
		opeval.EvaluateAll()
	case "Decoder":
		fmt.Printf("Preparing model...\n")
		helper.PrepareWeights(size, []string{"q", "k", "v", "out", "up", "gate", "down", "RoPE"}, llama)
		helper.PrepareCache(size, []string{"k", "v"}, llama)
		fmt.Printf("Preparation finished!\nEvaluating one decoder...\n")
		outCt := llama.DecoderThor(x)
		outPad := llama.helper.Dec(outCt, 0)
		out := make([]complex128, size.hidDim)
		for i := 0; i < size.hidDim; i++ {
			out[i] = outPad[i*x.Slots()/size.hidDim]
		}
		outMsg := llama.DecoderMsg(xMsg)
		llama.helper.MSE(out, outMsg)
	case "Model":
		fmt.Printf("Preparing model...\n")
		helper.PrepareWeights(size, []string{"q", "k", "v", "out", "up", "gate", "down", "RoPE"}, llama)
		helper.PrepareCache(size, []string{"k", "v"}, llama)
		fmt.Printf("Preparation finished!\nEvaluating End-to-end Inference!\n")
		llama.Model(x)
	case "ModelPlaintext":
		fmt.Printf("Preparing model with plaintext non-linear operations...\n")
		helper.PrepareWeights(size, []string{"q", "k", "v", "out", "up", "gate", "down", "RoPE"}, llama)
		helper.PrepareCache(size, []string{"k", "v"}, llama)
		fmt.Printf("Preparation finished!\nEvaluating End-to-end Inference with plaintext non-linear ops!\n")
		llama.ModelPlaintext(x)
	default:
		fmt.Print("Please specify the module to evaluate.")
	}
}
