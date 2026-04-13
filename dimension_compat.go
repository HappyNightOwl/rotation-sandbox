// dimension_compat.go
// 维度兼容性处理模块
// 本模块提供对非常规输入尺寸（质数、非平方、不对称配置等）的兼容性支持
// 设计原则：最小侵入、可扩展、向后兼容
//
// Phase 1: 自动维度对齐 - 调整配置参数使其满足整除约束
// Phase 2: Zero-Padding 处理 - 在数据层面处理对齐，支持真实任意维度

package main

import (
	"fmt"
	"math"
)

// DimensionCompat 维度兼容性处理器
type DimensionCompat struct {
	numSlots int
	mode     CompatMode
}

// CompatMode 兼容模式
type CompatMode int

const (
	// ModeAlign Phase 1: 自动对齐模式 - 调整维度值使其满足整除约束
	ModeAlign CompatMode = 1

	// ModePadding Phase 2: Padding 模式 - 保持原始维度，通过数据padding处理
	ModePadding CompatMode = 2
)

// DimensionInfo 维度信息，记录原始值和对齐值
type DimensionInfo struct {
	Original int // 用户指定的原始维度
	Aligned  int // 实际使用的对齐后维度
	Padded   int // Phase 2: padding后的维度（如果启用）
}

// AlignedLlamaSize 对齐后的模型尺寸
type AlignedLlamaSize struct {
	HidDim   DimensionInfo
	ExpDim   DimensionInfo
	SeqLen   DimensionInfo
	NumHeads DimensionInfo

	// 内部计算参数
	preProc  int // numSlots / hidDim
	postProc int // numSlots / expDim (expand>0) or numSlots / hidDim (expand<=0)
}

// PaddingHandler Phase 2: Padding 处理器
type PaddingHandler struct {
	compat   *DimensionCompat
	size     *LlamaSize
	original LlamaSize // 保存原始尺寸
}

// NewDimensionCompat 创建维度兼容性处理器
func NewDimensionCompat(numSlots int) *DimensionCompat {
	return &DimensionCompat{
		numSlots: numSlots,
		mode:     ModeAlign, // 默认 Phase 1
	}
}

// SetMode 设置兼容模式
func (dc *DimensionCompat) SetMode(mode CompatMode) {
	dc.mode = mode
}

// GetMode 获取当前兼容模式
func (dc *DimensionCompat) GetMode() CompatMode {
	return dc.mode
}

// =============================================================================
// Phase 1: 自动维度对齐
// =============================================================================

// AlignLlamaSize 对齐模型尺寸（Phase 1 核心函数）
// 输入：用户指定的尺寸
// 输出：满足 numSlots 整除约束的对齐后尺寸
func (dc *DimensionCompat) AlignLlamaSize(size LlamaSize) AlignedLlamaSize {
	result := AlignedLlamaSize{}

	// 对齐 hidDim
	result.HidDim.Original = size.hidDim
	result.HidDim.Aligned = dc.alignToDivisor(size.hidDim, "hidDim")
	result.HidDim.Padded = result.HidDim.Aligned

	// 对齐 expDim
	result.ExpDim.Original = size.expDim
	result.ExpDim.Aligned = dc.alignToDivisor(size.expDim, "expDim")
	result.ExpDim.Padded = result.ExpDim.Aligned

	// seqLen 和 numHeads 不需要对齐 slots，但保持一致性
	result.SeqLen.Original = size.seqLen
	result.SeqLen.Aligned = size.seqLen
	result.SeqLen.Padded = size.seqLen

	result.NumHeads.Original = size.numHeads
	result.NumHeads.Aligned = size.numHeads
	result.NumHeads.Padded = size.numHeads

	// 计算内部参数
	result.preProc = dc.numSlots / result.HidDim.Aligned
	result.postProc = dc.numSlots / result.HidDim.Aligned

	return result
}

// alignToDivisor 将维度对齐到 numSlots 的最近因数
// 策略：
// 1. 如果 dim 能整除 numSlots，直接返回
// 2. 优先向上取（保持容量）：找到最小的 d >= dim 使得 numSlots % d == 0
// 3. 如果向上取超过 numSlots，则向下取
func (dc *DimensionCompat) alignToDivisor(dim int, name string) int {
	if dim <= 0 {
		return dim
	}

	// 已经对齐
	if dc.numSlots%dim == 0 {
		return dim
	}

	// 策略1：向上取整到下一个能整除的值
	alignedUp := -1
	for d := dim; d <= dc.numSlots; d++ {
		if dc.numSlots%d == 0 {
			alignedUp = d
			break
		}
	}

	// 策略2：向下取整到上一个能整除的值
	alignedDown := -1
	for d := dim; d >= 1; d-- {
		if dc.numSlots%d == 0 {
			alignedDown = d
			break
		}
	}

	// 选择策略
	var aligned int
	if alignedUp != -1 {
		// 优先向上取，保持模型容量
		aligned = alignedUp
	} else if alignedDown != -1 {
		// 向上取失败（超过numSlots），向下取
		aligned = alignedDown
	} else {
		// 极端情况，使用1
		aligned = 1
	}

	// 输出警告
	if aligned != dim {
		fmt.Printf("⚠️  [%s] Auto-aligned: %d → %d (nearest divisor of numSlots=%d)\n",
			name, dim, aligned, dc.numSlots)
	}

	return aligned
}

// ApplyToLlamaSize 将对齐后的尺寸应用到 LlamaSize（原地修改）
func (dc *DimensionCompat) ApplyToLlamaSize(size *LlamaSize) AlignedLlamaSize {
	aligned := dc.AlignLlamaSize(*size)

	// 只有当对齐后的值不同时才修改
	if aligned.HidDim.Aligned != size.hidDim {
		size.hidDim = aligned.HidDim.Aligned
	}
	if aligned.ExpDim.Aligned != size.expDim {
		size.expDim = aligned.ExpDim.Aligned
	}
	// seqLen 和 numHeads 通常不需要修改

	return aligned
}

// CheckDivisible 检查维度是否能整除 numSlots
func (dc *DimensionCompat) CheckDivisible(size LlamaSize) (bool, []string) {
	issues := []string{}

	if dc.numSlots%size.hidDim != 0 {
		issues = append(issues, fmt.Sprintf("hidDim=%d not divisible by numSlots=%d (remainder=%d)",
			size.hidDim, dc.numSlots, dc.numSlots%size.hidDim))
	}
	if dc.numSlots%size.expDim != 0 {
		issues = append(issues, fmt.Sprintf("expDim=%d not divisible by numSlots=%d (remainder=%d)",
			size.expDim, dc.numSlots, dc.numSlots%size.expDim))
	}

	return len(issues) == 0, issues
}

// PrintAlignmentInfo 打印对齐信息
func (dc *DimensionCompat) PrintAlignmentInfo(aligned AlignedLlamaSize) {
	fmt.Println("==================================================")
	fmt.Println("Dimension Alignment Info")
	fmt.Println("==================================================")
	fmt.Printf("numSlots: %d\n", dc.numSlots)
	fmt.Println()
	fmt.Printf("%-10s %10s %10s %10s\n", "Param", "Original", "Aligned", "Delta")
	fmt.Println("--------------------------------------------------")

	printInfo := func(name string, info DimensionInfo) {
		delta := info.Aligned - info.Original
		if delta != 0 {
			fmt.Printf("%-10s %10d %10d %+10d ⚠️\n", name, info.Original, info.Aligned, delta)
		} else {
			fmt.Printf("%-10s %10d %10d %10d ✓\n", name, info.Original, info.Aligned, delta)
		}
	}

	printInfo("hidDim", aligned.HidDim)
	printInfo("expDim", aligned.ExpDim)
	printInfo("seqLen", aligned.SeqLen)
	printInfo("numHeads", aligned.NumHeads)

	fmt.Println("==================================================")
}

// =============================================================================
// Phase 2: Zero-Padding 处理（预留接口，待实现）
// =============================================================================

// =============================================================================
// Phase 2: Zero-Padding 处理
// =============================================================================

// NewPaddingHandler 创建 Padding 处理器（Phase 2 入口）
// Phase 2 核心：不改变模型参数，而是在数据层面进行 padding 处理
// 这样用户可以指定任意维度（如 hidDim=17），系统自动处理到 numSlots 对齐
func (dc *DimensionCompat) NewPaddingHandler(size *LlamaSize) *PaddingHandler {
	// 计算需要的 padding 维度（下一个能整除 numSlots 的值）
	paddedSize := *size

	// 计算 hidDim 需要 padding 到的维度
	if numSlots := dc.numSlots; numSlots%size.hidDim != 0 {
		for d := size.hidDim + 1; d <= numSlots; d++ {
			if numSlots%d == 0 {
				paddedSize.hidDim = d
				break
			}
		}
	}

	// 计算 expDim 需要 padding 到的维度
	if numSlots := dc.numSlots; numSlots%size.expDim != 0 {
		for d := size.expDim + 1; d <= numSlots; d++ {
			if numSlots%d == 0 {
				paddedSize.expDim = d
				break
			}
		}
	}

	// 保存原始尺寸
	original := LlamaSize{
		hidDim:   size.hidDim,
		expDim:   size.expDim,
		seqLen:   size.seqLen,
		numHeads: size.numHeads,
	}

	return &PaddingHandler{
		compat:   dc,
		size:     &paddedSize,
		original: original,
	}
}

// PadInputVector 对输入向量进行 Zero-Padding
// 将原始维度 padding 到能整除 numSlots 的维度
func (ph *PaddingHandler) PadInputVector(input []complex128, targetDim int) []complex128 {
	if len(input) >= targetDim {
		return input[:targetDim]
	}

	// 创建 padded 向量，尾部补零
	padded := make([]complex128, targetDim)
	copy(padded, input)
	// 剩余部分默认为零（complex128零值）
	return padded
}

// UnpadOutputVector 从 padded 输出中提取有效部分
// 从 padding 后的维度恢复到原始维度
func (ph *PaddingHandler) UnpadOutputVector(output []complex128, originalDim int) []complex128 {
	if len(output) >= originalDim {
		return output[:originalDim]
	}
	// 如果输出比原始维度还小，直接返回
	return output
}

// PadWeightMatrix 对权重矩阵进行 Zero-Padding
// 将原始 (rows, cols) 的矩阵 padding 到 (targetRows, targetCols)
func (ph *PaddingHandler) PadWeightMatrix(matrix [][]complex128, targetRows, targetCols int) [][]complex128 {
	originalRows := len(matrix)
	originalCols := 0
	if originalRows > 0 {
		originalCols = len(matrix[0])
	}

	// 创建 padded 矩阵
	padded := make([][]complex128, targetRows)
	for i := 0; i < targetRows; i++ {
		padded[i] = make([]complex128, targetCols)
		// 复制原始数据
		if i < originalRows {
			copyLen := originalCols
			if copyLen > targetCols {
				copyLen = targetCols
			}
			for j := 0; j < copyLen; j++ {
				padded[i][j] = matrix[i][j]
			}
		}
		// 超出原始行/列的部分保持为零
	}
	return padded
}

// UnpadWeightMatrix 从 padded 矩阵中提取有效部分
func (ph *PaddingHandler) UnpadWeightMatrix(paddedMatrix [][]complex128, originalRows, originalCols int) [][]complex128 {
	if len(paddedMatrix) >= originalRows {
		result := make([][]complex128, originalRows)
		for i := 0; i < originalRows; i++ {
			result[i] = make([]complex128, originalCols)
			if len(paddedMatrix[i]) >= originalCols {
				copy(result[i], paddedMatrix[i][:originalCols])
			} else {
				copy(result[i], paddedMatrix[i])
			}
		}
		return result
	}
	return paddedMatrix
}

// GetEffectiveDimensions 获取 padding 后的有效维度
func (ph *PaddingHandler) GetEffectiveDimensions() LlamaSize {
	return *ph.size
}

// GetOriginalDimensions 获取原始用户指定的维度
func (ph *PaddingHandler) GetOriginalDimensions() LlamaSize {
	return ph.original
}

// GetPaddingInfo 获取 padding 信息
func (ph *PaddingHandler) GetPaddingInfo() (origHidDim, paddedHidDim, origExpDim, paddedExpDim int) {
	return ph.original.hidDim, ph.size.hidDim, ph.original.expDim, ph.size.expDim
}

// =============================================================================
// Phase 2 包装层：用于集成到现有代码
// =============================================================================

// Phase2Wrapper Phase 2 包装器，提供透明的数据转换
type Phase2Wrapper struct {
	handler  *PaddingHandler
	numSlots int
	enabled  bool
}

// NewPhase2Wrapper 创建 Phase 2 包装器
func NewPhase2Wrapper(numSlots int, originalSize *LlamaSize) *Phase2Wrapper {
	compat := NewDimensionCompat(numSlots)
	compat.SetMode(ModePadding)
	handler := compat.NewPaddingHandler(originalSize)

	return &Phase2Wrapper{
		handler:  handler,
		numSlots: numSlots,
		enabled:  true,
	}
}

// WrapInput 包装输入数据（自动 padding）
func (w *Phase2Wrapper) WrapInput(x []complex128, dimType string) []complex128 {
	if !w.enabled {
		return x
	}

	effective := w.handler.GetEffectiveDimensions()

	switch dimType {
	case "hid":
		if len(x) < effective.hidDim {
			return w.handler.PadInputVector(x, effective.hidDim)
		}
	case "exp":
		if len(x) < effective.expDim {
			return w.handler.PadInputVector(x, effective.expDim)
		}
	case "seq":
		if len(x) < effective.seqLen {
			return w.handler.PadInputVector(x, effective.seqLen)
		}
	}
	return x
}

// UnwrapOutput 解包输出数据（自动 unpadding）
func (w *Phase2Wrapper) UnwrapOutput(y []complex128, dimType string) []complex128 {
	if !w.enabled {
		return y
	}

	orig := w.handler.GetOriginalDimensions()

	switch dimType {
	case "hid":
		return w.handler.UnpadOutputVector(y, orig.hidDim)
	case "exp":
		return w.handler.UnpadOutputVector(y, orig.expDim)
	case "seq":
		return w.handler.UnpadOutputVector(y, orig.seqLen)
	}
	return y
}

// WrapWeightMatrix 包装权重矩阵
func (w *Phase2Wrapper) WrapWeightMatrix(matrix [][]complex128, rows, cols int) [][]complex128 {
	if !w.enabled {
		return matrix
	}
	return w.handler.PadWeightMatrix(matrix, rows, cols)
}

// IsPaddingNeeded 检查是否需要进行 padding
func (w *Phase2Wrapper) IsPaddingNeeded() bool {
	if !w.enabled {
		return false
	}
	orig := w.handler.GetOriginalDimensions()
	effective := w.handler.GetEffectiveDimensions()
	return orig.hidDim != effective.hidDim || orig.expDim != effective.expDim
}

// =============================================================================
// 工具函数
// =============================================================================

// FindNearestDivisors 找到 numSlots 的所有因数（用于调试）
func FindNearestDivisors(numSlots int, target int) []int {
	divisors := []int{}
	for d := 1; d <= numSlots; d++ {
		if numSlots%d == 0 {
			// 只收集接近 target 的因数
			diff := abs(d - target)
			if diff <= target/2 || diff <= 10 {
				divisors = append(divisors, d)
			}
		}
	}
	return divisors
}

// abs 绝对值
func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// IsPowerOfTwo 检查是否为2的幂
func IsPowerOfTwo(n int) bool {
	return n > 0 && (n&(n-1)) == 0
}

// NextPowerOfTwo 下一个2的幂
func NextPowerOfTwo(n int) int {
	if n <= 0 {
		return 1
	}
	p := 1
	for p < n {
		p *= 2
	}
	return p
}

// CalculateSlotsNeeded 计算给定维度所需的最小 slots
func CalculateSlotsNeeded(hidDim, expDim int) int {
	// 基于当前算法：需要 numSlots 能被 hidDim 和 expDim 整除
	// 最小 slots 是 LCM(hidDim, expDim)
	return lcm(hidDim, expDim)
}

// gcd 最大公约数
func gcd(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

// lcm 最小公倍数
func lcm(a, b int) int {
	if a == 0 || b == 0 {
		return 0
	}
	return abs(a*b) / gcd(a, b)
}

// SuggestCryptoParams 根据模型维度建议加密参数
// 返回建议的 logN 值
func SuggestCryptoParams(hidDim, expDim int) int {
	minSlots := CalculateSlotsNeeded(hidDim, expDim)
	// logN = log2(numSlots)
	logN := int(math.Ceil(math.Log2(float64(minSlots))))
	// 确保至少为8（Lattigo最小要求）
	if logN < 8 {
		logN = 8
	}
	// CKKS 通常使用 13-16
	if logN > 16 {
		fmt.Printf("⚠️  Required logN=%d exceeds typical CKKS range (13-16)\n", logN)
	}
	return logN
}
