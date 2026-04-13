# 维度兼容性处理文档

本文档描述 Cachemir 项目中处理非常规维度（如质数、非 2 的幂、不对称配置）的实现方案。

## 问题背景

CKKS 全同态加密方案要求矩阵维度必须满足整除约束：
- `numSlots % hidDim == 0`
- `numSlots % expDim == 0`

其中 `numSlots = 2^(logN-1)` 由加密参数决定。

当用户指定非常规维度（如 `hidDim=17`）时，系统需要自动适配。

## 解决方案

### Phase 1: 自动维度对齐（已实现）

**策略**：自动调整配置参数到最近的满足整除约束的值。

**实现**：
- 文件：`dimension_compat.go`
- 核心函数：`AlignLlamaSize()`

**对齐规则**：
```
对于维度 d 和 numSlots n：
1. 如果 n % d == 0，保持 d
2. 否则，找到最小的 d' >= d，使得 n % d' == 0
3. 如果找不到（d' > n），则向下找到最大的 d' <= d
```

**示例**：
```bash
# 用户指定 hidDim=17，numSlots=256
# 自动对齐: 17 → 32
# 因为 256 % 32 == 0

$ go run . -hidDim=17 -expDim=50
⚠️  [hidDim] Auto-aligned: 17 → 32 (nearest divisor of numSlots=256)
⚠️  [expDim] Auto-aligned: 50 → 64 (nearest divisor of numSlots=256)
```

**注意**：
- Phase 1 **改变了模型维度本身**（17→32）
- 生成的权重矩阵是 32×32（而非 17×17）
- 额外的维度填充的是实际随机值，而非零
- 这实际上是**训练了一个不同维度的模型**

### Phase 2: Zero-Padding（已实现）

**策略**：保持原始维度，在数据编码层面进行 Zero-Padding。

**实现**：
- 文件：`dimension_compat.go`
- 核心函数：`NewPaddingHandler()`
- 开关：`-phase2` 命令行参数
- 集成：`main.go`, `util.go`, `llama.go`

**工作原理**：
```
用户指定 hidDim=17
权重生成：17×17（原始维度）
slots 编码：17→32（padding）
  - slots 0-16: 有效权重值
  - slots 17-31: 零值
计算：使用 32 维 FHE 操作
输出：unpad 回 17 维
```

**使用方式**：
```bash
# 启用 Phase 2 模式
$ go run . -phase2 -hidDim=17
>>> Phase 2: Zero-Padding Mode <<<
Phase 2 Zero-Padding:
  hidDim: 17 → 32 (padded)
  expDim: 50 → 64 (padded)
  effective dimensions will be used for FHE operations
```

**测试结果**：
```bash
$ go run . -phase2 -hidDim=17 -expDim=50 -test=QKV
Error: 0.000000
Precision: 32.87 bits
```

**与 Phase 1 的区别**：

| 特性 | Phase 1 (自动对齐) | Phase 2 (Zero-Padding) |
|------|-------------------|----------------------|
| 配置值 | 修改（17→32） | 保持（17） |
| 权重矩阵 | 32×32（新维度） | 17×17（原始）→pad→32×32 |
| 额外权重值 | 随机值（非零） | 零值 |
| 计算复杂度 | O(32²) | O(17²)+O(pad) |
| 输出精度 | 32维 | 17维（unpad） |
| MSE基准 | 不同模型 | ✅ 可比（同一模型） |

## 使用指南

### 命令行参数

```bash
# 默认模式（Phase 1）
go run . -hidDim=17 -expDim=50

# Phase 2 模式
go run . -phase2 -hidDim=17 -expDim=50

# 生成配置文件模板
go run . -gen-config myconfig.json
```

### 测试

```bash
# 测试非常规参数（Phase 1）
go test -v -run TestStage1_LinearPipelineMSE -args -hidDim=17 -expDim=50

# 预期输出：自动对齐警告，测试通过
⚠️  [hidDim] Auto-aligned: 17 → 32
⚠️  [expDim] Auto-aligned: 50 → 64
... precision=32.XX bits
PASS
```

### 配置文件

示例 `config.json`：
```json
{
  "model": {
    "hidDim": 17,      // 可以是非常规值
    "expDim": 50,
    "numHeads": 2,
    "seqLen": 29
  },
  "crypto": {
    "logN": 8,
    "logDefaultScale": 41,
    "logQ": [...],
    "logP": [...],
    "xsH": 192
  }
}
```

## 技术实现

### 核心数据结构

```go
// dimension_compat.go

type DimensionCompat struct {
    numSlots int
    mode     CompatMode  // ModeAlign or ModePadding
}

type AlignedLlamaSize struct {
    HidDim   DimensionInfo  // Original, Aligned, Padded
    ExpDim   DimensionInfo
    SeqLen   DimensionInfo
    NumHeads DimensionInfo
}

type PaddingHandler struct {
    compat   *DimensionCompat
    size     *LlamaSize      // Padded dimensions
    original LlamaSize       // Original dimensions
}
```

### 关键函数

**Phase 1**：
```go
func NewDimensionCompat(numSlots int) *DimensionCompat
func (dc *DimensionCompat) AlignLlamaSize(size LlamaSize) AlignedLlamaSize
func (dc *DimensionCompat) ApplyToLlamaSize(size *LlamaSize) AlignedLlamaSize
```

**Phase 2**（完整）：
```go
// dimension_compat.go
func (dc *DimensionCompat) NewPaddingHandler(size *LlamaSize) *PaddingHandler
func (ph *PaddingHandler) PadInputVector(input []complex128, targetDim int) []complex128
func (ph *PaddingHandler) UnpadOutputVector(output []complex128, originalDim int) []complex128
func (ph *PaddingHandler) PadWeightMatrix(matrix [][]complex128, targetRows, targetCols int) [][]complex128

// Phase2Wrapper 包装器
type Phase2Wrapper struct {
    handler  *PaddingHandler
    numSlots int
    enabled  bool
}
func NewPhase2Wrapper(numSlots int, originalSize *LlamaSize) *Phase2Wrapper

// llama.go - 输出 unpadding
func (llama *LlamaInference) UnpadOutput(msg []complex128, dimType string) []complex128
```

## 集成点

### main.go

```go
// 在 config 加载后，创建 params 前
numSlots := 1 << config.Crypto.LogN
var origSize LlamaSize

if *phase2 {
    // Phase 2: 保持原始维度
    origSize = LlamaSize{
        hidDim:   config.Model.HiddenDim,
        expDim:   config.Model.ExpandedDim,
    }
    compat := NewDimensionCompat(numSlots)
    compat.SetMode(ModePadding)
    handler := compat.NewPaddingHandler(&origSize)
    effective := handler.GetEffectiveDimensions()
    config.Model.HiddenDim = effective.hidDim
    config.Model.ExpandedDim = effective.expDim
} else {
    // Phase 1: 自动对齐维度
    compat := NewDimensionCompat(numSlots)
    aligned := compat.AlignLlamaSize(LlamaSize{...})
    config.Model.HiddenDim = aligned.HidDim.Aligned
    config.Model.ExpandedDim = aligned.ExpDim.Aligned
}

// Phase 2 wrapper 传递给 PrepareContext
var phase2Wrapper *Phase2Wrapper
if *phase2 {
    phase2Wrapper = NewPhase2Wrapper(numSlots, &origSize)
}
llama, helper, size, _ := PrepareContextWithConfig(params, bpLit, config, phase2Wrapper)
```

### util.go - PrepareContextWithConfig

```go
func PrepareContextWithConfig(params ckks.Parameters, btpParametersLit bootstrapping.ParametersLiteral, config *Config, phase2Wrapper *Phase2Wrapper) (...) {
    // ...
    size = config.Model.ToLlamaSize()
    var originalSize *LlamaSize
    phase2Mode := false
    if phase2Wrapper != nil {
        phase2Mode = true
        orig := phase2Wrapper.handler.GetOriginalDimensions()
        originalSize = &orig
    }
    llama = &LlamaInference{
        size:         size,
        originalSize: originalSize,  // Phase 2: 保存原始维度
        phase2Mode:   phase2Mode,
        // ...
    }
}
```

### llama.go - UnpadOutput

```go
func (llama *LlamaInference) UnpadOutput(msg []complex128, dimType string) []complex128 {
    if !llama.phase2Mode || llama.originalSize == nil {
        return msg
    }
    // 从 padded 输出提取原始维度
    orig := llama.originalSize
    effective := llama.size
    // ... 提取逻辑
    return unpadded
}
```

## 已知限制

### Phase 1 限制

1. **改变模型维度**：对齐后的维度生成的是完整新模型，不是原模型的 padding 版本
2. **MSE 基准问题**：无法与原维度模型进行公平对比（因为模型本身不同）
3. **预训练权重**：无法直接使用预训练的非常规模型权重

### Phase 2 限制

1. **计算开销**：仍需 padding 到对齐维度计算，但模型权重为原始维度
2. **输出处理**：需要显式调用 `UnpadOutput` 提取原始维度结果

## 结论

**当前状态**：
- ✅ Phase 1 完全可用：自动对齐非常规维度，测试通过
- ✅ Phase 2 完全可用：Zero-Padding 模式，权重保持原始维度

**推荐用法**：
- 实验/测试场景：使用 Phase 1（自动对齐）
- 生产/预训练模型：使用 Phase 2（Zero-Padding）

**关键理解**：
Phase 1 的"对齐"是**训练不同维度的模型**，Phase 2 是**同一模型的 padded 推理**。
