
---

# linear_test.go 使用文档

## 1. 功能概述

`linear_test.go` 是 Cachemir 项目的核心测试文件，用于验证基于 **CKKS 全同态加密（FHE）** 的 LLaMA 模型线性层的正确性。该文件通过对比密文计算结果与明文参考结果，评估线性层的精度（以 MSE 和比特精度衡量）。

### 测试策略
- **线性层**：在密文中直接执行（矩阵乘法、旋转等）
- **非线性层**：通过"解密-计算-加密"方式在明文模拟（Softmax、SiLU、Norm）

---

## 2. 测试配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `Stage1SeqLen` | 29 | 序列长度 |
| `Stage1HidDim` | 32 | 隐藏层维度 |
| `Stage1ExpDim` | 64 | FFN 扩展维度 |
| `Stage1NumHeads` | 8 | 注意力头数 |
| `logN` | 8 | 多项式次数参数（N=256） |
| `level` | 6 | 密文层级 |
| `defaultMSETol` | 1e-5 | 默认 MSE 容忍阈值 |

### 环境变量控制

```bash
# 设置 MSE 容忍阈值（覆盖默认值）
export CACHEMIR_TOL=1e-4

# 禁用断言模式（仅输出精度信息，不触发测试失败）
export CACHEMIR_ASSERT=0
```

---

## 3. 测试函数详解

### 3.1 `TestStage1_LinearPipelineMSE`
**功能**：单独测试各个线性投影层的精度

**测试子项**：
- `QProjection` - Query 投影
- `KProjection` - Key 投影  
- `VProjection` - Value 投影
- `OutProjection` - Attention 输出投影
- `UpGateProjection` - FFN 的 Up 和 Gate 投影
- `DownProjection` - FFN 的 Down 投影

**使用**：
```bash
go test -v -run TestStage1_LinearPipelineMSE
```

---

### 3.2 `TestStage1_NonlinearInPlaintext`
**功能**：完整 Decoder 层端到端测试，非线性操作在明文执行

**测试子项**：
- `FullDecoderWithPlaintextNonlinear` - 完整 Decoder（Attention + FFN）
- `SiLUInPlaintext` - 仅 SiLU 激活函数
- `NormInPlaintext` - 仅 LayerNorm

**流程示意**：
```
输入 x
  ↓
[密文] QKV 投影 → RoPE → Cache
  ↓
[密文] QK^T 计算
  ↓
[明文] Softmax (解密→计算→加密)
  ↓
[密文] AttnV → Out 投影 → 残差连接
  ↓
[明文] Norm (解密→计算→加密)
  ↓
[密文] UpGate 投影
  ↓
[明文] SiLU (解密→计算→加密)
  ↓
[密文] 逐元素乘 → Down 投影 → 残差连接
  ↓
[明文] Final Norm
```

**使用**：
```bash
go test -v -run TestStage1_NonlinearInPlaintext
```

---

### 3.3 `TestStage1_DecoderPlaintext`
**功能**：测试 `DecoderPlaintext` 方法（封装好的完整 Decoder）

**使用**：
```bash
go test -v -run TestStage1_DecoderPlaintext
```

---

### 3.4 `TestStage1_AttentionOnly`
**功能**：仅测试 Attention 部分（包含明文 Softmax）

**使用**：
```bash
go test -v -run TestStage1_AttentionOnly
```

---

### 3.5 `TestStage1_FFNOnly`
**功能**：仅测试 FFN 部分（包含明文 SiLU 和 Norm）

**使用**：
```bash
go test -v -run TestStage1_FFNOnly
```

---

## 4. 核心辅助函数

### 4.1 `setupStage1Context(t)`
初始化测试上下文，包括：
- CKKS 参数配置
- Bootstrapping 参数
- 权重和缓存准备
- 步幅（stride）计算

### 4.2 `makeStage1Input(t, c, trial, dim, stride)`
生成测试输入：
- **返回**：密文（`*rlwe.Ciphertext`）和 明文参考（`[]complex128`）
- 使用固定种子（`stage1Seed=42`）保证可重复性

### 4.3 `decryptThenEncrypt(t, c, ct, fn, targetLevel)`
**解密-计算-加密**辅助函数：
- 解密密文
- 应用明文函数 `fn`
- 在指定层级重新加密

### 4.4 `assertMSE(t, c, block, got, want)`
精度断言：
- 计算 MSE 和有效比特精度
- 输出日志：`MSE=%.3e | precision=%.2f bits`
- 在断言模式下，若 MSE > tol 则测试失败

---

## 5. 运行示例

### 运行所有 Stage1 测试
```bash
go test -v -run "TestStage1" ./...
```

### 仅运行特定测试
```bash
# 仅测试线性投影
go test -v -run TestStage1_LinearPipelineMSE

# 仅测试 Attention 模块
go test -v -run TestStage1_AttentionOnly
```

### 调整精度阈值运行
```bash
# 放宽精度要求
CACHEMIR_TOL=1e-3 go test -v -run TestStage1_LinearPipelineMSE

# 仅查看精度信息，不触发失败
CACHEMIR_ASSERT=0 go test -v -run TestStage1_NonlinearInPlaintext
```

---

## 6. 精度解读

测试输出示例：
```
Stage1 block=QProjection | seqLen=29 hidDim=32 expDim=64 | MSE=1.234e-06 | precision=19.85 bits
```

- **MSE**（均方误差）：密文结果与明文参考的误差，越小越好
- **Precision（比特精度）**：有效比特数，计算公式为 `-log2(sqrt(MSE))`
  - 20 bits ≈ 6 位十进制精度
  - 一般要求 > 15 bits 以保证模型正确性

---

## 7. 依赖说明

- **Lattigo v6**：CKKS 方案实现
  - `github.com/tuneinsight/lattigo/v6/schemes/ckks`
  - `github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping`

- **项目内部依赖**：
  - `linear.go` - 线性层实现
  - `llama.go` - LlamaInference 结构
  - `util.go` - TestHelper 工具类