# 配置系统使用说明

## 概述

本项目现已支持外部配置文件来管理运行参数，替代原有的硬编码 flag 参数。

## 配置文件格式

配置文件采用 JSON 格式，包含以下四个主要部分：

### 1. Model（模型维度配置）
```json
"model": {
  "hidDim": 32,     // 隐藏层维度
  "expDim": 64,     // 扩展维度（FFN中间层）
  "numHeads": 2,    // 注意力头数
  "seqLen": 29      // 序列长度
}
```

### 2. Crypto（同态加密参数配置）
```json
"crypto": {
  "logN": 8,                    // 多项式次数的对数
  "logDefaultScale": 41,        // 默认缩放因子对数
  "logQ": [53, 41, ...],        // 模数链对数
  "logP": [61, 61, 61, 61],     // 辅助模数对数
  "xsH": 192                    // 三元分布参数H
}
```

### 3. Bootstrapping（自举参数配置）
```json
"bootstrapping": {
  "logP": [61, 61, 61, 61]      // 自举辅助模数对数
}
```

### 4. Runtime（运行时配置）
```json
"runtime": {
  "test": "Decoder",    // 测试模块名称
  "level": 16,          // 输入层级
  "btpLevel": 15,       // 自举层级
  "parallel": false     // 是否并行计算
}
```

## 使用方法

### 1. 使用默认配置运行
如果不指定配置文件，程序会自动使用默认配置：
```bash
./cachemir_linear
```

### 2. 指定配置文件运行
```bash
./cachemir_linear -config myconfig.json
```

### 3. 生成默认配置模板
```bash
./cachemir_linear -gen-config default_config.json
```

### 4. 使用命令行 flag 覆盖配置
所有原有的 flag 参数仍可使用，会覆盖配置文件中的对应值：
```bash
./cachemir_linear -config config.json -hidDim 64 -expDim 128 -numHeads 4
```

## 优先级

配置加载优先级（从高到低）：
1. 命令行 flag 参数（显式设置时）
2. 配置文件中的值
3. 内置默认值

## 测试使用配置

测试**默认使用 `config.json`**，也支持通过环境变量 `CACHEMIR_CONFIG` 指定其他配置文件：

```bash
# 默认使用 config.json 运行测试
go test -v -run TestStage1_LinearPipelineMSE

# 使用指定配置文件运行测试
CACHEMIR_CONFIG=myconfig.json go test -v -run TestStage1

# 使用环境变量指定的配置文件
CACHEMIR_CONFIG=/path/to/config.json go test -v -run TestStage1
```

### 配置文件不存在时的行为

如果 `config.json` 不存在且未指定 `CACHEMIR_CONFIG`，测试会自动使用与之前相同的硬编码默认值，以保证测试始终可以运行。

## 向后兼容性

原有代码和测试仍然兼容。`PrepareContext` 函数保留用于基于 flag 的初始化，新增 `PrepareContextWithConfig` 用于基于配置的初始化。
