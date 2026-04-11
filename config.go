package main

import (
	"encoding/json"
	"fmt"
	"os"
)

// ModelConfig 定义模型维度配置
type ModelConfig struct {
	HiddenDim  int `json:"hidDim"`   // 隐藏层维度
	ExpandedDim int `json:"expDim"`  // 扩展维度 (FFN中间层)
	NumHeads   int `json:"numHeads"`  // 注意力头数
	SeqLen     int `json:"seqLen"`    // 序列长度
}

// CryptoConfig 定义同态加密参数配置
type CryptoConfig struct {
	LogN            int    `json:"logN"`            // 多项式次数的对数
	LogDefaultScale int    `json:"logDefaultScale"` // 默认缩放因子对数
	LogQ            []int  `json:"logQ"`            // 模数链对数
	LogP            []int  `json:"logP"`            // 辅助模数对数
	XsH             int    `json:"xsH"`             // 三元分布参数H
}

// BootstrappingConfig 定义自举参数配置
type BootstrappingConfig struct {
	LogP []int `json:"logP"` // 自举辅助模数对数
}

// RuntimeConfig 定义运行时配置
type RuntimeConfig struct {
	TestModule string `json:"test"`      // 测试模块名称
	Level      int    `json:"level"`     // 输入层级
	BtpLevel   int    `json:"btpLevel"`  // 自举层级
	Parallel   bool   `json:"parallel"`  // 是否并行计算
}

// Config 是完整的应用配置
type Config struct {
	Model         ModelConfig         `json:"model"`
	Crypto        CryptoConfig        `json:"crypto"`
	Bootstrapping BootstrappingConfig `json:"bootstrapping"`
	Runtime       RuntimeConfig       `json:"runtime"`
}

// DefaultConfig 返回默认配置
func DefaultConfig() *Config {
	return &Config{
		Model: ModelConfig{
			HiddenDim:   32,
			ExpandedDim: 64,
			NumHeads:    2,
			SeqLen:      29,
		},
		Crypto: CryptoConfig{
			LogN:            8,
			LogDefaultScale: 41,
			LogQ:            []int{53, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41},
			LogP:            []int{61, 61, 61, 61},
			XsH:             192,
		},
		Bootstrapping: BootstrappingConfig{
			LogP: []int{61, 61, 61, 61},
		},
		Runtime: RuntimeConfig{
			TestModule: "Decoder",
			Level:      16,
			BtpLevel:   15,
			Parallel:   false,
		},
	}
}

// LoadConfig 从文件加载配置
// 如果文件不存在，返回默认配置
// 如果文件存在但部分字段缺失，使用默认值填充
func LoadConfig(path string) (*Config, error) {
	config := DefaultConfig()

	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			// 配置文件不存在，返回默认配置
			fmt.Printf("Config file not found at %s, using default configuration\n", path)
			return config, nil
		}
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	// 解析JSON，使用默认值作为基础，文件内容覆盖
	if err := json.Unmarshal(data, config); err != nil {
		return nil, fmt.Errorf("failed to parse config file: %w", err)
	}

	return config, nil
}

// SaveConfig 保存配置到文件(便于生成模板)
func SaveConfig(path string, config *Config) error {
	data, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}

	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("failed to write config file: %w", err)
	}

	return nil
}

// ToLlamaSize 将ModelConfig转换为LlamaSize
func (mc *ModelConfig) ToLlamaSize() *LlamaSize {
	return &LlamaSize{
		hidDim:   mc.HiddenDim,
		expDim:   mc.ExpandedDim,
		numHeads: mc.NumHeads,
		seqLen:   mc.SeqLen,
	}
}

// MergeWithFlags 用命令行flag值覆盖配置值(如果flag被显式设置)
type FlagOverrides struct {
	LogN     *int
	HidDim   *int
	ExpDim   *int
	SeqLen   *int
	NumHeads *int
	Level    *int
	BtpLevel *int
	Test     *string
	Parallel *bool
}

// ApplyOverrides 应用命令行flag覆盖
func (c *Config) ApplyOverrides(overrides *FlagOverrides) {
	if overrides.LogN != nil && *overrides.LogN != 0 {
		c.Crypto.LogN = *overrides.LogN
	}
	if overrides.HidDim != nil && *overrides.HidDim != 0 {
		c.Model.HiddenDim = *overrides.HidDim
	}
	if overrides.ExpDim != nil && *overrides.ExpDim != 0 {
		c.Model.ExpandedDim = *overrides.ExpDim
	}
	if overrides.SeqLen != nil && *overrides.SeqLen != 0 {
		c.Model.SeqLen = *overrides.SeqLen
	}
	if overrides.NumHeads != nil && *overrides.NumHeads != 0 {
		c.Model.NumHeads = *overrides.NumHeads
	}
	if overrides.Level != nil && *overrides.Level != 0 {
		c.Runtime.Level = *overrides.Level
	}
	if overrides.BtpLevel != nil && *overrides.BtpLevel != 0 {
		c.Runtime.BtpLevel = *overrides.BtpLevel
	}
	if overrides.Test != nil && *overrides.Test != "" {
		c.Runtime.TestModule = *overrides.Test
	}
	if overrides.Parallel != nil {
		c.Runtime.Parallel = *overrides.Parallel
	}
}
