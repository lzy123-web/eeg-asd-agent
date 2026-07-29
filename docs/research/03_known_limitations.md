# Known Limitations

- 没有真实受试者 EEG，原始链路仅使用合成信号。
- 师兄 OOF/LoRA 数据存在划分、重叠和泛化风险。
- Stage1/Stage2 大模型没有在本机 GPU 上重新验证。
- 离线 KG 是项目资产，不是正式临床指南。
- 当前 Provider 是 fixture；Benchmark 不支持真实 LLM 效果主张。
- Responses Provider 已实现但未在本轮使用真实密钥；接口通过 mock 测试。
- 规则式 IntentParser 适合受控实验集，不等价于开放域语言理解。
- 系统不能用于临床确诊、排除诊断、处方或自动干预。
