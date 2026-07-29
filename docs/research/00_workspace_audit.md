# 工作区审计

| 路径 | 结论 |
|---|---|
| `/Users/nick/programs/eeg-system/bishe` | 可复用 Schema 校验、质量、OOF/RAG、Trace、Integrity；不复用硬编码主流程 |
| `/Users/nick/programs/twobro/paicli-main` | 借鉴 Registry、DAG、Budget、Provider；不跨语言集成 |
| `/Users/nick/eeg/迁移/daima/autodl_latest_data` | 特征 112 行、5 个 OOF 输出、Stage1/2、离线 KG；模型只可受限回放 |
| `eegsys-bottom-main` | 具备 EDF/MAT、PSD、bandpower；服务层耦合 HDFS/RabbitMQ |
| Vue/Java 工程 | 数据管理和 UI，不包含可直接复用的 ASD 推理 |

未发现真实 EDF/BDF/SET/FIF；111 个 MAT 位于第三方测试数据目录。

