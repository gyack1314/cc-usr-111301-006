# 卫星定位RINEX观测的历元与动态头采集

此项目保存科学资料交换的原始文件、清单及冻结HTTP记录。`capture_data.io`读取JSON、核对资产身份，提供固定字节范围读取和顺序响应重放；`capture_data.domain`只核对资料类别、路径和数量，`capture_data.cli`输出概况。`capture_data.rinex3`在已校验字节上采集RINEX 3.04观测文件：固定列解释头部与历元、按系统观测目录定长读取16字符观测槽、逐历元标注目录版本与事件上下文，并按冻结闰秒表把GPS/UTC归算到TAI整数纳秒轴；`capture_data.rinex_cli`输出结构化JSON。[资料约定](docs/data-contract.md)说明时间、字段与格式范围。所有站点、观测和机构资料均为合成，不连接航天器、接收机或真实研究系统。

Python3.11以上，无第三方Python依赖。项目根目录运行：

```sh
python -m unittest discover -s tests -v
python -m compileall -q capture_data tests
python -m capture_data.cli inspect data/normal.json
python -m capture_data.rinex_cli rinex-manifest data/normal.json
python -m capture_data.rinex_cli rinex-file data/utc-leap.rnx --leap data/utc-positive-leap.json
```

`inspect`不解文件格式或选择HTTP版本，只显示资产身份及数量；`invalid-identity.json`有预期资产摘要错误，其余清单有合法来源身份但内容可含需领域操作识别的异常。`Source.read_at`记录请求范围，`Replay.next`记录实际method、URL与headers；它不替调用者完成协议或缓存策略。

RINEX采集器只处理UTF-8兼容ASCII的3.04观测文件，不读导航电文、不做定位求解或物理改正：观测值/LLI/SSI分开保留、空槽为null而非0、D/E指数等价且非法字符报错；flag 2/3/5事件行不计为卫星，flag 4头更新整批验证通过才原子切换目录版本（失败保留旧版本），flag 6周跳单独标记；接收机钟偏单列归档不再二次校正；UTC闰秒仅按清单内冻结表转换，表覆盖不到的日期标注`unknown_leap`，不使用本机时区。每条观测带系统、观测码、原槽字节范围与当时头版本，支持在完整历元边界checkpoint/resume，缺测位置可由目录与实际槽之差还原。

Dockerfile固定Linux基础镜像，`docker build -t science-material .`构建；`docker run --rm science-material`读取概况，`docker run --rm science-material python -m unittest discover -s tests -v`运行已有检查。资产均可离线读取，不需要下载数据或真实账号。
