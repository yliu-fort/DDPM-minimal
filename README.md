# diffusion-sandbox

一个最小但工业风的 Diffusion 学习模板：YAML 配置、确定性、追踪与日志、单元测试、类型检查与风格工具、可视化示例齐全。

## 快速开始
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m unittest
python -m diffusion_sandbox.train --config configs/default.yaml
tensorboard --logdir runs
```

## 配置
修改 `configs/default.yaml` 中的数据集/模型/扩散/训练/追踪参数。

## 风格与类型
- 格式化: `black . && isort .`
- 静态检查: `ruff check .`
- 类型检查: `mypy src tests`

## 结构
详见仓库目录与源码内 docstring。
