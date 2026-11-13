# 赛道闸门程序化渲染组件

确定性 CPU 软件光栅化管线（无 GPU / 外部资产 / 网络）。详见 [TECHNICAL.md](TECHNICAL.md)。

```bat
python main.py run --seed 7 --segments 6 --out artifacts\run
python main.py edges
python -m unittest discover -s tests -v
```
