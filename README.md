<div align="center">   
  
# End-to-End Autonomous Driving through V2X Cooperation
</div>

<h3 align="center">
  <a href="https://arxiv.org/abs/2404.00717">arXiv</a> |
  Demo |
  <a href="https://github.com/AIR-THU/UniV2X">Code</a>
</h3>

![teaser](assets/UniV2X-Framework.png)

<br><br>

## Table of Contents:
1. [Highlights](#high)
2. [News](#news)
3. [Getting Started](#getting-started)
    - [Installation](docs/INSTALL.md)
    - [Prepare Dataset](docs/DATA_PREP.md)
    - [Train/Eval](docs/TRAIN_EVAL.md)
4. [TODO List](#todos)
5. [License](#license)
6. [Citation](#citation)

## Highlights <a name="high"></a>

- UniV2X is the first cooperative autonomous driving framework that seamlessly integrates all key driving modules across diverse driving views into a unified network. 

## News <a name="news"></a>

- **`2025/01/26`** Code & model initial release `v1.0`.
- **`2024/12/10`** UniV2X is accepted by AAAI 2025.
- **`2024/04/02`** UniV2X [paper](https://arxiv.org/abs/2404.00717) is available on arXiv.


## Getting Started <a name="getting-started"></a>

- [Installation](docs/INSTALL.md)
- [Prepare Dataset](docs/DATA_PREP.md)
- [Evaluation Example](docs/TRAIN_EVAL.md)
- [Train/Eval](docs/TRAIN_EVAL.md)


## TODO List <a name="todos"></a>
- [x] Base-model code release (on DAIR-V2X)
- [x] Base-model configs & checkpoints (on DAIR-V2X)
- [ ] Benchmark code release (on DAIR-V2X)
- [ ] Benchmark configs & checkpoints (on DAIR-V2X)
- [ ] Support more datasets


## License <a name="license"></a>

All assets and code are under the [Apache 2.0 license](./LICENSE) unless specified otherwise.

## Citation <a name="citation"></a>

Please consider citing our paper if the project helps your research with the following BibTex:

```bibtex
@inproceedings{yu2024_univ2x,
 title={End-to-End Autonomous Driving through V2X Cooperation}, 
 author={Haibao Yu and Wenxian Yang and Jiaru Zhong and Zhenwei Yang and Siqi Fan and Ping Luo and Zaiqing Nie},
 booktitle={The 39th Annual AAAI Conference on Artificial Intelligence},
 year={2025}
}
```

## Related resources

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
- [DAIR-V2X](https://github.com/AIR-THU/DAIR-V2X) (:rocket:Ours!)
- [V2X-Seq](https://github.com/AIR-THU/DAIR-V2X-Seq) (:rocket:Ours!)
- [FFNET](https://github.com/haibao-yu/FFNet-VIC3D) (:rocket:Ours!)
- [UniAD](https://github.com/OpenDriveLab/UniAD)