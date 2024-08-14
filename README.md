This is the source code of Project **Compressing Pipeline for implicit neural representation**

## Build & run instructions:

First, put `Synthetic_NeRF` to `your_path_to/code/`

- `cd your_path_to/code/NeRF`
- `conda create -n Neural_compression python=3.9`
- `conda activate Neural_compression`
- `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r  requirements.txt` 

Once the requirements are installed, within the conda environment:
```
cd ..
git clone https://github.com/UCMerced-ML/LC-model-compression
pip install -e ./LC-model-compression
cd NeRF
```

To generate the baseline mode:

- `python main.py` (Please refer to `param.py` for parameters)
  
To evaluate:

- `python eval.py` (for generating results)

To Compress:

- `python low_rank.py` (for generating the Fixed-rank model)
- `python auto_rank.py` (for generating the Auto-rank model)
- `python quantization.py` (for generating the quantized model)

Also, you need to change the datasets' path for different scenes in param.py (--test_model)



# **Compressing Pipeline for implicit neural representation**

## **Team Member**
* Xuanhang Diao 
* Letong Han  
* Yifan Chen 


## **Summary**
Deep neural networks have demonstrated impressive capabilities in many domains. However, as deep learning models become increasingly complex, 
they require hardware with increasingly demanding performance and specifications. 
While various deep learning model compression methods have been widely discussed, 
recent developments in deep learning, particularly in neural rendering and large language models, have raised the bar for the effectiveness, 
speed, and robustness of compression methods. To address this, 
we propose a deep learning model compression pipeline that combines multiple compression strategies and Graphics Processing Unit (GPU) based parallel acceleration for downstream tasks in neural rendering. Our approach outperforms traditional single-strategy serial compression algorithms in both speed and effect.


