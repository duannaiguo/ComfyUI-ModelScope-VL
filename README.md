# ComfyUI ModelScope 插件

这是一个用于在ComfyUI中调用魔搭平台视觉模型的插件。

## 功能特性

- 🖼️ 单图图像反推（视觉推理）
- 🔄 批量图像处理（打标推理）
- 🎨 图像生成 (Qwen-Image)

## 安装方法

### 方法一：直接安装
1. 将本插件文件夹放到 `ComfyUI/custom_nodes/` 目录下
2. 去插件目录下填入你的魔塔API KEY，没有就去 https://modelscope.cn 注册个号，每天有免费的2000次调用，单个模型是500次，轻度使用是够了的，不够就多注册两个号，有提供3个KEY的调用选项
3. 重启ComfyUI

### 方法二：使用Git
```bash
cd ComfyUI/custom_nodes

git clone https://github.com/duannaiguo/ComfyUI-ModelScope-VL.git

