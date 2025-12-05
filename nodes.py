import torch
import numpy as np
from PIL import Image
import io
import base64
import json
import requests
import time
import os
import glob
from typing import Dict, List, Any, Optional
import folder_paths
from pathlib import Path

# 导入配置管理器
try:
    from .config_manager import get_config_manager
    config_manager = get_config_manager()
except ImportError:
    print("ModelScope插件: 无法导入配置管理器，将使用默认配置")
    config_manager = None

def tensor_to_pil(tensor_image):
    """将tensor图像转换为PIL Image"""
    # 转换tensor为numpy数组
    image_np = tensor_image.cpu().numpy()
    
    # 确保数值范围在0-255之间
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = image_np.astype(np.uint8)
    
    # 处理批次维度
    if len(image_np.shape) == 4:
        image_np = image_np[0]  # 取批次中的第一张
    
    # 转换通道顺序（如果必要）
    if image_np.shape[2] == 4:  # RGBA
        image = Image.fromarray(image_np[:, :, :3])  # 只取RGB
    elif image_np.shape[2] == 3:  # RGB
        image = Image.fromarray(image_np)
    else:
        # 处理灰度图
        image = Image.fromarray(image_np[:, :, 0])
    
    return image

def tensor_to_base64(tensor_image):
    """将tensor图像转换为base64编码"""
    image = tensor_to_pil(tensor_image)
    
    # 转换为base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=95)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return f"data:image/jpeg;base64,{img_str}"

def pil_to_tensor(image):
    """将PIL Image转换为tensor"""
    image_np = np.array(image).astype(np.float32) / 255.0
    if len(image_np.shape) == 2:
        image_np = np.stack([image_np] * 3, axis=2)
    elif image_np.shape[2] == 4:
        image_np = image_np[:, :, :3]
    
    image_np = np.expand_dims(image_np, axis=0)
    return torch.from_numpy(image_np)

def get_available_keys():
    """获取可用的API Key列表"""
    if config_manager:
        available_keys = config_manager.get_all_keys()
        if not available_keys:
            available_keys = ["key_1", "key_2", "key_3"]
    else:
        available_keys = ["key_1", "key_2", "key_3"]
    return available_keys

class ModelScopeImageAnalysis:
    """
    魔搭平台图像分析节点（视觉推理）
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key_name": (get_available_keys(),),
                "prompt": ("STRING", {
                    "default": "详细描述这幅图片的内容，包括场景、物体、人物特征等",
                    "multiline": True
                }),
                "model_name": (["Qwen/Qwen3-VL-8B-Instruct", 
                               "Qwen/Qwen3-VL-30B-A3B-Instruct", 
                               "Qwen/Qwen3-VL-235B-A22B-Instruct"],),
                "max_tokens": ("INT", {
                    "default": 1024,
                    "min": 1,
                    "max": 4096,
                    "step": 64
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "default": "你是一个专业的图像分析助手，请详细描述图像内容。",
                    "multiline": True
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text_output",)
    FUNCTION = "analyze_image"
    CATEGORY = "ModelScope/视觉推理"
    OUTPUT_NODE = True
    
    def analyze_image(self, image, api_key_name, prompt, model_name, max_tokens, temperature, system_prompt=""):
        """分析图像并返回文本描述"""
        
        try:
            # 获取API Key
            if config_manager:
                api_key = config_manager.get_api_key(api_key_name)
            else:
                api_key = ""
            
            # 检查API Key是否填写
            if not api_key or api_key.strip() == "":
                error_msg = f"错误：配置文件中未找到API Key '{api_key_name}'\n\n请按以下步骤操作：\n1. 打开插件目录下的 config.json 文件\n2. 在 'api_keys' 部分添加你的API Key\n3. 格式：\"{api_key_name}\": \"你的API Key\""
                print(f"ModelScope图像分析错误: {error_msg}")
                return (error_msg,)
            
            # 转换图像为base64
            image_base64 = tensor_to_base64(image)
            
            # 构建请求
            from openai import OpenAI
            client = OpenAI(
                base_url='https://api-inference.modelscope.cn/v1',
                api_key=api_key.strip(),
            )
            
            # 构建消息
            messages = []
            
            # 添加系统提示（如果有）
            if system_prompt and system_prompt.strip():
                messages.append({
                    "role": "system",
                    "content": system_prompt.strip()
                })
            
            # 添加用户消息（图像+文本）
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_base64
                        }
                    }
                ]
            })
            
            # 调用API
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False
            )
            
            # 提取响应文本
            text_output = response.choices[0].message.content
            
            return (text_output,)
            
        except Exception as e:
            error_msg = f"图像分析失败: {str(e)}"
            print(f"ModelScope图像分析错误: {error_msg}")
            import traceback
            traceback.print_exc()
            return (error_msg,)

class ModelScopeBatchAnalyze:
    """
    批量图像分析节点
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "api_key_name": (get_available_keys(),),
                "prompt": ("STRING", {
                    "default": "详细描述这幅图片的内容，包括场景、物体、人物特征等",
                    "multiline": True
                }),
                "model_name": (["Qwen/Qwen3-VL-8B-Instruct", 
                               "Qwen/Qwen3-VL-30B-A3B-Instruct", 
                               "Qwen/Qwen3-VL-235B-A22B-Instruct"],),
                "max_tokens": ("INT", {
                    "default": 1024,
                    "min": 1,
                    "max": 4096,
                    "step": 64
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "default": "你是一个专业的图像分析助手，请详细描述图像内容。",
                    "multiline": True
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text_outputs",)
    FUNCTION = "analyze_batch"
    CATEGORY = "ModelScope/视觉推理"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)
    
    def analyze_batch(self, images, api_key_name, prompt, model_name, max_tokens, temperature, system_prompt=""):
        """批量分析图像"""
        
        try:
            # 获取API Key
            if config_manager:
                api_key = config_manager.get_api_key(api_key_name)
            else:
                api_key = ""
            
            # 检查API Key是否填写
            if not api_key or api_key.strip() == "":
                error_msg = f"错误：配置文件中未找到API Key '{api_key_name}'"
                return ([error_msg],)
            
            # 初始化OpenAI客户端
            from openai import OpenAI
            client = OpenAI(
                base_url='https://api-inference.modelscope.cn/v1',
                api_key=api_key.strip(),
            )
            
            results = []
            
            # 处理每张图像
            for i in range(images.shape[0]):
                try:
                    print(f"ModelScope批量分析: 处理第 {i+1}/{images.shape[0]} 张图片")
                    
                    # 提取单张图像
                    single_image = images[i:i+1]
                    
                    # 转换图像为base64
                    image_base64 = tensor_to_base64(single_image)
                    
                    # 构建消息
                    messages = []
                    
                    # 添加系统提示（如果有）
                    if system_prompt and system_prompt.strip():
                        messages.append({
                            "role": "system",
                            "content": system_prompt.strip()
                        })
                    
                    # 添加用户消息（图像+文本）
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_base64
                                }
                            }
                        ]
                    })
                    
                    # 调用API
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=False
                    )
                    
                    # 提取响应文本
                    text_output = response.choices[0].message.content
                    results.append(text_output)
                    
                except Exception as e:
                    error_msg = f"图像{i+1}分析失败: {str(e)}"
                    print(f"ModelScope批量分析错误: {error_msg}")
                    results.append(error_msg)
            
            print(f"ModelScope批量分析: 完成 {len(results)} 张图片的分析")
            return (results,)
            
        except Exception as e:
            error_msg = f"批量分析失败: {str(e)}"
            print(f"ModelScope批量分析错误: {error_msg}")
            import traceback
            traceback.print_exc()
            return ([error_msg],)

class ModelScopeImageGeneration:
    """
    Qwen-Image图像生成节点
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key_name": (get_available_keys(),),
                "prompt": ("STRING", {
                    "default": "一只金色的猫",
                    "multiline": True
                }),
                "model_name": (["Qwen/Qwen-Image"],),
                "width": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 2048,
                    "step": 64
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 2048,
                    "step": 64
                }),
                "num_images": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                    "step": 1
                }),
            },
            "optional": {
                "lora_config": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "LoRA配置（可选）\n格式：{\"lora_repo_id\": 0.6, \"another_lora\": 0.4}"
                }),
                "negative_prompt": ("STRING", {
                    "default": "",
                    "multiline": True
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2**32-1,
                    "step": 1
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate_image"
    CATEGORY = "ModelScope/图像生成"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)
    
    def generate_image(self, api_key_name, prompt, model_name, width, height, num_images, 
                      lora_config="", negative_prompt="", seed=-1):
        """生成图像"""
        
        try:
            # 获取API Key
            if config_manager:
                api_key = config_manager.get_api_key(api_key_name)
                polling_interval = config_manager.get_setting("polling_interval", 3)
                max_polling_time = config_manager.get_setting("max_polling_time", 300)
            else:
                api_key = ""
                polling_interval = 3
                max_polling_time = 300
            
            # 检查API Key是否填写
            if not api_key or api_key.strip() == "":
                error_msg = f"错误：配置文件中未找到API Key '{api_key_name}'"
                print(f"ModelScope图像生成错误: {error_msg}")
                # 返回黑色图像作为错误指示
                error_image = torch.zeros((1, height, width, 3))
                return ([error_image],)
            
            base_url = 'https://api-inference.modelscope.cn/'
            common_headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            
            # 构建请求数据
            data = {
                "model": model_name,
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_images_per_prompt": num_images,
            }
            
            # 添加可选参数
            if negative_prompt and negative_prompt.strip():
                data["negative_prompt"] = negative_prompt
            
            if seed != -1:
                data["seed"] = seed
            
            # 处理LoRA配置
            if lora_config and lora_config.strip():
                try:
                    lora_data = json.loads(lora_config)
                    if isinstance(lora_data, dict):
                        data["loras"] = lora_data
                    elif isinstance(lora_data, str):
                        data["loras"] = lora_data
                except:
                    print(f"ModelScope图像生成: LoRA配置解析失败，忽略LoRA设置")
            
            print(f"ModelScope图像生成: 发送请求，参数: {json.dumps(data, ensure_ascii=False)}")
            
            # 发送异步请求
            response = requests.post(
                f"{base_url}v1/images/generations",
                headers={**common_headers, "X-ModelScope-Async-Mode": "true"},
                data=json.dumps(data, ensure_ascii=False).encode('utf-8')
            )
            
            response.raise_for_status()
            task_id = response.json()["task_id"]
            
            print(f"ModelScope图像生成: 任务已提交，任务ID: {task_id}")
            
            # 轮询任务状态
            start_time = time.time()
            images = []
            
            while time.time() - start_time < max_polling_time:
                result = requests.get(
                    f"{base_url}v1/tasks/{task_id}",
                    headers={**common_headers, "X-ModelScope-Task-Type": "image_generation"},
                )
                result.raise_for_status()
                data = result.json()
                
                task_status = data.get("task_status", "UNKNOWN")
                
                if task_status == "SUCCEED":
                    # 下载所有生成的图像
                    output_images = data.get("output_images", [])
                    
                    for img_url in output_images:
                        img_response = requests.get(img_url)
                        img_response.raise_for_status()
                        
                        # 转换为PIL图像
                        image = Image.open(io.BytesIO(img_response.content))
                        
                        # 转换为tensor
                        image_tensor = pil_to_tensor(image)
                        images.append(image_tensor)
                    
                    print(f"ModelScope图像生成: 成功生成 {len(images)} 张图像")
                    break
                    
                elif task_status == "FAILED":
                    error_msg = "图像生成失败"
                    error_detail = data.get("error_message", "未知错误")
                    print(f"ModelScope图像生成错误: {error_msg} - {error_detail}")
                    # 返回黑色图像作为错误指示
                    error_image = torch.zeros((1, height, width, 3))
                    return ([error_image],)
                
                elif task_status in ["QUEUED", "RUNNING"]:
                    print(f"ModelScope图像生成: 任务状态: {task_status}, 等待 {polling_interval} 秒后重试...")
                    time.sleep(polling_interval)
                    continue
                
                else:
                    print(f"ModelScope图像生成: 未知任务状态: {task_status}")
                    time.sleep(polling_interval)
                    continue
            
            else:
                # 超时
                error_msg = f"图像生成超时（{max_polling_time}秒）"
                print(f"ModelScope图像生成错误: {error_msg}")
                error_image = torch.zeros((1, height, width, 3))
                return ([error_image],)
            
            if not images:
                error_msg = "未生成任何图像"
                print(f"ModelScope图像生成错误: {error_msg}")
                error_image = torch.zeros((1, height, width, 3))
                return ([error_image],)
            
            return (images,)
            
        except Exception as e:
            error_msg = f"图像生成失败: {str(e)}"
            print(f"ModelScope图像生成错误: {error_msg}")
            import traceback
            traceback.print_exc()
            # 返回黑色图像作为错误指示
            error_image = torch.zeros((1, height, width, 3))
            return ([error_image],)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "ModelScopeImageAnalysis": ModelScopeImageAnalysis,
    "ModelScopeBatchAnalyze": ModelScopeBatchAnalyze,
    "ModelScopeImageGeneration": ModelScopeImageGeneration,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelScopeImageAnalysis": "魔搭图像反推 (Qwen3-VL)",
    "ModelScopeBatchAnalyze": "魔搭图像批量反推 (Qwen3-VL)",
    "ModelScopeImageGeneration": "魔搭图像生成 (Qwen-Image)",
}