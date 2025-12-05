import os
import json
import sys
from pathlib import Path
import traceback

class ConfigManager:
    """配置文件管理器"""
    
    def __init__(self):
        # 获取插件目录
        self.plugin_dir = Path(__file__).parent
        self.config_file = self.plugin_dir / "config.json"
        self.config = self.load_config()
    
    def load_config(self):
        """加载配置文件"""
        default_config = {
            "api_keys": {
                "key_1": "",
                "key_2": "",
                "key_3": ""
            },
            "settings": {
                "polling_interval": 3,
                "max_polling_time": 300
            }
        }
        
        if not self.config_file.exists():
            # 创建默认配置文件
            self.save_config(default_config)
            print(f"ModelScope插件: 已创建默认配置文件: {self.config_file}")
            print(f"请在 {self.config_file} 中填写你的API Key")
            return default_config
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 确保配置包含必要的字段
            if "api_keys" not in config:
                config["api_keys"] = default_config["api_keys"]
            
            if "settings" not in config:
                config["settings"] = default_config["settings"]
            
            # 确保至少有3个key
            for i in range(1, 4):
                key_name = f"key_{i}"
                if key_name not in config["api_keys"]:
                    config["api_keys"][key_name] = ""
            
            return config
            
        except Exception as e:
            print(f"ModelScope插件: 读取配置文件失败: {e}")
            print(f"将使用默认配置")
            return default_config
    
    def save_config(self, config_data):
        """保存配置文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
            self.config = config_data
            return True
        except Exception as e:
            print(f"ModelScope插件: 保存配置文件失败: {e}")
            return False
    
    def get_api_key(self, key_name="key_1"):
        """获取API Key"""
        try:
            return self.config.get("api_keys", {}).get(key_name, "")
        except:
            return ""
    
    def get_all_keys(self):
        """获取所有可用的Key名称"""
        try:
            keys = list(self.config.get("api_keys", {}).keys())
            return [k for k in keys if self.config["api_keys"].get(k, "").strip()]
        except:
            return ["key_1", "key_2", "key_3"]
    
    def get_setting(self, setting_name, default_value):
        """获取设置值"""
        try:
            return self.config.get("settings", {}).get(setting_name, default_value)
        except:
            return default_value
    
    def set_api_key(self, api_key, key_name="key_1"):
        """设置API Key"""
        if "api_keys" not in self.config:
            self.config["api_keys"] = {}
        
        self.config["api_keys"][key_name] = api_key
        return self.save_config(self.config)

# 全局配置管理器实例
config_manager = None

def get_config_manager():
    """获取配置管理器实例"""
    global config_manager
    if config_manager is None:
        config_manager = ConfigManager()
    return config_manager