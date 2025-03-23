# -*- coding: utf-8 -*-
# 声明：本代码仅供学习和研究目的使用。遵守原则同快代理代码

import os
import time
from typing import Dict, List

import httpx
from pydantic import BaseModel, Field

from proxy import IpCache, IpInfoModel, ProxyProvider
from proxy.types import ProviderNameEnum
from tools import utils

class WindScribeProxyModel(BaseModel):
    """
    WindScribe代理响应模型[1](@ref)
    注意：实际字段需根据API文档调整（WindScribe未公开代理API）
    """
    ip: str = Field(..., alias="ip_address")
    port: int = Field(443, alias="proxy_port")
    protocol: str = Field("socks5", alias="proxy_type")
    expire_ts: int = Field(
        default_factory=lambda: int(time.time()) + 7200,  # 默认2小时有效期[3](@ref)
        description="代理过期时间戳"
    )

class WindScribeProxy(ProxyProvider):
    def __init__(
        self, 
        ws_user: str, 
        ws_password: str,
        api_key: str = None
    ):
        """
        WindScribe代理实现[3](@ref)
        :param ws_user: 官网注册的用户名/邮箱
        :param ws_password: 账户密码
        :param api_key: 付费账户专用API密钥（若有）
        """
        self.ws_auth = (ws_user, ws_password)
        self.api_base = "https://api.windscribe.com/"
        self.ip_cache = IpCache()
        self.proxy_brand_name = ProviderNameEnum.WINDSCRIBE_PROVIDER.value
        
        # 认证参数组合策略[1](@ref)
        self.auth_params = {
            "api_key": api_key  # 付费版可能需要
        } if api_key else {
            "user": ws_user,
            "pass": ws_password
        }

    async def _get_api_token(self) -> str:
        """获取OAuth令牌（若接口需要）[1](@ref)"""
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.api_base}/oauth/token",
                data={"grant_type": "client_credentials"},
                auth=self.ws_auth
            )
            return resp.json()["access_token"]

    async def get_proxies(self, num: int) -> List[IpInfoModel]:
        """
        获取WindScribe代理IP列表（需根据实际API调整）[3](@ref)
        注意：WindScribe未公开代理API，此处为模拟实现
        """
        uri = "Serverlist"  # 实际接口路径需确认
        headers = {"Authorization": f"Bearer {await self._get_api_token()}"}

        # 优先读取缓存
        cached_ips = self.ip_cache.load_all_ip(self.proxy_brand_name)
        if len(cached_ips) >= num:
            return cached_ips[:num]

        # 调用API获取新IP
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.api_base}{uri}",
                params={**self.auth_params, "count": num},
                headers=headers
            )
            resp.raise_for_status()
            
            ip_list = [
                WindScribeProxyModel(**item) 
                for item in resp.json()["data"]["proxies"]
            ]

        # 转换为标准模型并缓存
        ip_infos = []
        for proxy in ip_list:
            ip_model = IpInfoModel(
                ip=proxy.ip,
                port=proxy.port,
                protocol=proxy.protocol,
                user=self.ws_auth[0],
                password=self.ws_auth[1],
                expired_time_ts=proxy.expire_ts
            )
            self.ip_cache.set_ip(
                f"{self.proxy_brand_name}_{ip_model.ip}_{ip_model.port}",
                ip_model.model_dump_json(),
                ex=ip_model.expired_time_ts - int(time.time())
            )
            ip_infos.append(ip_model)

        return cached_ips + ip_infos

def new_windscribe_proxy() -> WindScribeProxy:
    """构造WindScribe代理实例"""
    return WindScribeProxy(
        ws_user=os.getenv("WS_USER", "your_windscribe_user"),
        ws_password=os.getenv("WS_PASSWORD", "your_windscribe_pass"),
        api_key=os.getenv("WS_API_KEY")  # 付费账户可能需要
    )