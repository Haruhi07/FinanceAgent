"""
智能代理控制
==============

解决问题: clash 代理对 push2.eastmoney.com 等域名的兼容性问题。

策略:
- 默认:环境变量里有什么用什么
- 加 --no-proxy:禁用代理,但白名单内的域名走代理(其他直连)

实现:
1. Monkey-patch requests.get/post/request(模块级 + Session 级)
2. 根据 URL 域名决定走代理还是直连
3. 白名单在 config.PROXY_WHITELIST 配置

用法(在 main.py 入口处):
    from tools.disable_proxy import disable_proxy
    disable_proxy()
"""

from __future__ import annotations

import os
import logging
from urllib.parse import urlparse

import config

logger = logging.getLogger(__name__)


def _should_use_proxy(url: str, whitelist: set) -> bool:
    """URL 域名是否在白名单内(走代理)"""
    if not url:
        return False
    try:
        host = urlparse(url).hostname
        if not host:
            return False
        host_lower = host.lower()
        for domain in whitelist:
            if host_lower == domain or host_lower.endswith("." + domain):
                return True
        return False
    except Exception:
        return False


def _get_proxies_from_env() -> dict:
    """从环境变量获取代理配置"""
    http = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    https = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    return {"http": http, "https": https}


def _select_proxies(url: str, whitelist: set) -> dict:
    """根据 URL 决定走代理还是直连"""
    if _should_use_proxy(url, whitelist):
        # 白名单内:用代理
        return _get_proxies_from_env()
    # 白名单外:直连
    return {"http": None, "https": None}


def disable_proxy(whitelist: list | None = None) -> None:
    """禁用代理,但白名单内的域名仍走代理
    
    Args:
        whitelist: 走代理的域名后缀列表(如 ["push2.eastmoney.com"])
                  默认用 config.PROXY_WHITELIST
    """
    whitelist = whitelist or config.PROXY_WHITELIST
    whitelist_set = set(whitelist)
    
    logger.info(f"代理白名单: {whitelist_set}")
    
    # 1. Patch urllib(以防某些库用 urllib)
    try:
        import urllib.request
        _orig_proxy_bypass = urllib.request.proxy_bypass
        def _patched_proxy_bypass(host):
            if any(host == d or host.endswith("." + d) for d in whitelist_set):
                return False
            return True
        urllib.request.proxy_bypass = _patched_proxy_bypass
    except Exception as e:
        logger.warning(f"failed to patch urllib: {e}")
    
    # 2. Patch requests(akshare 主要用)
    try:
        import requests
        
        # 2a. Session.request
        _original_session_request = requests.Session.request
        def _patched_session_request(self, method, url, **kwargs):
            if "proxies" not in kwargs:
                kwargs["proxies"] = _select_proxies(url, whitelist_set)
            return _original_session_request(self, method, url, **kwargs)
        _patched_session_request.__name__ = "patched_session_request"
        requests.Session.request = _patched_session_request
        
        # 2b. 模块级
        _original_module_get = requests.get
        _original_module_post = requests.post
        _original_module_request = requests.request
        
        def _patched_module_get(url, **kwargs):
            if "proxies" not in kwargs:
                kwargs["proxies"] = _select_proxies(url, whitelist_set)
            return _original_module_get(url, **kwargs)
        
        def _patched_module_post(url, **kwargs):
            if "proxies" not in kwargs:
                kwargs["proxies"] = _select_proxies(url, whitelist_set)
            return _original_module_post(url, **kwargs)
        
        def _patched_module_request(method, url=None, **kwargs):
            if "proxies" not in kwargs:
                kwargs["proxies"] = _select_proxies(url, whitelist_set)
            return _original_module_request(method, url, **kwargs)
        
        _patched_module_get.__name__ = "patched_module_get"
        _patched_module_post.__name__ = "patched_module_post"
        _patched_module_request.__name__ = "patched_module_request"
        
        requests.get = _patched_module_get
        requests.post = _patched_module_post
        requests.request = _patched_module_request
        
        logger.debug("patched requests (Session.request + module-level)")
    except Exception as e:
        logger.warning(f"failed to patch requests: {e}")
    
    # 3. Patch httpx
    try:
        import httpx
        _orig_httpx_init = httpx.Client.__init__
        def _patched_httpx_init(self, *args, **kwargs):
            kwargs["trust_env"] = True
            return _orig_httpx_init(self, *args, **kwargs)
        httpx.Client.__init__ = _patched_httpx_init
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"failed to patch httpx: {e}")
    
    logger.info(
        f"代理控制已启用:白名单内({len(whitelist_set)} 个域名)走代理,"
        f"其他域名走直连"
    )
