"""
全局配置
========

所有可调参数集中在这里,方便不同场景调优。
"""

from pathlib import Path

# ============ 路径 ============
BASE_DIR = Path(__file__).parent
CACHE_DIR = BASE_DIR / "cache"
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
ARTICLES_DIR = OUTPUT_DIR / "articles"
REPORTS_DIR = OUTPUT_DIR / "reports"
LOGS_DIR = OUTPUT_DIR / "logs"
ANOMALIES_DIR = OUTPUT_DIR / "anomalies"   # 2026-08 新增:Detector 异常存这里
CANDIDATES_DIR = OUTPUT_DIR / "candidates" # 2026-08 新增:Hotspot 候选存这里
BRIEFS_DIR = OUTPUT_DIR / "briefs"         # 2026-08-04 新增:Researcher 简报存这里

# 自动建目录
for d in [CACHE_DIR, DATA_DIR, ARTICLES_DIR, REPORTS_DIR, LOGS_DIR, ANOMALIES_DIR, CANDIDATES_DIR, BRIEFS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============ Agent 行为参数 ============
# Scanner 配置(2026-08 改造)
SCANNER_SCAN_DAYS = 3                  # 扫最近 N 个工作日

# 异动检测阈值
ANOMALY_ZSCORE_THRESHOLD = 2.5          # 涨跌幅 Z-score 阈值
ANOMALY_VOLUME_RATIO_THRESHOLD = 2.0    # 量比阈值
ANOMALY_PRICE_CHANGE_THRESHOLD = 0.05   # 涨跌幅阈值(5%)

# 热点评分阈值
HOTSPOT_MIN_SCORE = 0.55                 # 进入研究环节的最低分
HOTSPOT_HIGH_CONF = 0.75                 # 高确信度,直接写稿
HOTSPOT_MID_CONF = 0.55                  # 中等,人工确认
HOTSPOT_LOW_CONF = 0.35                  # 低,持续观察

# 评分权重(对应方案中的 6 维特征向量,Lead Time 已移除)
WEIGHTS = {
    # 2026-08-06:Lead Time 维度移除(它应该是 evaluation 指标,不是建模维度)
    # 重新归一化使权重和 = 1.0
    "intensity": 0.22,
    "persistence": 0.11,
    "virality": 0.11,
    "novelty": 0.17,
    "relevance": 0.17,
    "value_density": 0.22,
}

# ============ 缓存 ============
CACHE_TTL_SECONDS = {
    "realtime": 60,           # 实时数据 1 分钟
    "intraday": 300,          # 日内数据 5 分钟
    "daily": 3600 * 4,        # 日级数据 4 小时
    "fundamental": 3600 * 24, # 基本面 1 天
}

# ============ 网络 ============
REQUEST_TIMEOUT = 15
REQUEST_MAX_RETRY = 1       # 减少重试节省时间(网络失败时直接放弃该工具)
REQUEST_RETRY_DELAY = 1.0   # 秒

# Proxy 白名单:这些域名强制走代理(其他域名走直连)
# 原因:这些域名直连不通(clash 代理能代理)
PROXY_WHITELIST = [
    "push2.eastmoney.com",
    "82.push2.eastmoney.com",
    "17.push2.eastmoney.com",
    "datacenter-web.eastmoney.com",
    "datacenter.eastmoney.com",
    "data.eastmoney.com",
    "vip.stock.finance.sina.com.cn",
    "money.finance.sina.com.cn",
    "hq.sinajs.cn",
    "qt.gtimg.cn",
    "web.ifzq.gtimg.cn",
]

# 已知不可用接口(在某些网络环境会失败,Scanner 跳过这些)
# 这些接口用的是 push2.* 数字前缀子域,在用户网络环境 GET 请求被关闭连接
DISABLED_TOOLS = {
    "get_sector_fund_flow",        # 板块资金流 - 用 push2.eastmoney.com
    "get_individual_fund_flow",    # 个股资金流 - 用 push2.eastmoney.com
    "get_individual_info",         # 个股信息 - 用 push2.eastmoney.com
    "get_zh_a_spot",               # 全 A 实时行情 - 用 push2.eastmoney.com
    "get_zh_a_hist",               # 历史 K 线 - 用 push2.eastmoney.com
    # get_financial_report 已解禁:走 sina 数据源,实测稳定可用(2026-08-03)
    "get_research_report",         # 个股研报 - akshare 1.18 KeyError bug,等升级
}

# ============ LLM (DeepSeek) ============
# 接入 DeepSeek(OpenAI 兼容协议)
# 申请 API key: https://platform.deepseek.com/api_keys
#
# ⚠️ 不要把 API key 直接写在 config.py 里!走环境变量:
#   export DEEPSEEK_API_KEY="sk-xxx"
#   (兼容: 也支持 DEEPSEEK_API 作为环境变量名)
#
# 留空 → tools.llm_client._get_api_key() 自动从 env 读
# is_llm_available() 在没有 key 时返回 False,系统会降级到 mock
USE_LLM = True  # 是否有可用的 LLM(自动根据 key 检测)

# 模型配置
LLM_PROVIDER = "deepseek"            # 目前只支持 deepseek
LLM_BASE_URL = "https://api.deepseek.com"
LLM_MODEL = "deepseek-v4-flash"      # 闪速版,更快更便宜;另可换 deepseek-v4-pro
LLM_API_KEY = ""                     # 留空,自动从 env 读 DEEPSEEK_API_KEY / DEEPSEEK_API

# 调用参数
LLM_TIMEOUT = 180                    # 秒(长 CoT 需要更多时间)
LLM_MAX_RETRIES = 3
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 32768               # flash 最大 384K,调到 32K 给 CoT + 工具结果留空间
LLM_THINKING = True                  # 开启思考模式(推荐)
LLM_REASONING_EFFORT = "high"        # flash 支持 thinking + tool calls,放开了用

# Writer / Reviewer / Long-COT 专用 token 配额
LLM_WRITER_MAX_TOKENS = 32768        # 长文写作 + thinking 余量(2026-08 调高)
LLM_REVIEWER_MAX_TOKENS = 8192       # 审核任务(需要输出 checklist + issues)
LLM_LONG_COT_MAX_TOKENS = 49152      # 长链推理任务(Orchestrator ReAct / Researcher Plan-and-Execute)

# Tool Calls 配置
LLM_MAX_TOOL_ROUNDS = 10             # 2026-08 调高:Plan-and-Execute / ReAct 都要更多轮

# CoT / Tool Calling 日志
LLM_REASONING_LOG_ENABLED = True     # 把 reasoning_content 落 log(2026-08 新增)
LLM_TOOL_CALL_LOG_ENABLED = True     # 把 tool_calls 落 log(2026-08 新增)

# Writer-Reviewer 迭代配置
# 写完文章后,Reviewer 审,不通过则让 Writer 修改,直到通过或达到最大轮数
LLM_REFINEMENT_MAX_ROUNDS = 2        # 最多修改 2 次(总共 3 次 review)
LLM_REFINEMENT_MIN_SCORE = 85       # 达到此分数即认为通过(默认 85,要求较高)
LLM_REFINEMENT_ACCEPT_ALL = True     # True=达到 max_rounds 后即使不通过也接受;False=丢弃

# Orchestrator ReAct 配置(2026-08 新增)
ORCHESTRATOR_MAX_ROUNDS = 6          # ReAct Orchestrator 单轮最大工具调用数
ORCHESTRATOR_MAX_OUTER_ROUNDS = 3    # 外层循环(一轮不通过时重试)最大次数
ORCHESTRATOR_TOP_N = 5               # 2026-08-23:并行 ReAct 的 top-N 数量
ORCHESTRATOR_MAX_WORKERS = 3         # 2026-08-23:并行 ReAct 的最大线程数

# ============ Logging ============
LOG_LEVEL = "INFO"
