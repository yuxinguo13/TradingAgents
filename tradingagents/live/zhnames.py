"""Chinese company names, so the reader is not asked to translate a ticker.

Three layers, checked in order, and the report says which one answered:

1. **``~/.tradingagents/company_names_zh.json``** — the reader's own file. It
   wins over everything, because a name they have decided on is not a thing
   this module should argue with. Written by hand; never written to by code.
2. **:data:`NAMES`** — a curated table. Hand-checked, and deliberately not
   generated: there is no keyless feed of Chinese equity names, and a wrong
   name is worse than an English one because it looks authoritative.
3. **:func:`derive`** — a mechanical gloss off the English name's suffix
   ("… Therapeutics" → "…制药"). A Nasdaq momentum screen is mostly biotech
   tickers nobody has a Chinese name for, and "NRIX（…制药）" is more use to a
   Chinese reader than "NRIX". It is marked with ``°`` in every rendering so it
   can never be mistaken for the curated layer.

Anything a reader disagrees with is one line in their own JSON file away from
being fixed, which is the point of putting the override first.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Marks a name assembled by rule rather than looked up. Printed, and explained
# in the report footer.
DERIVED_MARK = "°"


def override_path() -> Path:
    env = os.getenv("TRADINGAGENTS_ZH_NAMES_PATH")
    if env:
        return Path(env).expanduser()
    home = Path(os.getenv("TRADINGAGENTS_HOME", Path.home() / ".tradingagents"))
    return home / "company_names_zh.json"


# --- the curated table ------------------------------------------------------
# Grouped by why the name is here, so a reader adding to it knows where to put
# a new one. Names are the form used in Chinese financial media, not a literal
# translation of the English.
NAMES: dict[str, str] = {
    # --- benchmarks and sector ETFs ---
    "SPY": "标普500ETF", "VOO": "标普500ETF(先锋)", "IVV": "标普500ETF(安硕)",
    "QQQ": "纳指100ETF", "QQQM": "纳指100ETF(迷你)", "IWM": "罗素2000ETF",
    "DIA": "道琼斯ETF", "VTI": "全美股市ETF", "ARKK": "方舟创新ETF",
    "SMH": "半导体ETF(VanEck)", "SOXX": "半导体ETF(安硕)", "SOXL": "半导体三倍做多ETF",
    "XLK": "科技板块ETF", "XLF": "金融板块ETF", "XLE": "能源板块ETF",
    "XLV": "医疗板块ETF", "XLI": "工业板块ETF", "XLY": "可选消费ETF",
    "XLP": "必需消费ETF", "XLU": "公用事业ETF", "XLB": "原材料ETF",
    "XLRE": "房地产ETF", "XLC": "通信服务ETF", "TLT": "20年期美债ETF",
    "GLD": "黄金ETF", "SLV": "白银ETF", "USO": "原油ETF", "VXX": "波动率ETF",
    "TQQQ": "纳指三倍做多ETF", "SQQQ": "纳指三倍做空ETF",

    # --- semiconductors ---
    "NVDA": "英伟达", "AMD": "超微半导体", "INTC": "英特尔", "AVGO": "博通",
    "QCOM": "高通", "TXN": "德州仪器", "ADI": "亚德诺半导体", "MU": "美光科技",
    "MRVL": "迈威尔科技", "NXPI": "恩智浦半导体", "ON": "安森美半导体",
    "AMAT": "应用材料", "LRCX": "泛林集团", "KLAC": "科天半导体",
    "ASML": "阿斯麦", "TSM": "台积电", "ARM": "Arm控股", "ALAB": "Astera Labs",
    "CRDO": "Credo科技", "SMCI": "超微电脑", "AXTI": "AXT晶体",
    "MCHP": "微芯科技", "SWKS": "思佳讯", "QRVO": "威讯联合", "MPWR": "芯源系统",
    "TER": "泰瑞达", "ENTG": "英特格", "ONTO": "昂图创新", "COHR": "相干公司",
    "GFS": "格芯", "UMC": "联电", "STM": "意法半导体", "IFNNY": "英飞凌",
    "WOLF": "Wolfspeed", "LSCC": "莱迪思半导体", "SITM": "SiTime",
    "POWI": "power集成", "ALGM": "Allegro微系统", "AMBA": "安霸",
    "SLAB": "芯科科技", "RMBS": "Rambus", "SNPS": "新思科技", "CDNS": "楷登电子",

    # --- big tech / software / internet ---
    "AAPL": "苹果", "MSFT": "微软", "GOOGL": "谷歌(A类)", "GOOG": "谷歌(C类)",
    "AMZN": "亚马逊", "META": "Meta(脸书)", "TSLA": "特斯拉", "NFLX": "奈飞",
    "ORCL": "甲骨文", "CRM": "赛富时", "ADBE": "奥多比", "NOW": "ServiceNow",
    "PLTR": "帕兰提尔", "IBM": "IBM", "CSCO": "思科", "ACN": "埃森哲",
    "INTU": "财捷集团", "UBER": "优步", "ABNB": "爱彼迎", "BKNG": "缤客控股",
    "SHOP": "Shopify", "SQ": "Block", "XYZ": "Block", "PYPL": "PayPal",
    "COIN": "Coinbase", "HOOD": "罗宾汉", "SOFI": "SoFi科技", "RBLX": "Roblox",
    "SNAP": "Snap", "PINS": "Pinterest", "SPOT": "Spotify", "ZM": "Zoom",
    "DOCU": "DocuSign", "CRWD": "CrowdStrike", "PANW": "派拓网络",
    "ZS": "Zscaler", "S": "SentinelOne", "NET": "Cloudflare", "FTNT": "飞塔",
    "DDOG": "Datadog", "SNOW": "Snowflake", "MDB": "MongoDB", "TEAM": "Atlassian",
    "WDAY": "Workday", "ADSK": "欧特克", "HUBS": "HubSpot", "VEEV": "Veeva系统",
    "TTD": "The Trade Desk", "APP": "AppLovin", "U": "Unity软件",
    "MGNI": "Magnite", "ROKU": "Roku", "DASH": "DoorDash", "LYFT": "来福车",
    "CRWV": "CoreWeave", "AI": "C3.ai", "PATH": "UiPath", "GTLB": "GitLab",
    "ESTC": "Elastic", "CFLT": "Confluent", "TWLO": "Twilio", "OKTA": "Okta",
    "ZD": "Ziff Davis", "EXPE": "亿客行", "FLYW": "Flywire", "NAVN": "Navan",
    "CHYM": "Chime金融", "SLDE": "Slide保险",

    # --- healthcare / pharma / biotech ---
    "LLY": "礼来", "JNJ": "强生", "UNH": "联合健康", "ABBV": "艾伯维",
    "MRK": "默沙东", "PFE": "辉瑞", "TMO": "赛默飞世尔", "ABT": "雅培",
    "DHR": "丹纳赫", "AMGN": "安进", "GILD": "吉利德", "VRTX": "福泰制药",
    "REGN": "再生元", "BIIB": "渤健", "MRNA": "莫德纳", "BNTX": "BioNTech",
    "ISRG": "直觉外科", "SYK": "史赛克", "BSX": "波士顿科学", "MDT": "美敦力",
    "CI": "信诺集团", "CVS": "西维斯健康", "ELV": "Elevance健康", "HUM": "哈门那",
    "MCK": "麦克森", "COR": "Cencora", "ZTS": "硕腾", "IDXX": "爱德士",
    "NTRA": "Natera", "GH": "Guardant Health", "EXAS": "Exact Sciences",
    "ATAI": "ATAI生命科学", "KYMR": "Kymera制药", "NRIX": "Nurix制药",
    "IMVT": "Immunovant", "TRVI": "Trevi制药", "DYN": "Dyne制药",
    "IDYA": "IDEAYA生物", "NUTX": "Nutex健康", "CRSP": "CRISPR疗法",
    "NTLA": "Intellia疗法", "BEAM": "Beam疗法", "ALNY": "Alnylam制药",
    "SRPT": "Sarepta疗法", "BMRN": "BioMarin制药", "INCY": "因塞特",
    "NBIX": "神经分泌生物", "UTHR": "联合治疗", "JAZZ": "爵士制药",
    "HALO": "Halozyme", "RARE": "Ultragenyx", "IONS": "Ionis制药",

    # --- financials ---
    "BRK-B": "伯克希尔(B类)", "BRK.B": "伯克希尔(B类)", "JPM": "摩根大通",
    "BAC": "美国银行", "WFC": "富国银行", "GS": "高盛", "MS": "摩根士丹利",
    "C": "花旗集团", "SCHW": "嘉信理财", "BLK": "贝莱德", "V": "维萨",
    "MA": "万事达", "AXP": "美国运通", "SPGI": "标普全球", "CME": "芝商所",
    "ICE": "洲际交易所", "COF": "第一资本", "PGR": "前进保险", "TRV": "旅行者保险",
    "AIG": "美国国际集团", "MET": "大都会人寿", "PRU": "保德信金融",
    "BWIN": "鲍德温保险集团", "KKR": "KKR集团", "BX": "黑石集团", "APO": "阿波罗全球",

    # --- consumer ---
    "WMT": "沃尔玛", "COST": "好市多", "HD": "家得宝", "LOW": "劳氏",
    "TGT": "塔吉特", "PG": "宝洁", "KO": "可口可乐", "PEP": "百事可乐",
    "MCD": "麦当劳", "SBUX": "星巴克", "NKE": "耐克", "LULU": "露露乐蒙",
    "TJX": "TJX公司", "ROST": "罗斯百货", "ORLY": "奥莱利汽配", "AZO": "汽车地带",
    "CMG": "墨式烧烤", "YUM": "百胜餐饮", "DPZ": "达美乐比萨",
    "CAKE": "芝乐坊餐饮", "BJRI": "BJ's餐厅", "CHEF": "美食之家",
    "PM": "菲利普莫里斯", "MO": "奥驰亚", "MDLZ": "亿滋国际", "KHC": "卡夫亨氏",
    "GIS": "通用磨坊", "CL": "高露洁", "KMB": "金佰利", "EL": "雅诗兰黛",
    "DIS": "迪士尼", "CMCSA": "康卡斯特", "WBD": "华纳兄弟探索",

    # --- industrials / energy / materials / utilities ---
    "GE": "通用电气", "BA": "波音", "CAT": "卡特彼勒", "DE": "迪尔公司",
    "HON": "霍尼韦尔", "RTX": "雷神技术", "LMT": "洛克希德马丁",
    "NOC": "诺斯罗普格鲁曼", "GD": "通用动力", "UNP": "联合太平洋",
    "UPS": "联合包裹", "FDX": "联邦快递", "EMR": "艾默生电气", "ETN": "伊顿",
    "PH": "派克汉尼汾", "ITW": "伊利诺伊工具", "MMM": "3M公司",
    "AXON": "Axon企业", "URI": "联合租赁", "PWR": "广达服务",
    "XOM": "埃克森美孚", "CVX": "雪佛龙", "COP": "康菲石油", "EOG": "EOG资源",
    "SLB": "斯伦贝谢", "HAL": "哈里伯顿", "OXY": "西方石油", "PSX": "菲利普斯66",
    "VLO": "瓦莱罗能源", "MPC": "马拉松原油", "KMI": "金德摩根", "WMB": "威廉姆斯",
    "NESR": "国民能源服务", "CLMT": "Calumet", "SBLK": "星散海运",
    "LIN": "林德集团", "APD": "空气化工", "SHW": "宣伟涂料", "ECL": "艺康集团",
    "NUE": "纽柯钢铁", "FCX": "自由港麦克莫兰", "NEM": "纽蒙特矿业",
    "DOW": "陶氏化学", "LYB": "利安德巴塞尔", "ALB": "雅保锂业", "CF": "CF实业",
    "NEE": "新纪元能源", "SO": "南方电力", "DUK": "杜克能源", "AEP": "美国电力",
    "D": "道明尼能源", "EXC": "爱克斯龙", "VST": "Vistra能源", "CEG": "星座能源",

    # --- real estate / telecom ---
    "PLD": "安博物流", "AMT": "美国电塔", "CCI": "冠城国际", "EQIX": "易昆尼克斯",
    "SPG": "西蒙地产", "O": "房地产收入公司", "PSA": "大众仓储", "VICI": "VICI地产",
    "WELL": "Welltower", "DLR": "数字房地产", "T": "美国电话电报", "VZ": "威瑞森",
    "TMUS": "T-Mobile美国",
}

# --- the mechanical fallback ------------------------------------------------
# Order matters: the longest, most specific phrase must be tried first, or
# "Life Sciences" is matched by "Sciences" and glossed as the wrong thing.
_SUFFIX_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bLife Sciences?\b", re.I), "生命科学"),
    (re.compile(r"\bBiosciences?\b", re.I), "生物科技"),
    (re.compile(r"\bBiotechnolog(?:y|ies)\b", re.I), "生物技术"),
    (re.compile(r"\bTherapeutics?\b", re.I), "制药"),
    (re.compile(r"\bPharmaceuticals?\b|\bPharma\b", re.I), "制药"),
    (re.compile(r"\bMedical\b|\bMedicines?\b", re.I), "医药"),
    (re.compile(r"\bHealthcare\b|\bHealth\b", re.I), "健康"),
    (re.compile(r"\bDiagnostics?\b", re.I), "诊断"),
    (re.compile(r"\bGenomics?\b|\bGenetics?\b", re.I), "基因"),
    (re.compile(r"\bSemiconductors?\b", re.I), "半导体"),
    (re.compile(r"\bMicroelectronics\b|\bElectronics?\b", re.I), "电子"),
    (re.compile(r"\bTechnolog(?:y|ies)\b|\bTech\b", re.I), "科技"),
    (re.compile(r"\bSoftware\b", re.I), "软件"),
    (re.compile(r"\bNetworks?\b|\bNetworking\b", re.I), "网络"),
    (re.compile(r"\bCommunications?\b", re.I), "通信"),
    (re.compile(r"\bSystems?\b", re.I), "系统"),
    (re.compile(r"\bSolutions?\b", re.I), "解决方案"),
    (re.compile(r"\bBancorp\b|\bBancshares\b|\bBank\b|\bBanks\b", re.I), "银行"),
    (re.compile(r"\bFinancial\b|\bFinance\b", re.I), "金融"),
    (re.compile(r"\bCapital\b", re.I), "资本"),
    (re.compile(r"\bInsurance\b", re.I), "保险"),
    (re.compile(r"\bRealty\b|\bReal Estate\b|\bProperties\b", re.I), "地产"),
    (re.compile(r"\bEnergy\b", re.I), "能源"),
    (re.compile(r"\bPetroleum\b|\bOil\b", re.I), "石油"),
    (re.compile(r"\bMining\b|\bMines\b|\bGold\b", re.I), "矿业"),
    (re.compile(r"\bResources?\b", re.I), "资源"),
    (re.compile(r"\bMaterials?\b", re.I), "材料"),
    (re.compile(r"\bChemicals?\b", re.I), "化工"),
    (re.compile(r"\bIndustries\b|\bIndustrial\b", re.I), "工业"),
    (re.compile(r"\bManufacturing\b", re.I), "制造"),
    (re.compile(r"\bMotors?\b|\bAutomotive\b", re.I), "汽车"),
    (re.compile(r"\bAirlines?\b|\bAviation\b|\bAerospace\b", re.I), "航空"),
    (re.compile(r"\bShipping\b|\bMaritime\b|\bCarriers?\b", re.I), "海运"),
    (re.compile(r"\bLogistics\b|\bTransport(?:ation)?\b", re.I), "物流"),
    (re.compile(r"\bRestaurants?\b|\bDining\b", re.I), "餐饮"),
    (re.compile(r"\bRetail\b|\bStores?\b", re.I), "零售"),
    (re.compile(r"\bFoods?\b|\bBeverages?\b", re.I), "食品"),
    (re.compile(r"\bEntertainment\b|\bMedia\b", re.I), "传媒"),
    (re.compile(r"\bGaming\b|\bGames?\b", re.I), "游戏"),
    (re.compile(r"\bUtilities\b|\bElectric\b|\bPower\b", re.I), "电力"),
    (re.compile(r"\bHoldings?\b", re.I), "控股"),
    (re.compile(r"\bPartners\b|\bGroup\b", re.I), "集团"),
]

_LEGAL = re.compile(
    r"[,\s]+(?:inc|incorporated|corp|corporation|co|company|ltd|limited|plc|"
    r"s\.?a\.?|n\.?v\.?|ag|se|llc|l\.?p\.?|class\s+[a-c]|"
    r"common stock|ordinary shares?)\.?$", re.I)

# Nasdaq's listing file names a *security*, not a company: "Adaptive
# Biotechnologies Corporation - Common Stock". The descriptor after the dash is
# what the exchange is quoting, and it is separated by " - " rather than by a
# comma, so _LEGAL — which is anchored to the end and requires a comma or space
# in front — never reached the "Corporation" hiding behind it. The gloss then
# ran on "Adaptive Biotechnologies Corporation" and produced "Adaptive
# Corporation生物技术": a suffix removed from the middle of a name that still
# carried its legal form.
_INSTRUMENT = re.compile(
    r"\s+-\s+(?:common stock|ordinary shares?|class\s+[a-z](?:\s+\w+)*|"
    r"american depositary shares?|depositary shares?|units?|warrants?|rights?|"
    r"preferred stock|preference shares?|shares? of beneficial interest|"
    r"limited partnership|.*\bshares?\b|.*\bstock\b).*$", re.I)


def _strip_legal(name: str) -> str:
    """Company name out of an exchange's security name."""
    out = _INSTRUMENT.sub("", (name or "").strip()).strip()
    prev = None
    while out and out != prev:                 # "X Holdings, Inc." needs two passes
        prev = out
        out = _LEGAL.sub("", out).strip(" ,.")
    return out


def derive(english: str) -> str:
    """A gloss off the English name, or "" when no rule fires.

    Returns e.g. ``Nurix Therapeutics`` → ``Nurix制药``: the proper noun is kept
    in Latin script because transliterating it would invent a name, and the
    descriptive suffix — which is the part a reader actually wants translated,
    since it says what the company does — is replaced.
    """
    base = _strip_legal(english)
    if not base:
        return ""
    for pattern, zh in _SUFFIX_RULES:
        if pattern.search(base):
            head = pattern.sub("", base).strip(" ,.&-")
            head = re.sub(r"\s{2,}", " ", head)
            return f"{head}{zh}" if head else zh
    return ""


@dataclass(frozen=True)
class ZhName:
    """A resolved name and where it came from."""

    symbol: str
    zh: str = ""
    english: str = ""
    source: str = "none"          # override | curated | derived | none

    @property
    def derived(self) -> bool:
        return self.source == "derived"

    def label(self, *, mark: bool = True) -> str:
        """``英伟达`` / ``Nurix制药°`` / ``Nurix Therapeutics`` / ``NRIX``.

        The English name is the fallback rather than the ticker: a reader who
        cannot place ``NRIX`` can place ``Nurix Therapeutics``, and printing the
        ticker twice tells them nothing they did not already have.
        """
        if self.zh:
            return self.zh + (DERIVED_MARK if mark and self.derived else "")
        return self.english or self.symbol

    def full(self, *, mark: bool = True) -> str:
        """Both names when both exist, for the one place per page that has room."""
        zh = self.label(mark=mark)
        if self.zh and self.english and self.english != self.zh:
            return f"{zh}（{self.english}）"
        return zh


class ZhNames:
    """Resolver with the override file loaded once per run."""

    def __init__(self, overrides: dict | None = None, *, path: Path | None = None):
        self.path = path or override_path()
        # Normalised on both paths, not only when read from disk: a caller
        # passing {"nvda": ...} and a file containing {"nvda": ...} must behave
        # the same, and the file is the one a human types into.
        self.overrides = self._clean(overrides) if overrides is not None else self._load()

    @staticmethod
    def _clean(raw: dict) -> dict[str, str]:
        return {str(k).strip().upper(): str(v).strip()
                for k, v in (raw or {}).items() if str(k).strip() and str(v).strip()}

    def _load(self) -> dict[str, str]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except Exception as exc:
            # A malformed override file must cost the override, never the run.
            logger.warning("could not read %s (%s); using the built-in names only",
                           self.path, exc)
            return {}
        if not isinstance(raw, dict):
            return {}
        return self._clean(raw)

    def get(self, symbol: str, english: str = "") -> ZhName:
        sym = (symbol or "").strip().upper()
        en = _strip_legal(english)
        if sym in self.overrides:
            return ZhName(sym, self.overrides[sym], en, "override")
        if sym in NAMES:
            return ZhName(sym, NAMES[sym], en, "curated")
        guess = derive(english)
        if guess:
            return ZhName(sym, guess, en, "derived")
        return ZhName(sym, "", en, "none")

    def label(self, symbol: str, english: str = "") -> str:
        return self.get(symbol, english).label()


_DEFAULT: ZhNames | None = None


def resolver() -> ZhNames:
    """A process-wide resolver, so the override file is read once."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = ZhNames()
    return _DEFAULT


def zh(symbol: str, english: str = "") -> str:
    """Convenience for the renderers: the label only."""
    return resolver().label(symbol, english)
