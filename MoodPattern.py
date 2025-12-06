import sqlite3
from pathlib import Path
import io
import zipfile
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Union
import logging
from openai import OpenAI

# ========== AI 配置 ==========
api_key = st.secrets["API_KEY"]
api_base = st.secrets["API_BASE"]
model_id = st.secrets["MODEL_ID"]

# 初始化客户端（全局，在侧边栏配置时重新初始化）

client = OpenAI(api_key=api_key, base_url=api_base)

def init_ai_client(api_key: str = None, api_base: str = None, model_id: str = None):
    """初始化AI客户端"""
    global client
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=api_base
        )
        return client is not None
    except Exception as e:
        st.error(f"AI客户端初始化失败: {str(e)}")
        return False


# ========== 通用 AI 调用封装 ==========
def ask_ai(messages, json_type=False, model_id=st.secrets["MODEL_ID"]):
    """
    通用 AI 查询接口
    messages: str 或 list
    json_type: 是否要求返回 JSON（默认关闭，因为情绪分析更适合自然语言）
    """
    global client
    if client is None:
        return "AI功能未初始化，请在侧边栏配置API Key"

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    extra_body = {}
    if json_type:
        extra_body = {
            "response_format": {"type": "json_object"},
            "search_disable": True
        }

    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=messages,
            extra_body=extra_body
        )
        content = resp.choices[0].message.content
        return json.loads(content) if json_type else content
    except Exception as e:
        return f"AI调用失败: {str(e)}"


# ========== 情绪 AI 解读专用函数 ==========
def ai_explain_mood(df):
    """
    输入：你的情绪 DataFrame
    输出：情绪趋势 + 关键因素 + 管理建议（面向用户、自然）
    """
    if df.empty or len(df) < 2:
        return "需要至少2条记录才能进行情绪分析。"

    # 简单统计
    avg_mood = df["mood_score"].mean()
    worst = df.loc[df["mood_score"].idxmin()]
    best = df.loc[df["mood_score"].idxmax()]
    last = df.iloc[-1]

    # 构建摘要信息
    summary = f"""
## 情绪数据统计
最近情绪平均分：{avg_mood:.2f}/10
记录总数：{len(df)}条
时间范围：{df['record_date'].min().strftime('%Y-%m-%d')} 至 {df['record_date'].max().strftime('%Y-%m-%d')}

## 关键记录点
最近一次记录：{last['mood_score']}分
- 活动：{last.get('activities', '无')}
- 备注：{last.get('notes', '无')[:50]}...

情绪最低点：{worst['mood_score']}分
- 日期：{worst['record_date'].strftime('%Y-%m-%d')}
- 活动：{worst.get('activities', '无')}
- 备注：{worst.get('notes', '无')[:50]}...

情绪最高点：{best['mood_score']}分  
- 日期：{best['record_date'].strftime('%Y-%m-%d')}
- 活动：{best.get('activities', '无')}
- 备注：{best.get('notes', '无')[:50]}...
"""

    prompt = f"""
你是一名专业的心理情绪教练，请用**温柔、现实、面向行动**的方式，分析用户近一段时间的情绪数据。

以下是用户的情绪记录摘要：
{summary}

请生成一份温暖、实用的情绪分析报告，包含以下部分：

## 1. 情绪趋势总结
用普通人能理解的语言描述整体情绪变化趋势

## 2. 可能的影响因素
基于活动记录、备注内容等，指出可能的情绪诱因

## 3. 个性化建议（3~5条）
给出具体、可执行的建议，比如：
- 如果发现某些活动带来积极情绪，建议增加这些活动
- 如果发现压力较大，提供简单的减压方法
- 如果情绪波动较大，建议稳定情绪的小技巧

## 4. 温馨提醒
用温暖的话语鼓励用户，肯定TA记录情绪的努力

**注意事项：**
- 语气温柔、避免专业术语
- 避免负面评价，用建设性语言
- 不要输出代码或技术性内容
- 针对数据特点提供具体建议
"""

    return ask_ai(prompt, json_type=False)


def ai_generate_weekly_report(df):
    """生成周度情绪报告"""
    if df.empty or len(df) < 3:
        return "需要更多记录才能生成周报（建议至少3条）。"

    # 获取最近7天数据
    recent_df = df[df["record_date"] >= (datetime.now() - timedelta(days=7))]
    if len(recent_df) < 2:
        return "本周记录较少，建议多记录几天。"

    # 构建周报数据
    avg_mood = recent_df["mood_score"].mean()
    mood_std = recent_df["mood_score"].std()

    # 分析活动影响
    activity_summary = ""
    if "activities" in recent_df.columns:
        activity_data = []
        for _, row in recent_df.iterrows():
            if pd.notna(row["activities"]) and row["activities"]:
                activities = [a.strip() for a in str(row["activities"]).split(",")]
                for activity in activities:
                    if activity:
                        activity_data.append({"activity": activity, "mood": row["mood_score"]})

        if activity_data:
            activity_df = pd.DataFrame(activity_data)
            activity_stats = activity_df.groupby("activity")["mood"].agg(["mean", "count"]).round(2)
            top_activities = activity_stats.sort_values("mean", ascending=False).head(3)

            activity_summary = "\n## 活动影响分析\n"
            for activity, row in top_activities.iterrows():
                activity_summary += f"- {activity}: 平均情绪 {row['mean']:.1f}分（出现{row['count']}次）\n"

    prompt = f"""
你是一名贴心的情绪管理助手，请为用户生成一份温暖、鼓励的周度情绪报告。

## 本周情绪概览
- 记录天数：{len(recent_df)}天
- 平均情绪：{avg_mood:.1f}/10
- 情绪稳定性：{'较稳定' if mood_std < 2 else '波动较大'}
- 时间范围：{recent_df['record_date'].min().strftime('%m/%d')} - {recent_df['record_date'].max().strftime('%m/%d')}

{activity_summary if activity_summary else ''}

## 请生成包含以下内容的周报：
1. **本周情绪总结**：用温暖的语言描述本周情绪特点
2. **进步与亮点**：肯定用户的积极变化和努力
3. **发现与洞察**：基于数据指出有意义的现象
4. **下周小目标**：2-3个简单可行的建议
5. **温馨鼓励**：用支持性的话语结束报告

**风格要求：**
- 语气亲切、鼓励、实用
- 避免说教，用建议而非命令
- 结合具体数据提供个性化反馈
- 保持积极向上的基调
"""

    return ask_ai(prompt, json_type=False)


# ========== 新增：查询函数 ==========
def query_records(
        conn,
        user_id: int = None,
        username: str = None,
        start_date: datetime = None,
        end_date: datetime = None,
        min_score: int = None,
        max_score: int = None,
        keyword: str = None
) -> pd.DataFrame:
    """
    查询情绪记录
    """
    # 构建基础查询
    sql = """
    SELECT 
        mr.*,
        u.username
    FROM mood_records mr
    JOIN users u ON mr.user_id = u.user_id
    WHERE 1=1
    """

    params = []

    # 按用户ID筛选
    if user_id is not None:
        sql += " AND mr.user_id = ?"
        params.append(user_id)

    # 按用户名筛选
    if username is not None and username != "所有用户":
        sql += " AND u.username = ?"
        params.append(username)

    # 按日期筛选
    if start_date:
        sql += " AND DATE(mr.record_date) >= ?"
        params.append(start_date.strftime('%Y-%m-%d'))

    if end_date:
        sql += " AND DATE(mr.record_date) <= ?"
        params.append(end_date.strftime('%Y-%m-%d'))

    # 按分数筛选
    if min_score is not None:
        sql += " AND mr.mood_score >= ?"
        params.append(min_score)

    if max_score is not None:
        sql += " AND mr.mood_score <= ?"
        params.append(max_score)

    sql += " ORDER BY mr.record_date DESC"

    # 执行查询
    df = pd.read_sql(sql, conn, params=params)

    # 关键词搜索
    if keyword and not df.empty:
        keyword = keyword.lower()
        mask = (
                df["notes"].str.lower().str.contains(keyword, na=False) |
                df["activities"].str.lower().str.contains(keyword, na=False) |
                df["tags"].str.lower().str.contains(keyword, na=False)
        )
        df = df[mask]

    return df


# ========== 在这里插入批量查询功能（功能3）==========
def batch_query_records(
        conn,
        user_ids: List[int],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        return_type: str = "dataframe"  # 或 "dict", "json"
) -> Union[Dict[int, pd.DataFrame], str]:
    """
    批量查询多个用户的记录
    """
    results = {}
    for user_id in user_ids:
        # 先获取用户名
        username_result = conn.execute(
            "SELECT username FROM users WHERE user_id = ?",
            (user_id,)
        ).fetchone()

        if username_result:
            username = username_result[0]
            user_df = query_records(
                conn,
                user_id=user_id,
                username=username,
                start_date=start_date,
                end_date=end_date
            )
            results[user_id] = user_df

    if return_type == "json":
        json_result = {}
        for user_id, df in results.items():
            if not df.empty:
                json_result[str(user_id)] = df.to_dict(orient='records')
        return json.dumps(json_result, ensure_ascii=False, indent=2)
    elif return_type == "dict":
        dict_result = {}
        for user_id, df in results.items():
            if not df.empty:
                dict_result[user_id] = df.to_dict(orient='records')
        return dict_result
    else:
        return results


def get_user_record(conn, user_id: int, record_id: int) -> Optional[Dict]:
    """获取特定用户的某条记录"""
    sql = """
    SELECT mr.*
    FROM mood_records mr
    WHERE mr.user_id = ? AND mr.id = ?
    """

    df = pd.read_sql(sql, conn, params=(user_id, record_id))

    if not df.empty:
        return df.iloc[0].to_dict()
    return None


# ========== 新增：用户数据隔离函数 ==========
def load_user_data(conn, user_id: int) -> pd.DataFrame:
    """加载特定用户的数据"""
    sql = """
    SELECT 
        mr.*,
        u.username
    FROM mood_records mr
    JOIN users u ON mr.user_id = u.user_id
    WHERE mr.user_id = ?
    ORDER BY mr.record_date DESC
    """

    try:
        df = pd.read_sql(sql, conn, params=(user_id,))

        # 确保日期类型正确
        if not df.empty and 'record_date' in df.columns:
            df['record_date'] = pd.to_datetime(df['record_date'])
        if not df.empty and 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])

        return df
    except Exception as e:
        st.error(f"加载用户数据失败: {e}")
        return pd.DataFrame()


# ========== 在这里插入哈希加密功能（功能4）==========
import hashlib


def calculate_data_signature(df: pd.DataFrame) -> str:
    """
    计算数据签名，用于验证数据完整性
    """
    if df.empty:
        return ""

    # 将DataFrame转换为字符串并计算哈希
    data_string = df.to_csv(index=False)
    signature = hashlib.sha256(data_string.encode()).hexdigest()

    return signature


def encrypt_sensitive_field(text: str, secret_key: str = "") -> str:
    """
    加密敏感字段（简化版，实际应用应使用更安全的加密）
    """
    if not text or not secret_key:
        return text

    # 使用HMAC进行消息认证
    h = hashlib.sha256()
    h.update(f"{text}{secret_key}".encode())
    return h.hexdigest()[:20]  # 返回部分哈希作为加密值


def verify_data_integrity(original_hash: str, current_df: pd.DataFrame) -> bool:
    """
    验证数据完整性
    """
    current_hash = calculate_data_signature(current_df)
    return original_hash == current_hash


def create_backup_with_verification(conn, backup_name: str) -> Dict:
    """
    创建带完整性验证的备份
    """
    from pathlib import Path

    # 确保备份目录存在
    BACKUP_DIR = Path("./backups")
    BACKUP_DIR.mkdir(exist_ok=True)

    backup_info = {
        'name': backup_name,
        'timestamp': datetime.now().isoformat(),
        'data_hash': '',
        'verification_passed': False,
        'backup_path': ''
    }

    try:
        # 获取所有数据并计算哈希
        all_data = pd.read_sql("SELECT * FROM mood_records", conn)
        backup_info['data_hash'] = calculate_data_signature(all_data)
        backup_info['record_count'] = len(all_data)

        # 保存备份文件
        backup_path = BACKUP_DIR / f"{backup_name}.db"
        backup_info['backup_path'] = str(backup_path)

        # 使用SQLite的备份功能
        with sqlite3.connect(backup_path) as backup_conn:
            conn.backup(backup_conn)

        # 验证备份
        with sqlite3.connect(backup_path) as backup_conn:
            backup_data = pd.read_sql("SELECT * FROM mood_records", backup_conn)
            backup_info['verification_passed'] = verify_data_integrity(
                backup_info['data_hash'],
                backup_data
            )

        # 记录备份信息
        backup_log_path = BACKUP_DIR / "backup_log.json"
        backup_log = []
        if backup_log_path.exists():
            with open(backup_log_path, 'r', encoding='utf-8') as f:
                backup_log = json.load(f)

        backup_log.append(backup_info)

        with open(backup_log_path, 'w', encoding='utf-8') as f:
            json.dump(backup_log, f, ensure_ascii=False, indent=2, default=str)

        return backup_info

    except Exception as e:
        st.error(f"备份创建失败: {e}")
        backup_info['error'] = str(e)
        return backup_info


def restore_from_backup(backup_path: str, conn) -> bool:
    """
    从备份恢复数据
    """
    try:
        # 验证备份文件
        with sqlite3.connect(backup_path) as backup_conn:
            backup_data = pd.read_sql("SELECT * FROM mood_records", backup_conn)

        if backup_data.empty:
            st.warning("备份文件为空")
            return False

        # 清空当前表
        conn.execute("DELETE FROM mood_records")

        # 恢复数据
        backup_conn = sqlite3.connect(backup_path)
        backup_conn.backup(conn)
        backup_conn.close()

        # 验证恢复的数据
        restored_data = pd.read_sql("SELECT * FROM mood_records", conn)
        if len(restored_data) == len(backup_data):
            st.success(f"成功恢复 {len(restored_data)} 条记录")
            return True
        else:
            st.error("数据恢复验证失败")
            return False

    except Exception as e:
        st.error(f"恢复失败: {e}")
        return False


# ========== 新增：管理员功能 ==========
def is_admin(username: str) -> bool:
    """检查是否为管理员（这里只是示例，实际需要更安全的认证）"""
    # 这里可以改成从配置文件或数据库读取管理员列表
    admins = ["admin", "管理员", "system"]
    return username in admins


def get_all_users_data(conn) -> Dict[str, pd.DataFrame]:
    """获取所有用户的数据（仅管理员可用）"""
    sql = """
    SELECT 
        mr.*,
        u.username
    FROM mood_records mr
    JOIN users u ON mr.user_id = u.user_id
    ORDER BY u.username, mr.record_date DESC
    """

    try:
        df = pd.read_sql(sql, conn)

        if df.empty:
            return {}

        # 确保日期类型正确
        if 'record_date' in df.columns:
            df['record_date'] = pd.to_datetime(df['record_date'])
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])

        # 按用户分组
        users_data = {}
        for user in df["username"].unique():
            user_df = df[df["username"] == user].copy()
            users_data[user] = user_df

        return users_data
    except Exception as e:
        st.error(f"获取所有用户数据失败: {e}")
        return {}


def get_user_stats(conn):
    """获取用户统计信息"""
    sql = """
    SELECT 
        u.username,
        COUNT(mr.id) as record_count,
        AVG(mr.mood_score) as avg_mood,
        MIN(mr.record_date) as first_record,
        MAX(mr.record_date) as last_record
    FROM users u
    LEFT JOIN mood_records mr ON u.user_id = mr.user_id
    GROUP BY u.username
    ORDER BY record_count DESC
    """

    return pd.read_sql(sql, conn)


# ========== 情绪分析工具 ==========
def analyze_mood_patterns(df: pd.DataFrame) -> Dict:
    """分析情绪模式"""
    if df.empty or 'mood_score' not in df.columns:
        return {}

    analysis = {
        "overall_score": round(df["mood_score"].mean(), 2) if not df.empty else 0,
        "trend": "上升" if len(df) > 1 and df["mood_score"].iloc[-1] > df["mood_score"].iloc[0] else "下降",
        "best_day": df.loc[df["mood_score"].idxmax(), "record_date"].strftime("%Y-%m-%d") if len(df) > 0 else None,
        "worst_day": df.loc[df["mood_score"].idxmin(), "record_date"].strftime("%Y-%m-%d") if len(df) > 0 else None,
        "consistency": round(df["mood_score"].std(), 2) if len(df) > 1 else 0
    }

    # 按星期分析
    if 'record_date' in df.columns and not df.empty:
        df["weekday"] = df["record_date"].dt.day_name()
        weekday_avg = df.groupby("weekday")["mood_score"].mean()
        if not weekday_avg.empty:
            analysis["best_weekday"] = weekday_avg.idxmax()

    return analysis


def detect_mood_anomalies(df: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
    """检测情绪异常点"""
    if len(df) < 3 or 'mood_score' not in df.columns:
        return pd.DataFrame()

    scores = df["mood_score"].values
    mean_score = np.mean(scores)
    std_score = np.std(scores)

    if std_score == 0:
        return pd.DataFrame()

    z_scores = np.abs((scores - mean_score) / std_score)
    anomalies = df[z_scores > threshold].copy()

    if not anomalies.empty:
        anomalies["z_score"] = z_scores[z_scores > threshold]
        anomalies["deviation"] = anomalies["mood_score"] - mean_score

    return anomalies


# 设置页面
st.set_page_config(
    page_title="MoodPattern — 情绪管理助手",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== 在这里插入路径管理功能（功能2）==========
# 路径配置
BASE_DIR = Path(__file__).parent if "__file__" in locals() else Path.cwd()
DATA_DIR = BASE_DIR / "data"
BACKUP_DIR = BASE_DIR / "backups"
EXPORT_DIR = BASE_DIR / "exports"
LOG_DIR = BASE_DIR / "logs"

# 创建必要的目录
for directory in [DATA_DIR, BACKUP_DIR, EXPORT_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# 多数据库支持
def get_available_databases() -> List[Path]:
    """获取所有可用的数据库文件"""
    return list(DATA_DIR.glob("*.db"))


def create_new_database(db_name: str) -> Path:
    """创建新的数据库文件"""
    db_path = DATA_DIR / f"{db_name}.db"
    if not db_path.exists():
        conn = sqlite3.connect(db_path)
        # 初始化数据库结构
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS users(
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            created_at TEXT
        );
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS mood_records(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            mood_score INTEGER CHECK (mood_score BETWEEN 1 AND 10),
            mood_label TEXT,
            activities TEXT,
            notes TEXT,
            sleep_hours REAL,
            stress_level INTEGER CHECK (stress_level BETWEEN 1 AND 10),
            tags TEXT,
            weather TEXT,
            record_date TEXT,
            created_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON mood_records(user_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_record_date ON mood_records(record_date);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mood_score ON mood_records(mood_score);")
        conn.commit()
        conn.close()
    return db_path


def manage_database_files():
    """管理数据库文件"""
    dbs = get_available_databases()
    if dbs:
        st.write("可用数据库文件:")
        for db in dbs:
            size = db.stat().st_size
            st.write(f"- {db.name} ({size:,} bytes)")
    else:
        st.info("暂无数据库文件")


# 设置默认数据库路径
DEFAULT_DB_PATH = DATA_DIR / "mood_system.db"


# ========== 数据库初始化 ==========
def init_database(db_path: Path = DEFAULT_DB_PATH):
    """初始化数据库"""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys=ON")

    # 创建用户表
    conn.execute("""
    CREATE TABLE IF NOT EXISTS users(
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        created_at TEXT
    );
    """)

    # 创建情绪记录表（增强版）
    conn.execute("""
    CREATE TABLE IF NOT EXISTS mood_records(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        mood_score INTEGER CHECK (mood_score BETWEEN 1 AND 10),
        mood_label TEXT,
        activities TEXT,
        notes TEXT,
        sleep_hours REAL,
        stress_level INTEGER CHECK (stress_level BETWEEN 1 AND 10),
        tags TEXT,
        weather TEXT,
        record_date TEXT,
        created_at TEXT,
        FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
    );
    """)

    # 创建索引以提高查询性能
    conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON mood_records(user_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_record_date ON mood_records(record_date);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_mood_score ON mood_records(mood_score);")

    # 创建备份日志表
    conn.execute("""
    CREATE TABLE IF NOT EXISTS backup_logs(
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        backup_name TEXT,
        backup_time TEXT,
        record_count INTEGER,
        data_hash TEXT,
        verification_status INTEGER,
        backup_path TEXT
    );
    """)

    return conn


# 情绪标签映射
MOOD_LABELS = {
    1: "😭 非常低落",
    2: "😔 低落",
    3: "😟 有点低落",
    4: "😕 轻微低落",
    5: "😐 平静",
    6: "🙂 轻微愉悦",
    7: "😊 愉悦",
    8: "😄 开心",
    9: "🤩 非常开心",
    10: "🎉 兴奋"
}


# ========== 主应用界面 ==========
def main():
    # 初始化数据库连接
    conn = init_database()

    # 侧边栏
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/brain.png", width=80)
        st.title("🧠 MoodPattern")
        st.caption("你的情绪管理伙伴")

        # 数据库管理（新增功能）
        st.divider()
        st.subheader("🗃️ 数据库管理")

        # 显示当前数据库信息
        db_size = DEFAULT_DB_PATH.stat().st_size if DEFAULT_DB_PATH.exists() else 0
        st.info(f"当前数据库: {DEFAULT_DB_PATH.name}")
        st.metric("数据库大小", f"{db_size:,} bytes")

        # 数据库操作
        with st.expander("数据库操作"):
            # 创建新数据库
            new_db_name = st.text_input("新数据库名称", placeholder="输入数据库名称")
            if st.button("创建新数据库"):
                if new_db_name:
                    new_db_path = create_new_database(new_db_name)
                    st.success(f"已创建数据库: {new_db_path.name}")
                else:
                    st.warning("请输入数据库名称")

            # 切换数据库
            available_dbs = get_available_databases()
            if available_dbs:
                db_options = [db.name for db in available_dbs]
                selected_db = st.selectbox("选择数据库", db_options)
                if st.button("切换数据库"):
                    st.session_state.selected_db = DATA_DIR / selected_db
                    st.rerun()

            # 查看数据库信息
            if st.button("查看数据库信息"):
                manage_database_files()

        # 用户管理部分
        st.divider()
        st.subheader("👤 用户管理")
        username = st.text_input("请输入用户名")

        if st.button("选择/创建用户"):
            if username:
                # 检查用户是否存在
                user_check = conn.execute(
                    "SELECT user_id FROM users WHERE username = ?",
                    (username,)
                ).fetchone()

                if not user_check:
                    # 创建新用户
                    conn.execute(
                        "INSERT INTO users (username, created_at) VALUES (?, datetime('now'))",
                        (username,)
                    )
                    conn.commit()
                    st.success(f"新用户 {username} 已创建！")
                else:
                    st.info(f"欢迎回来，{username}！")

                # 设置当前用户到session state
                st.session_state.current_user = username
                st.session_state.user_id = conn.execute(
                    "SELECT user_id FROM users WHERE username = ?",
                    (username,)
                ).fetchone()[0]
                st.rerun()

        # 显示当前用户
        if 'current_user' in st.session_state:
            st.divider()
            st.subheader("当前用户")
            st.success(f"👤 {st.session_state.current_user}")

            # 检查是否为管理员
            admin_mode = False
            if is_admin(st.session_state.current_user):
                st.success("👑 管理员模式已识别")
                admin_view = st.checkbox("管理员视图（查看所有用户）", value=False)
                if admin_view:
                    admin_mode = True
                    st.warning("⚠️ 正在查看所有用户数据")
                    st.session_state.admin_mode = True
                else:
                    if 'admin_mode' in st.session_state:
                        del st.session_state.admin_mode
            else:
                if 'admin_mode' in st.session_state:
                    del st.session_state.admin_mode

        # 分析设置
        st.divider()
        st.subheader("📊 分析设置")
        anomaly_threshold = st.slider(
            "异常检测灵敏度",
            1.0, 3.0, 2.0, 0.1,
            help="Z-score阈值，值越小越敏感"
        )

        # 数据统计
        st.divider()
        st.subheader("📦 数据统计")

        # 根据模式加载数据
        if 'current_user' in st.session_state:
            if 'admin_mode' in st.session_state and st.session_state.admin_mode:
                # 管理员模式下显示所有用户数据
                all_users_data = get_all_users_data(conn)
                if all_users_data:
                    selected_user = st.selectbox(
                        "选择查看用户",
                        options=["所有用户"] + list(all_users_data.keys()),
                        index=0
                    )

                    if selected_user == "所有用户":
                        # 合并所有用户数据
                        all_dfs = []
                        for user, user_df in all_users_data.items():
                            all_dfs.append(user_df)
                        if all_dfs:
                            df = pd.concat(all_dfs, ignore_index=True)
                        else:
                            df = pd.DataFrame()
                    else:
                        df = all_users_data[selected_user]
                else:
                    df = pd.DataFrame()
                    st.info("暂无用户数据")
            else:
                # 普通用户模式下只加载自己的数据
                user_id = st.session_state.user_id
                df = load_user_data(conn, user_id)

            record_count = len(df) if not df.empty else 0
            st.metric("总记录数", record_count)

            if not df.empty and not ('admin_mode' in st.session_state and st.session_state.admin_mode):
                avg_mood = df["mood_score"].mean()
                st.metric("平均情绪", f"{avg_mood:.1f}/10")

        if st.button("🔄 刷新数据"):
            st.rerun()

    # 主界面
    st.title("🌈 MoodPattern — 情绪管理助手")

    # 检查是否已登录
    if 'current_user' not in st.session_state:
        st.info("请在左侧输入用户名开始使用")
        conn.close()
        return

    current_user = st.session_state.current_user
    user_id = st.session_state.user_id

    # 顶部状态栏
    if not df.empty and not ('admin_mode' in st.session_state and st.session_state.admin_mode):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            latest_mood = df.iloc[-1]["mood_score"] if not df.empty else 0
            st.metric("当前情绪", f"{latest_mood}/10", MOOD_LABELS.get(latest_mood, ""))
        with col2:
            streak = 0
            if not df.empty and 'record_date' in df.columns:
                dates = sorted(df["record_date"].unique())
                for i in range(1, min(7, len(dates)) + 1):
                    if (dates[-i].date() == (datetime.now().date() - timedelta(days=i - 1))):
                        streak += 1
                    else:
                        break
            st.metric("连续记录", f"{streak}天")
        with col3:
            if not df.empty and 'record_date' in df.columns:
                avg_week = df[df["record_date"] >= (datetime.now() - timedelta(days=7))]["mood_score"].mean()
                st.metric("本周平均", f"{avg_week:.1f}/10" if not np.isnan(avg_week) else "--")
            else:
                st.metric("本周平均", "--")
        with col4:
            if not df.empty and 'record_date' in df.columns:
                last_record = df["record_date"].max()
                days_since = (datetime.now().date() - last_record.date()).days
                if days_since >= 3:
                    st.error(f"{days_since}天未记录")
                else:
                    st.success("记录正常")
            else:
                st.info("暂无记录")
    elif 'admin_mode' in st.session_state and st.session_state.admin_mode:
        st.info("👑 管理员视图：您可以查看和搜索所有用户的数据")

    # 标签页 - 修改这里增加Tab 7和Tab 8
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📝 记录情绪",
        "📊 情绪分析",
        "🤖 AI助手",
        "📈 趋势",
        "⚙️ 设置",
        "🔧 管理",
        "📊 分析中心",
        "🔐 安全中心"
    ])

    # Tab 1: 记录情绪
    with tab1:
        st.subheader("记录今日情绪")

        with st.form("mood_form", clear_on_submit=True):
            col1, col2 = st.columns([2, 1])

            with col1:
                notes = st.text_area(
                    "📔 今日心情日记",
                    placeholder="写下今天的感受、发生的事情、想法...",
                    height=150,
                    help="详细记录有助于更好的分析和回顾"
                )
                activities = st.multiselect(
                    "🏃 今日活动",
                    ["工作", "学习", "运动", "社交", "娱乐", "休息", "家务", "其他"],
                    help="选择今天的活动"
                )

            with col2:
                mood_score = st.slider(
                    "😊 情绪分数",
                    1, 10, 5,
                    help="1=非常低落, 10=非常开心"
                )
                st.markdown(f"**{MOOD_LABELS.get(mood_score, '')}**")

                sleep_hours = st.slider(
                    "😴 睡眠时长(小时)",
                    0.0, 12.0, 7.0, 0.5
                )

                stress_level = st.slider(
                    "💼 压力水平",
                    1, 10, 5,
                    help="1=无压力, 10=压力极大"
                )

            weather = st.selectbox(
                "☁️ 天气",
                ["", "晴天", "多云", "雨天", "雪天", "大风", "其他"]
            )

            tags = st.multiselect(
                "🏷️ 标签",
                ["重要事件", "突破", "挑战", "放松", "思考", "成就", "感恩"]
            )

            submitted = st.form_submit_button("💾 保存记录", type="primary")

            if submitted:
                # 获取情绪标签
                mood_label = MOOD_LABELS.get(mood_score, "")

                # 插入记录
                sql = """
                    INSERT INTO mood_records(
                        user_id, mood_score, mood_label, activities, notes, 
                        sleep_hours, stress_level, tags, weather, record_date, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, date('now'), datetime('now'))
                    """

                params = (
                    user_id,
                    mood_score,
                    mood_label,
                    ", ".join(activities) if activities else "",
                    notes,
                    sleep_hours,
                    stress_level,
                    ", ".join(tags) if tags else "",
                    weather
                )

                conn.execute(sql, params)
                conn.commit()

                st.success("🎉 记录已保存！")
                st.balloons()

                # 显示情绪反馈
                if mood_score >= 8:
                    st.info("🎯 继续保持好心情！")
                elif mood_score <= 4:
                    st.info("🤗 记得照顾好自己的情绪，需要的话可以看看AI建议")

    # Tab 2: 情绪分析
    with tab2:
        st.subheader("情绪分析报告")

        if df.empty:
            st.info("📝 还没有记录，先去记录一下吧！")
        else:
            # 选择分析范围
            period = st.radio(
                "分析时段",
                ["最近7天", "最近30天", "全部记录"],
                horizontal=True
            )

            if period == "最近7天":
                analysis_df = df[df["record_date"] >= (datetime.now() - timedelta(days=7))]
            elif period == "最近30天":
                analysis_df = df[df["record_date"] >= (datetime.now() - timedelta(days=30))]
            else:
                analysis_df = df

            if analysis_df.empty:
                st.warning("该时段暂无记录")
            else:
                # 基本统计
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("平均情绪", f"{analysis_df['mood_score'].mean():.1f}")
                with col2:
                    st.metric("最高情绪", f"{analysis_df['mood_score'].max():.0f}")
                with col3:
                    st.metric("最低情绪", f"{analysis_df['mood_score'].min():.0f}")

                # 情绪趋势图
                st.subheader("📈 情绪趋势")
                fig, ax = plt.subplots(figsize=(10, 4))
                analysis_df_sorted = analysis_df.sort_values("record_date")
                ax.plot(analysis_df_sorted["record_date"], analysis_df_sorted["mood_score"],
                        marker='o', linewidth=2, markersize=6)
                ax.axhline(y=analysis_df_sorted["mood_score"].mean(), color='r',
                           linestyle='--', alpha=0.5, label=f"平均线 ({analysis_df_sorted['mood_score'].mean():.1f})")
                ax.set_xlabel("日期")
                ax.set_ylabel("情绪分数")
                ax.set_ylim(0, 10.5)
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

                # 情绪分布
                st.subheader("📊 情绪分布")
                fig2, ax2 = plt.subplots(figsize=(8, 4))
                bins = np.arange(1, 12) - 0.5
                ax2.hist(analysis_df_sorted["mood_score"], bins=bins,
                         edgecolor='black', alpha=0.7)
                ax2.set_xlabel("情绪分数")
                ax2.set_ylabel("频次")
                ax2.set_xticks(range(1, 11))
                st.pyplot(fig2)

                # 异常检测
                st.subheader("🔍 情绪异常检测")
                anomalies = detect_mood_anomalies(analysis_df_sorted, anomaly_threshold)
                if not anomalies.empty:
                    st.warning(f"检测到 {len(anomalies)} 个异常情绪点")
                    st.dataframe(
                        anomalies[["record_date", "mood_score", "notes", "z_score"]].sort_values("z_score",
                                                                                                 ascending=False),
                        use_container_width=True
                    )
                else:
                    st.success("情绪波动正常")

                # 活动关联分析
                if "activities" in analysis_df_sorted.columns:
                    st.subheader("🏃 活动与情绪关联")
                    activity_data = []
                    for _, row in analysis_df_sorted.iterrows():
                        if pd.notna(row["activities"]) and row["activities"].strip():
                            activities = [a.strip() for a in str(row["activities"]).split(",")]
                            for activity in activities:
                                if activity:
                                    activity_data.append({"activity": activity, "mood": row["mood_score"]})

                    if activity_data:
                        activity_df = pd.DataFrame(activity_data)
                        activity_stats = activity_df.groupby("activity")["mood"].agg(["mean", "count"]).round(2)
                        activity_stats = activity_stats[activity_stats["count"] >= 2]  # 至少出现2次

                        if not activity_stats.empty:
                            st.dataframe(
                                activity_stats.sort_values("mean", ascending=False),
                                use_container_width=True
                            )

                # 记录查询功能
                st.subheader("🔍 记录查询")

                with st.expander("高级查询", expanded=False):
                    col1, col2 = st.columns(2)

                    with col1:
                        # 如果是管理员，可以选择用户
                        if 'admin_mode' in st.session_state and st.session_state.admin_mode:
                            query_user = st.selectbox(
                                "查询用户",
                                options=["所有用户"] + list(df["username"].unique()) if 'username' in df.columns else [
                                    "所有用户"],
                                index=0
                            )
                        else:
                            query_user = st.text_input("查询用户", value=current_user)

                        query_start = st.date_input("开始日期",
                                                    value=datetime.now().date() - timedelta(days=30))
                        query_end = st.date_input("结束日期",
                                                  value=datetime.now().date())

                    with col2:
                        query_min = st.slider("最低分数", 1, 10, 1)
                        query_max = st.slider("最高分数", 1, 10, 10)
                        keyword = st.text_input("关键词搜索",
                                                placeholder="在备注/活动/标签中搜索")

                    if st.button("执行查询", type="secondary"):
                        # 处理用户查询条件
                        user_filter_id = None
                        user_filter_name = None

                        if 'admin_mode' in st.session_state and st.session_state.admin_mode:
                            if query_user and query_user != "所有用户":
                                user_filter_name = query_user
                        else:
                            user_filter_id = user_id

                        query_result = query_records(
                            conn,
                            user_id=user_filter_id,
                            username=user_filter_name,
                            start_date=datetime.combine(query_start, datetime.min.time()),
                            end_date=datetime.combine(query_end, datetime.max.time()),
                            min_score=query_min,
                            max_score=query_max,
                            keyword=keyword
                        )

                        if query_result.empty:
                            st.info("未找到匹配的记录")
                        else:
                            st.success(f"找到 {len(query_result)} 条记录")

                            # 显示结果
                            display_cols = ["record_date", "username", "mood_score", "mood_label",
                                            "activities", "notes", "tags"]
                            # 确保列存在
                            available_cols = [col for col in display_cols if col in query_result.columns]
                            display_df = query_result[available_cols].copy()

                            # 格式化日期
                            if "record_date" in display_df.columns:
                                display_df["record_date"] = pd.to_datetime(display_df["record_date"]).dt.strftime(
                                    "%Y-%m-%d %H:%M")

                            # 分页显示
                            page_size = 10
                            total_pages = max(1, len(display_df) // page_size + (
                                1 if len(display_df) % page_size > 0 else 0))
                            page = st.number_input("页码", min_value=1, max_value=total_pages, value=1)

                            start_idx = (page - 1) * page_size
                            end_idx = min(start_idx + page_size, len(display_df))

                            st.dataframe(
                                display_df.iloc[start_idx:end_idx],
                                use_container_width=True,
                                hide_index=True
                            )

                            # 导出选项
                            csv_data = query_result.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "📥 导出查询结果",
                                csv_data,
                                file_name=f"查询结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )

    # Tab 3: AI助手
    with tab3:
        st.subheader("🤖 AI情绪助手")

        # 检查AI服务状态
        if client is None:
            st.info("💡 请在侧边栏配置并连接AI服务")
        else:
            st.success("✅ AI助手已就绪")

            # AI功能选择
            ai_function = st.radio(
                "选择AI功能",
                ["情绪分析报告", "周度情绪总结", "个性化对话"],
                horizontal=True
            )

            if df.empty:
                st.warning("暂无记录可供分析")
            else:
                if ai_function == "情绪分析报告":
                    st.markdown("#### 📋 情绪综合分析")
                    st.caption("基于你的所有记录，AI将提供全面的情绪分析和建议")

                    if st.button("生成情绪分析报告", type="primary"):
                        with st.spinner("AI正在分析你的情绪数据..."):
                            result = ai_explain_mood(df)

                        st.markdown("---")
                        st.markdown("### 🧠 AI情绪分析报告")
                        st.markdown(result)

                        # 下载选项
                        st.download_button(
                            "📥 下载报告",
                            result,
                            file_name=f"情绪分析报告_{datetime.now().strftime('%Y%m%d')}.md",
                            mime="text/markdown"
                        )

                elif ai_function == "周度情绪总结":
                    st.markdown("#### 📅 本周情绪总结")
                    st.caption("分析最近7天的情绪变化和模式")

                    if st.button("生成周度总结", type="primary"):
                        with st.spinner("AI正在生成周报..."):
                            result = ai_generate_weekly_report(df)

                        st.markdown("---")
                        st.markdown("### 📊 本周情绪周报")
                        st.markdown(result)

                elif ai_function == "个性化对话":
                    st.markdown("#### 💬 与AI情绪教练对话")
                    st.caption("可以询问任何与情绪、压力、心理健康相关的问题")

                    user_question = st.text_area(
                        "你想聊什么？",
                        placeholder="例如：\n• 最近压力很大怎么办？\n• 如何保持积极心态？\n• 情绪低落时可以做些什么？",
                        height=100
                    )

                    if st.button("发送问题", type="primary") and user_question:
                        with st.spinner("AI正在思考..."):
                            # 构建更专业的系统提示
                            system_prompt = """你是一名专业的心理情绪教练，拥有丰富的情绪管理和心理健康知识。
    你的回答应该：
    1. 温暖、支持、非评判性
    2. 提供具体、可操作的建议
    3. 基于科学心理学原理
    4. 用普通人能理解的语言
    5. 鼓励积极改变和成长"""

                            messages = [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_question}
                            ]

                            response = ask_ai(messages, json_type=False)

                        st.markdown("---")
                        st.markdown("### 🤖 AI回复")
                        st.markdown(response)

    # Tab 4: 趋势
    with tab4:
        st.subheader("长期趋势分析")

        if len(df) < 7:
            st.info("需要更多记录来显示趋势分析")
        else:
            # 周趋势
            df["week"] = df["record_date"].dt.isocalendar().week
            weekly_avg = df.groupby("week")["mood_score"].mean()

            # 月趋势
            df["month"] = df["record_date"].dt.to_period("M").astype(str)
            monthly_avg = df.groupby("month")["mood_score"].mean()

            col1, col2 = st.columns(2)
            with col1:
                st.line_chart(weekly_avg)
                st.caption("周平均情绪趋势")
            with col2:
                st.line_chart(monthly_avg)
                st.caption("月平均情绪趋势")

            # 相关性分析
            if 'sleep_hours' in df.columns and 'stress_level' in df.columns:
                st.subheader("🔗 因素关联分析")
                numeric_cols = ["mood_score", "sleep_hours", "stress_level"]
                numeric_df = df[numeric_cols].dropna()

                if not numeric_df.empty:
                    corr_df = numeric_df.corr()
                    fig, ax = plt.subplots(figsize=(6, 4))
                    im = ax.imshow(corr_df, cmap="coolwarm", vmin=-1, vmax=1)
                    ax.set_xticks(range(len(corr_df.columns)))
                    ax.set_yticks(range(len(corr_df.columns)))
                    ax.set_xticklabels(corr_df.columns, rotation=45)
                    ax.set_yticklabels(corr_df.columns)

                    # 添加数值标签
                    for i in range(len(corr_df.columns)):
                        for j in range(len(corr_df.columns)):
                            text = ax.text(j, i, f"{corr_df.iloc[i, j]:.2f}",
                                           ha="center", va="center", color="black")

                    plt.colorbar(im)
                    st.pyplot(fig)
                    st.caption("情绪与其他因素的相关性（颜色越暖正相关越强）")

    # Tab 5: 设置
    with tab5:
        st.subheader("设置与导出")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📤 数据导出")

            # CSV导出
            if not df.empty:
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "导出CSV",
                    csv_data,
                    file_name=f"mood_records_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

            # 完整报告导出
            if st.button("生成完整报告", type="primary"):
                with st.spinner("生成报告中..."):
                    # 创建ZIP文件
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                        # 添加CSV
                        zip_file.writestr("mood_records.csv", df.to_csv(index=False))

                        # 添加总结文本
                        summary = f"""MoodPattern 数据报告
    生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    记录总数: {len(df)}
    """
                        if not df.empty and 'record_date' in df.columns:
                            summary += f"""时间范围: {df['record_date'].min().strftime('%Y-%m-%d')} 至 {df['record_date'].max().strftime('%Y-%m-%d')}
    平均情绪: {df['mood_score'].mean():.2f}/10
    情绪波动: {df['mood_score'].std():.2f}
    """
                        else:
                            summary += "暂无详细数据统计"

                        zip_file.writestr("summary.txt", summary)

                        # 添加图表
                        if not df.empty and 'record_date' in df.columns:
                            # 趋势图
                            fig, ax = plt.subplots(figsize=(10, 5))
                            df_sorted = df.sort_values("record_date")
                            ax.plot(df_sorted["record_date"], df_sorted["mood_score"], marker='o')
                            ax.set_title("情绪趋势图")
                            ax.set_xlabel("日期")
                            ax.set_ylabel("情绪分数")
                            ax.grid(True, alpha=0.3)

                            img_buffer = io.BytesIO()
                            fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                            zip_file.writestr("trend_chart.png", img_buffer.getvalue())
                            plt.close(fig)

                    zip_buffer.seek(0)

                    st.download_button(
                        "📥 下载报告ZIP",
                        zip_buffer.getvalue(),
                        file_name=f"mood_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip"
                    )

        with col2:
            st.subheader("⚠️ 数据管理")

            # 数据清理（只删除当前用户数据）
            if st.button("清除我的记录", type="secondary"):
                if st.checkbox("确认永久删除所有记录"):
                    conn.execute("DELETE FROM mood_records WHERE user_id = ?", (user_id,))
                    conn.commit()
                    st.success("所有记录已清除")
                    st.rerun()

            st.divider()

            # 关于
            st.subheader("ℹ️ 关于")
            st.markdown("""
                **MoodPattern — 情绪管理助手**

                一个专注情绪管理与心理健康的工具。

                功能特点：
                - 📝 情绪日记记录
                - 📊 数据可视化分析
                - 🤖 AI智能建议（基于讯飞星辰API）
                - 🔍 模式识别
                - 💾 数据库存储
                - 📤 数据导出

                **AI功能说明：**
                使用讯飞星辰API提供智能情绪分析和建议。
                请在侧边栏配置API信息后使用。
                """)

    # Tab 6: 管理
    with tab6:
        st.subheader("🔧 系统管理")

        # 管理员功能
        if 'admin_mode' in st.session_state and st.session_state.admin_mode:
            st.info("👑 管理员管理面板")

            # 用户统计
            st.subheader("📊 用户统计")
            if st.button("显示用户统计"):
                stats_df = get_user_stats(conn)
                if not stats_df.empty:
                    st.dataframe(stats_df, use_container_width=True)
                else:
                    st.info("暂无用户数据")

            # 批量操作功能
            st.subheader("🔄 批量操作")

            # 获取所有用户ID
            all_users = pd.read_sql("SELECT user_id, username FROM users", conn)

            if not all_users.empty:
                # 批量查询
                st.write("批量查询用户数据:")
                selected_user_ids = st.multiselect(
                    "选择用户",
                    options=[f"{row['user_id']} - {row['username']}" for _, row in all_users.iterrows()]
                )

                if selected_user_ids and st.button("执行批量查询"):
                    user_ids = [int(uid.split(" - ")[0]) for uid in selected_user_ids]

                    start_date = st.date_input("开始日期",
                                               value=datetime.now().date() - timedelta(days=30),
                                               key="batch_start")
                    end_date = st.date_input("结束日期",
                                             value=datetime.now().date(),
                                             key="batch_end")

                    results = batch_query_records(
                        conn,
                        user_ids=user_ids,
                        start_date=datetime.combine(start_date, datetime.min.time()),
                        end_date=datetime.combine(end_date, datetime.max.time()),
                        return_type="dataframe"
                    )

                    if results:
                        total_records = sum(len(df) for df in results.values() if not df.empty)
                        st.success(f"批量查询完成，共获取 {total_records} 条记录")

                        # 显示每个用户的结果摘要
                        for user_id, user_df in results.items():
                            if not user_df.empty:
                                username = all_users[all_users['user_id'] == user_id]['username'].iloc[0]
                                with st.expander(f"用户: {username} ({len(user_df)} 条记录)"):
                                    st.dataframe(user_df.head(10), use_container_width=True)

            # 记录管理
            st.subheader("📝 记录管理")
            col1, col2 = st.columns(2)

            with col1:
                # 更新记录
                update_id = st.number_input("更新记录ID", min_value=1, step=1)
                if update_id:
                    # 获取记录详情
                    record_df = pd.read_sql("""
                        SELECT mr.*, u.username 
                        FROM mood_records mr 
                        JOIN users u ON mr.user_id = u.user_id 
                        WHERE mr.id = ?
                        """, conn, params=(update_id,))

                    if not record_df.empty:
                        record = record_df.iloc[0]
                        st.write(f"当前记录：用户={record['username']}, 分数={record['mood_score']}")

                        new_mood = st.slider("新情绪值", 1, 10, record['mood_score'], key="update_mood")
                        new_notes = st.text_input("新备注", value=record['notes'] if record['notes'] else "",
                                                  key="update_notes")

                        if st.button("更新记录"):
                            conn.execute("""
                                UPDATE mood_records 
                                SET mood_score = ?, notes = ?, created_at = datetime('now')
                                WHERE id = ?
                                """, (new_mood, new_notes, update_id))
                            conn.commit()
                            st.success("记录更新成功！")
                            st.rerun()

            with col2:
                # 删除记录
                delete_id = st.number_input("删除记录ID", min_value=1, step=1, key="admin_delete")
                if st.button("删除记录"):
                    conn.execute("DELETE FROM mood_records WHERE id = ?", (delete_id,))
                    conn.commit()
                    st.success("记录删除成功！")
                    st.rerun()

            # 数据库维护
            st.subheader("🛠️ 数据库维护")
            if st.button("优化数据库"):
                conn.execute("VACUUM")
                conn.commit()
                st.success("数据库优化完成")

            if st.button("导出完整数据库"):
                # 导出整个数据库
                db_data = conn.cursor().execute("SELECT * FROM mood_records").fetchall()
                df_all = pd.DataFrame(db_data, columns=[desc[0] for desc in conn.cursor().description])
                csv_data = df_all.to_csv(index=False).encode('utf-8')

                st.download_button(
                    "📥 导出完整数据库",
                    csv_data,
                    file_name=f"mood_database_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        else:
            st.warning("🔒 仅管理员可访问此页面")

    # Tab 7: 分析中心（功能7）
    with tab7:
        st.subheader("📊 分析中心")

        # 用户行为分析
        st.header("👤 用户行为分析")

        if df.empty:
            st.info("暂无数据进行分析")
        else:
            # 活跃时间分析
            st.subheader("⏰ 活跃时间分析")
            if 'created_at' in df.columns:
                df['record_hour'] = pd.to_datetime(df['created_at']).dt.hour

                # 创建小时分布图
                hour_counts = df['record_hour'].value_counts().sort_index()

                # 使用matplotlib创建图表
                fig_hour, ax_hour = plt.subplots(figsize=(10, 4))
                ax_hour.bar(hour_counts.index, hour_counts.values, color='skyblue', edgecolor='black')
                ax_hour.set_xlabel("小时 (24小时制)")
                ax_hour.set_ylabel("记录数量")
                ax_hour.set_title("记录时间分布")
                ax_hour.set_xticks(range(0, 24, 2))
                ax_hour.grid(True, alpha=0.3)
                st.pyplot(fig_hour)

            # 行为统计
            col1, col2, col3 = st.columns(3)
            with col1:
                if 'record_date' in df.columns:
                    active_days = df['record_date'].nunique()
                    st.metric("活跃天数", active_days)
                else:
                    st.metric("活跃天数", "--")

            with col2:
                if 'record_date' in df.columns and active_days > 0:
                    avg_daily = len(df) / active_days
                    st.metric("日均记录", f"{avg_daily:.1f}")
                else:
                    st.metric("日均记录", "--")

            with col3:
                if 'created_at' in df.columns and len(df) > 1:
                    try:
                        time_diffs = pd.to_datetime(df['created_at']).diff().dt.total_seconds() / 3600
                        avg_interval = time_diffs.mean()
                        st.metric("平均间隔", f"{avg_interval:.1f}小时")
                    except:
                        st.metric("平均间隔", "--")
                else:
                    st.metric("平均间隔", "--")

            # 数据完整性检查
            st.subheader("🔍 数据完整性检查")
            data_hash = calculate_data_signature(df)

            col1, col2, col3 = st.columns(3)
            with col1:
                missing_sleep = df['sleep_hours'].isna().sum() if 'sleep_hours' in df.columns else 0
                st.metric("缺失睡眠数据", missing_sleep)

            with col2:
                missing_stress = df['stress_level'].isna().sum() if 'stress_level' in df.columns else 0
                st.metric("缺失压力数据", missing_stress)

            with col3:
                missing_activities = df['activities'].isna().sum() if 'activities' in df.columns else 0
                st.metric("缺失活动数据", missing_activities)

            # 数据签名
            with st.expander("🔐 数据完整性验证"):
                st.code(f"数据签名: {data_hash}")
                if st.button("验证数据完整性"):
                    # 重新计算哈希并验证
                    current_hash = calculate_data_signature(df)
                    if current_hash == data_hash:
                        st.success("✅ 数据完整性验证通过")
                    else:
                        st.error("❌ 数据完整性验证失败，数据可能已被修改")

            # 预测分析（简化版）
            st.subheader("🔮 情绪预测")
            if len(df) >= 10:
                try:
                    # 简单的线性回归预测
                    X = np.arange(len(df)).reshape(-1, 1)
                    y = df['mood_score'].values

                    # 使用numpy进行简单线性回归
                    x_mean = np.mean(X)
                    y_mean = np.mean(y)

                    numerator = np.sum((X - x_mean) * (y - y_mean))
                    denominator = np.sum((X - x_mean) ** 2)

                    if denominator != 0:
                        slope = numerator / denominator
                        intercept = y_mean - slope * x_mean

                        # 预测未来3天
                        future_X = np.arange(len(df), len(df) + 3).reshape(-1, 1)
                        predictions = slope * future_X + intercept

                        # 确保预测值在合理范围内
                        predictions = np.clip(predictions, 1, 10)

                        st.write("基于历史数据的趋势预测：")
                        for i, pred in enumerate(predictions.flatten(), 1):
                            st.write(f"未来第{i}天预测情绪: {pred:.1f}/10")

                            # 给出简单建议
                            if pred >= 8:
                                st.info("预计情绪良好，继续保持！")
                            elif pred <= 4:
                                st.warning("预计情绪较低，建议提前做好心理准备")
                    else:
                        st.info("无法计算趋势，数据可能过于集中")

                except Exception as e:
                    st.error(f"预测分析出错: {e}")
            else:
                st.info("需要至少10条记录进行预测分析")

            # 数据质量评分
            st.subheader("📈 数据质量评分")
            quality_score = 100

            # 检查缺失数据
            total_fields = len(df) * 3  # 主要字段数
            missing_fields = 0

            for field in ['sleep_hours', 'stress_level', 'activities']:
                if field in df.columns:
                    missing_fields += df[field].isna().sum()

            if total_fields > 0:
                completeness = 100 - (missing_fields / total_fields * 100)
            else:
                completeness = 0

            # 检查记录频率
            if 'created_at' in df.columns and len(df) > 1:
                try:
                    time_diffs = pd.to_datetime(df['created_at']).diff().dt.total_seconds() / 86400  # 转换为天
                    avg_frequency = time_diffs.mean()

                    if avg_frequency <= 2:  # 平均每2天记录一次
                        frequency_score = 100
                    elif avg_frequency <= 7:  # 平均每周记录一次
                        frequency_score = 70
                    else:
                        frequency_score = 30
                except:
                    frequency_score = 50
            else:
                frequency_score = 50

            # 计算综合评分
            quality_score = (completeness + frequency_score) / 2

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("数据完整性", f"{completeness:.0f}%")

            with col2:
                if 'created_at' in df.columns and len(df) > 1:
                    try:
                        time_diffs = pd.to_datetime(df['created_at']).diff().dt.total_seconds() / 86400
                        avg_frequency = time_diffs.mean()
                        st.metric("记录频率", f"{avg_frequency:.1f}天/次")
                    except:
                        st.metric("记录频率", "--")
                else:
                    st.metric("记录频率", "--")

            with col3:
                st.metric("数据质量", f"{quality_score:.0f}/100")

                if quality_score >= 80:
                    st.success("数据质量优秀！")
                elif quality_score >= 60:
                    st.info("数据质量良好")
                elif quality_score >= 40:
                    st.warning("数据质量一般")
                else:
                    st.error("数据质量有待提高")

    # Tab 8: 安全中心（功能8）
    with tab8:
        st.subheader("🔐 安全中心")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🛡️ 数据保护")

            # 数据加密
            st.write("**数据加密设置**")
            encryption_key = st.text_input("加密密钥", type="password",
                                           help="用于加密敏感数据的密钥")

            if encryption_key:
                test_data = st.text_area("测试加密数据",
                                         placeholder="输入要加密的测试数据")
                if test_data and st.button("测试加密"):
                    encrypted = encrypt_sensitive_field(test_data, encryption_key)
                    st.code(f"加密结果: {encrypted}")

            # 数据完整性验证
            st.write("**数据完整性**")
            if st.button("验证所有数据完整性"):
                # 计算当前数据的哈希
                current_hash = calculate_data_signature(df)

                # 从数据库获取原始哈希（这里简化处理，实际应从备份或日志中获取）
                try:
                    backup_log_path = BACKUP_DIR / "backup_log.json"
                    if backup_log_path.exists():
                        with open(backup_log_path, 'r', encoding='utf-8') as f:
                            backup_log = json.load(f)

                        if backup_log:
                            latest_backup = backup_log[-1]
                            original_hash = latest_backup.get('data_hash', '')

                            if original_hash and verify_data_integrity(original_hash, df):
                                st.success("✅ 数据完整性验证通过")
                            else:
                                st.error("❌ 数据完整性验证失败")
                        else:
                            st.info("暂无备份记录")
                    else:
                        st.info("暂无备份文件")
                except Exception as e:
                    st.error(f"验证出错: {e}")

        with col2:
            st.subheader("💾 备份管理")

            # 创建备份
            backup_name = st.text_input("备份名称",
                                        value=f"backup_{datetime.now().strftime('%Y%m%d_%H%M')}")

            if st.button("创建备份", type="primary"):
                with st.spinner("正在创建备份..."):
                    backup_info = create_backup_with_verification(conn, backup_name)

                    if backup_info.get('verification_passed'):
                        st.success(f"✅ 备份创建成功！")
                        st.json(backup_info)
                    else:
                        st.error("备份创建失败或验证未通过")

            # 查看备份列表
            st.write("**备份列表**")
            try:
                backup_log_path = BACKUP_DIR / "backup_log.json"

                if backup_log_path.exists():
                    with open(backup_log_path, 'r', encoding='utf-8') as f:
                        backup_log = json.load(f)

                    if backup_log:
                        for i, backup in enumerate(reversed(backup_log[-5:]), 1):

                            with st.expander(f"备份 {i}: {backup.get('name', '未知')}"):
                                st.write(f"时间: {backup.get('timestamp', '未知')}")
                                st.write(f"记录数: {backup.get('record_count', 0)}")
                                st.write(f"验证状态: {'✅ 通过' if backup.get('verification_passed') else '❌ 失败'}")

                                # 放到 expander 内（每个备份都有按钮）
                                backup_path = backup.get('backup_path', '')
                                if backup_path and st.button(f"恢复此备份", key=f"restore_{i}"):
                                    if restore_from_backup(backup_path, conn):
                                        st.success("恢复成功！请刷新页面查看最新数据")
                                        st.rerun()
                                    else:
                                        st.error("恢复失败")

                    else:
                        st.info("暂无备份记录")

                else:
                    st.info("暂无备份文件")

            except Exception as e:
                st.error(f"加载备份时出错：{e}")


    # 最后关闭数据库连接
    st.divider()
    if st.button("关闭数据库连接"):
        conn.close()
        st.success("数据库连接已关闭")


if __name__ == "__main__":
    main()