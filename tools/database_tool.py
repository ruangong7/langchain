"""数据库查询工具"""
import pymysql
from typing import List
import logging
from config import MYSQL_HOST, MYSQL_PORT, MYSQL_DATABASE, MYSQL_USER, MYSQL_PASSWORD

logger = logging.getLogger(__name__)

class DatabaseTool:
    """数据库查询工具类"""
    
    def __init__(self):
        self.connection = None
        self._connect()
    
    def _connect(self):
        """连接数据库"""
        try:
            self.connection = pymysql.connect(
                host=MYSQL_HOST,
                port=MYSQL_PORT,
                user=MYSQL_USER,
                password=MYSQL_PASSWORD,
                database=MYSQL_DATABASE,
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
            logger.info("数据库连接成功")
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            raise
    
    def query_real_drug_database(self, drug_name: str) -> str:
        """查询real_drug数据库，获取药物的详细信息"""
        try:
            with self.connection.cursor() as cursor:
                sql = "SELECT * FROM real_drug WHERE drug_name LIKE %s"
                cursor.execute(sql, (f"%{drug_name}%",))
                results = cursor.fetchall()
                
                if not results:
                    return f"未找到药物 {drug_name} 的相关信息"
                
                # 格式化结果
                result_text = f"找到 {len(results)} 条关于 {drug_name} 的记录：\n\n"
                for i, row in enumerate(results, 1):
                    result_text += f"记录 {i}:\n"
                    for key, value in row.items():
                        if value:
                            result_text += f"  {key}: {value}\n"
                    result_text += "\n"
                
                logger.info(f"查询real_drug数据库成功: {drug_name}")
                return result_text
                
        except Exception as e:
            logger.error(f"查询real_drug数据库失败: {e}")
            return f"查询real_drug数据库失败: {str(e)}"
    
    def query_joint_data(self, question: str) -> str:
        """联合查询：先查yinshi获取用户用药，再以药物名称为外键查询real_drug"""
        try:
            with self.connection.cursor() as cursor:
                # 先查询yinshi表
                sql_yinshi = "SELECT * FROM yinshi WHERE question LIKE %s OR answer LIKE %s"
                cursor.execute(sql_yinshi, (f"%{question}%", f"%{question}%"))
                yinshi_results = cursor.fetchall()
                
                result_text = f"联合查询结果（问题：{question}）：\n\n"
                
                if yinshi_results:
                    result_text += "=== 用户用药信息（yinshi表）===\n"
                    for row in yinshi_results:
                        drug_name = row.get('drug_name', '')
                        result_text += f"药物名称: {drug_name}\n"
                        for key, value in row.items():
                            if key != 'drug_name' and value:
                                result_text += f"  {key}: {value}\n"
                        result_text += "\n"
                        
                        # 如果有关联的药物名称，查询real_drug表
                        if drug_name:
                            sql_real = "SELECT * FROM real_drug WHERE drug_name LIKE %s"
                            cursor.execute(sql_real, (f"%{drug_name}%",))
                            real_results = cursor.fetchall()
                            
                            if real_results:
                                result_text += f"=== {drug_name} 的详细信息（real_drug表）===\n"
                                for real_row in real_results:
                                    for key, value in real_row.items():
                                        if value:
                                            result_text += f"  {key}: {value}\n"
                                result_text += "\n"
                else:
                    result_text += "未找到相关的用户用药信息\n"
                
                logger.info(f"联合查询成功: {question}")
                return result_text
                
        except Exception as e:
            logger.error(f"联合查询失败: {e}")
            return f"联合查询失败: {str(e)}"

    def load_drug_lexicon(self, refresh: bool = False) -> List[str]:
        """加载 real_drug 表中的药品名词表，用于前置实体扫描。"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT DISTINCT drug_name
                    FROM real_drug
                    WHERE drug_name IS NOT NULL AND drug_name <> ''
                    """
                )
                rows = cursor.fetchall()
                names = {
                    str(row.get("drug_name", "")).strip()
                    for row in rows
                    if row.get("drug_name")
                }
                lexicon = sorted(names, key=len, reverse=True)
                logger.info("药品词表加载完成: %d", len(lexicon))
                return lexicon
        except Exception as e:
            logger.error(f"加载药品词表失败: {e}")
            return []
    
    def close(self):
        """关闭数据库连接"""
        if self.connection:
            self.connection.close()
            logger.info("数据库连接已关闭")
